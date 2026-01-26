
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any, NamedTuple
from dmipy_jax.models.mcdespot import McDESPOT, McDESPOTParameters
from dmipy_jax.models.qmt import qMT_SPGR

class JointInversionParameters(NamedTuple):
    # Volume Fractions (relative to Total)
    # Total = V_mw + V_ie + V_mm = 1
    # We parameterize freely, then normalize?
    # Or typically:
    # f_mw_water: MWF (MW / (MW+IE)) - This is what clinicians care about.
    # f_mm_total: MPF (MM / Total)
    
    f_mw_water: float # MW / (MW + IE)
    f_mm_total: float # MM / (MW + IE + MM)
    
    # Relaxation - Water Pools
    T1_mw: float  # ms
    T1_ie: float  # ms
    T2_mw: float  # ms
    T2_ie: float  # ms
    
    # Relaxation - Macromolecular
    T1_mm: float  # ms (Often fixed ~ 1000)
    T2_mm: float  # s (Very short, ~10us)
    
    # Exchange
    k_m_w: float  # Exchange MM <-> Water (s^-1)
    # k_mw_ie: float # Exchange MW <-> IE (ignored in non-exchanging mcDESPOT)
    
    # Scaling
    S0: float

class JointInversionModel(eqx.Module):
    """
    Joint Inversion of mcDESPOT and qMT data.
    
    Combines:
    1. 2-Pool Non-Exchanging mcDESPOT (Myelin Water vs IE Water)
    2. 2-Pool Exchanging qMT (Macromolecules vs Free Water)
    
    Coupling:
    - The 'Free Water' in qMT is treated as a weighted average of MW and IE pools from mcDESPOT.
    - Alternatively, simpler: qMT sees (MW+IE) as one pool.
    - Shared parameters: T1_ie (dominant water T1), S0.
    """
    
    mcdespot: McDESPOT = eqx.field(default_factory=McDESPOT)
    qmt_model: qMT_SPGR = eqx.field(default_factory=qMT_SPGR)
    
    def __call__(self, 
                 params: JointInversionParameters, 
                 proto: Dict[str, Any]) -> jax.Array:
        
        # 1. mcDESPOT Signal (SPGR + SSFP)
        # --------------------------------
        # mcDESPOT sees Water pools (MW + IE).
        # It assumes MM is invisible.
        # But we must account for the fact that M_water = (1 - f_mm_total) * M_total.
        # So mcDESPOT signal scales by (1 - f_mm_total).
        
        scale_water = (1.0 - params.f_mm_total) * params.S0
        
        # MW / (MW+IE) is directly f_mw_water
        mc_params = McDESPOTParameters(
            f_myelin=params.f_mw_water,
            T1_myelin=params.T1_mw,
            T2_myelin=params.T2_mw,
            T1_ie=params.T1_ie,
            T2_ie=params.T2_ie,
            off_resonance=0.0 # Assuming corrected or fit separately
        )
        
        # SPGR
        spgr_fa = proto['spgr_fa'] # degrees
        spgr_rad = jnp.radians(spgr_fa)
        spgr_tr = proto['spgr_tr'] # s
        
        # Map over flip angles
        def run_spgr(alpha):
            # Input TR in ms for mcDESPOT (standard convention in library)
            return self.mcdespot(mc_params, 'SPGR', spgr_tr * 1000.0, alpha)
            
        S_spgr = jax.vmap(run_spgr)(spgr_rad) * scale_water
        
        # SSFP
        ssfp_fa = proto['ssfp_fa'] # degrees
        ssfp_rad = jnp.radians(ssfp_fa)
        ssfp_tr = proto['ssfp_tr'] # s
        ssfp_dh = jnp.radians(proto['ssfp_phase']) # phase cycling
        
        def run_ssfp(alpha, dphi):
            return self.mcdespot(mc_params, 'SSFP', ssfp_tr * 1000.0, alpha, phase_cycling=dphi)
            
        S_ssfp = jax.vmap(run_ssfp)(ssfp_rad, ssfp_dh) * scale_water
        
        
        # 2. qMT Signal (SPGR)
        # --------------------
        # qMT sees Free Water (MW+IE) vs Macromolecules (MM).
        # We need "Effective" T1_free, T2_free for the single "Water" pool in qMT model.
        # T1_water_eff = f_mw * T1_mw + (1-f_mw) * T1_ie ? Or rate averaging?
        # Fast exchange limit for water pools -> Rate average: R1_eff = f R1a + (1-f) R1b.
        # Slow exchange -> Signal sum.
        # qMT usually assumes internal water equilibrium is fast (relative to MM exchange).
        # Let's use weighted rate average for robust "Free Pool" properties.
        
        f_mw = params.f_mw_water
        R1_mw = 1.0 / (params.T1_mw * 1e-3 + 1e-9) # s^-1
        R1_ie = 1.0 / (params.T1_ie * 1e-3 + 1e-9)
        R1_free_eff = f_mw * R1_mw + (1-f_mw) * R1_ie
        
        T2_mw_s = params.T2_mw * 1e-3
        T2_ie_s = params.T2_ie * 1e-3
        R2_free_eff = f_mw * (1.0/T2_mw_s) + (1-f_mw) * (1.0/T2_ie_s)
        T2_free_eff = 1.0 / R2_free_eff
        
        # Run qMT
        offsets = proto['qmt_offsets']
        sat_fa = proto['qmt_sat_fa']
        ex_fa = proto['qmt_ex_fa']
        tr_qmt = proto['qmt_tr'] # s
        
        # qMT Model uses f = MPF = f_mm_total.
        # Parameters
        # f, k_mf, R1_f, R1_m, T2_f, T2_m
        
        def run_qmt(off, sfa, efa, tr):
            return self.qmt_model.forward(
                f=params.f_mm_total,
                k_mf=params.k_m_w,
                R1_f=R1_free_eff,
                R1_m=1.0 / (params.T1_mm * 1e-3),
                T2_f=T2_free_eff,
                T2_m=params.T2_mm,
                tr=tr,
                exc_fa=efa,
                mt_fa=sfa,
                mt_offset=off
            )
            
        S_qmt = jax.vmap(run_qmt)(offsets, sat_fa, ex_fa, tr_qmt) * params.S0 
        # Note: qMT uses Total S0 (M0_total), because MPF partitions it.
        
        
        # Concatenate
        return jnp.concatenate([S_spgr, S_ssfp, S_qmt])

