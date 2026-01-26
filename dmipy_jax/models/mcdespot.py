
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Union, NamedTuple
from dmipy_jax.models.epg import JAXEPG

class McDESPOTParameters(NamedTuple):
    """
    Standard parameters for 2-pool McDESPOT.
    """
    f_myelin: float   # Myelin Water Fraction (0.0 to 1.0)
    T1_myelin: float  # ms
    T2_myelin: float  # ms
    T1_ie: float      # Intra/Extra-cellular T1 (ms)
    T2_ie: float      # Intra/Extra-cellular T2 (ms)
    off_resonance: float = 0.0 # B0 offset in radians per TR

class McDESPOT(eqx.Module):
    """
    Multi-Component Driven Equilibrium Single Pulse Observation of T1/T2.
    
    This implementation assumes a 2-Pool Non-Exchanging model:
    1. Myelin Water (Short T2)
    2. Intra/Extra-cellular Water (Long T2)
    
    Signal = f_m * S_myelin + (1 - f_m) * S_ie
    """
    
    def __call__(self, 
                 params: McDESPOTParameters, 
                 sequence_type: str, 
                 TR: float, 
                 alpha: float, 
                 phase_cycling: float = 0.0) -> jax.Array:
        """
        Forward model for a single voxel.
        
        Args:
            params: Tissue parameters.
            sequence_type: 'SPGR' or 'SSFP'.
            TR: Repetition Time (ms).
            alpha: Flip Angle (radians).
            phase_cycling: RF phase increment per TR (radians).
            
        Returns:
            Computed Signal Magnitude.
        """
        
        if sequence_type == 'SPGR':
            # Run SPGR for both pools
            S_m = JAXEPG.simulate_spgr(params.T1_myelin, params.T2_myelin, TR, alpha)
            S_ie = JAXEPG.simulate_spgr(params.T1_ie, params.T2_ie, TR, alpha)
            
        elif sequence_type == 'SSFP':
            # Run bSSFP for both pools
            # Note: T1_myelin is short, T2_myelin is short (~10-20ms)
            s_m_fn = lambda: JAXEPG.simulate_bssfp(
                params.T1_myelin, params.T2_myelin, TR, alpha, 
                off_resonance=params.off_resonance, phase_cycling=phase_cycling
            )
            s_ie_fn = lambda: JAXEPG.simulate_bssfp(
                params.T1_ie, params.T2_ie, TR, alpha, 
                off_resonance=params.off_resonance, phase_cycling=phase_cycling
            )
            
            S_m = s_m_fn()
            S_ie = s_ie_fn()
            
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}. Use 'SPGR' or 'SSFP'.")
            
        # Combine Signals
        # TODO: Complex summation? Usually mcDESPOT fits magnitude data.
        # But if phase cycling is used, we might want complex sum if data is complex?
        # Standard mcDESPOT fits magnitude images.
        # However, interference between pools happens in complex domain IF they have different phases.
        # EPG returns Magnitude for SPGR, Complex for SSFP?
        # Wait, my EPG `simulate_spgr` returns Magnitude (jnp.abs).
        # My EPG `simulate_bssfp` returns Complex signal (F0).
        # Let's verify `epg.py`.
        
        # Checking epg.py ...
        # simulate_spgr -> returns jnp.abs(signals[-1])
        # simulate_bssfp -> returns signals[-1] (Complex)
        
        # Consistency Fix:
        # If simulation is magnitude (SPGR), we sum magnitudes (assuming incoherent averaging / spoiling).
        # If simulation is complex (SSFP), we sum complex then take magnitude?
        
        if sequence_type == 'SPGR':
            # SPGR is spoiled, phases are random/cancelled. Sum of magnitudes is approximation?
            # Actually, multi-component SPGR implies partial volume.
            # If they are in the same voxel, they add vectorially.
            # But spoiling destroys transverse coherence *between* TRs.
            # Within a TR (at TE=0), they are coherent.
            # So complex sum is correct if we knew the phase?
            # But usually SPGR is modeled as Magnitude sum because of spoiling?
            # Let's stick to simple scalar weighting for now, considering standard practice.
            S_total = params.f_myelin * S_m + (1.0 - params.f_myelin) * S_ie
            
        else:
            # SSFP is coherent. Start with complex sum.
            S_total_complex = params.f_myelin * S_m + (1.0 - params.f_myelin) * S_ie
            S_total = jnp.abs(S_total_complex)
            
        return S_total

