
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.models.mcdespot import McDESPOT, McDESPOTParameters

def test_mcdespot_forward():
    """
    Test the Forward Model call for McDESPOT.
    """
    model = McDESPOT()
    
    # 1. Define Parameters representing White Matter
    # Myelin: Short In T2 (10-20ms)
    # IE: Long T2 (80-100ms)
    params = McDESPOTParameters(
        f_myelin=0.2,
        T1_myelin=500.0,
        T2_myelin=15.0,
        T1_ie=1000.0,
        T2_ie=90.0,
        off_resonance=0.0
    )
    
    TR = 5.0 # ms
    alpha = jnp.deg2rad(15.0)
    
    # 2. Test SPGR
    sig_spgr = model(params, 'SPGR', TR, alpha)
    print(f"SPGR Signal: {sig_spgr:.5f}")
    assert sig_spgr > 0
    assert jnp.isfinite(sig_spgr)
    
    # 3. Test SSFP (Phase Cycle 0)
    sig_ssfp_0 = model(params, 'SSFP', TR, alpha, phase_cycling=0.0)
    print(f"SSFP (0 deg) Signal: {sig_ssfp_0:.5f}")
    assert jnp.isfinite(sig_ssfp_0)

    # 4. Test SSFP (Phase Cycle 180 - should be different due to banding)
    sig_ssfp_180 = model(params, 'SSFP', TR, alpha, phase_cycling=jnp.pi)
    print(f"SSFP (180 deg) Signal: {sig_ssfp_180:.5f}")
    
    # With Off-Resonance=0 (On Resonance), 0 and 180 might behave similarly 
    # if perfectly balanced?
    # Actually bSSFP on-resonance is high signal. 180 cycle shifts the band.
    # At 0Hz off-resonance:
    # PC=0 -> On Resonance -> High Signal
    # PC=180 -> Effective Beta=180 -> Band -> Low Signal
    
    # 4. Test SSFP behavior
    # Phase Cycle 180 (Alternating RF: 0, 180, 0...) usually places the Pass Band ON RESONANCE.
    # Phase Cycle 0 (Constant RF: 0, 0, 0...) usually places the Dark Band ON RESONANCE.
    # Therefore, for off_resonance=0, we expect Signal(180) > Signal(0).
    
    assert sig_ssfp_180 > sig_ssfp_0 
    print(f"bSSFP Profile check passed: Alternating Phase ({sig_ssfp_180:.4f}) > Constant Phase ({sig_ssfp_0:.4f})")

