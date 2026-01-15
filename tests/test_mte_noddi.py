
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.mte_noddi import MTE_NODDI


def test_mte_noddi_run():
    # 1. Setup Acquisition with Multi-TE
    bvals = jnp.array([0.0, 1000.0, 1000.0, 2000.0, 2000.0])
    bvecs = jnp.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    # Varying TE for the same b-values to simulate multi-echo
    echo_time = jnp.array([0.05, 0.05, 0.08, 0.05, 0.08]) # s
    
    acq = JaxAcquisition(
        bvalues=bvals,
        gradient_directions=bvecs,
        echo_time=echo_time
    )
    
    # 2. Instantiate Model
    model = MTE_NODDI()
    
    # 3. Define Parameters
    # Using physical parameters directly
    params = {
        'f_iso': 0.1,
        'v_ic': 0.5,
        't2_iso': 0.500, # 500ms
        't2_ic': 0.080,  # 80ms
        't2_ec': 0.060,  # 60ms
        'mu': jnp.array([1.57, 0.0]), # theta=pi/2, phi=0 (along x)
        'd_par': 1.7e-9, # standard
        'd_iso': 3.0e-9  # standard
    }
    
    # 4. Predict
    signal = model(acq, **params)
    
    # 5. Check Basic Properties
    assert signal.shape == (5,)
    assert jnp.all(signal > 0), "Signal must be positive"
    assert jnp.all(signal <= 1.0), "Signal with T2 decay (and f summing to 1) should be <= 1 (assuming S0=1 implicit)"
    
    # Check T2 decay effect
    # Index 1 and 2 are b=1000, different TE (0.05 vs 0.08)
    # Signal at TE=0.08 (idx 2) should be LOWER than TE=0.05 (idx 1)
    # provided bvecs are different... wait idx 1 is (1,0,0), idx 2 is (0,1,0).
    # Since mu is (1,0,0) (x-axis), idx 1 is parallel (max attenuation), idx 2 is perp (less attenuation).
    # So idx 2 might be higher due to diffusion despite T2 decay.
    # Let's compare indices with SAME diffusion weighting but different TE if possible.
    # I don't have exact duplicate b/g in my mockup for that.
    # Let's create a better acquisition for this check.
    
    bvals_2 = jnp.array([1000.0, 1000.0])
    bvecs_2 = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    te_2 = jnp.array([0.05, 0.10])
    acq_2 = JaxAcquisition(bvalues=bvals_2, gradient_directions=bvecs_2, echo_time=te_2)
    
    signal_2 = model(acq_2, **params)
    print(f"Signal TE=50ms: {signal_2[0]}, TE=100ms: {signal_2[1]}")
    assert signal_2[1] < signal_2[0], "Higher TE should yield lower signal"

    print("MTE-NODDI Test Passed!")
    
if __name__ == "__main__":
    test_mte_noddi_run()
