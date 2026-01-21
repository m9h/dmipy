import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from dmipy_jax.optimization.acquisition import AcquisitionProtocol, CramerRaoBound, optimize_acquisition
from dmipy_jax.signal_models.sphere_models import SphereGPD

def test_acquisition_protocol_shapes():
    n_measurements = 30
    protocol_model = AcquisitionProtocol(n_measurements=n_measurements)
    
    scheme = protocol_model()
    
    assert scheme.bvalues.shape == (n_measurements,)
    assert scheme.gradient_directions.shape == (n_measurements, 3)
    assert scheme.echo_time.shape == (n_measurements,)
    assert scheme.delta.shape == (n_measurements,)
    assert scheme.Delta.shape == (n_measurements,)
    
    # Check bounds
    assert jnp.all(scheme.bvalues >= 0)
    assert jnp.all(scheme.bvalues <= protocol_model.max_b_value)
    assert jnp.all(scheme.echo_time >= protocol_model.min_TE)
    assert jnp.all(scheme.echo_time <= protocol_model.max_TE)
    
    # Check direction normalization
    norms = jnp.linalg.norm(scheme.gradient_directions, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)

def test_crb_computation():
    # Setup
    model = SphereGPD()
    # Parameters for the sphere: diameter=10um, D=3e-9 m^2/s
    target_params = {'diameter': 10e-6, 'diffusion_constant': 3e-9}
    
    protocol_model = AcquisitionProtocol(n_measurements=20)
    scheme = protocol_model()
    
    # Compute CRB
    crb_calc = CramerRaoBound(model)
    variances = crb_calc(target_params, scheme)
    
    assert variances.shape == (2,) # 2 parameters
    assert jnp.all(variances > 0) # Variances must be positive

def test_optimization_loop():
    # Setup
    model = SphereGPD()
    target_params = {'diameter': 6e-6, 'diffusion_constant': 2e-9}
    
    # Optimize
    optimized_protocol = optimize_acquisition(
        tissue_model=model,
        target_params=target_params,
        n_measurements=10,
        max_b_value=3000.0,
        seed=42
    )
    
    # Check that we got a valid protocol object back
    assert isinstance(optimized_protocol, AcquisitionProtocol)
    
    scheme = optimized_protocol()
    # Basic sanity checks on the result
    assert jnp.all(scheme.bvalues >= 0)

def test_optimization_improves_crb():
    # Setup
    model = SphereGPD()
    target_params = {'diameter': 8e-6, 'diffusion_constant': 2.5e-9}
    
    # 1. Random Protocol
    random_protocol_model = AcquisitionProtocol(
         n_measurements=30,
         key=jax.random.PRNGKey(123)
    )
    random_scheme = random_protocol_model()
    crb_calc = CramerRaoBound(model)
    random_variances = crb_calc(target_params, random_scheme)
    random_loss = jnp.mean(jnp.log(random_variances))
    
    # 2. Optimized Protocol
    # We use a known seed to ensure fair comparison if needed, but optimize_acquisition initializes its own
    optimized_protocol_model = optimize_acquisition(
        tissue_model=model,
        target_params=target_params,
        n_measurements=30,
        seed=123
    )
    optimized_scheme = optimized_protocol_model()
    optimized_variances = crb_calc(target_params, optimized_scheme)
    optimized_loss = jnp.mean(jnp.log(optimized_variances))
    
    print(f"Random Loss: {random_loss}, Optimized Loss: {optimized_loss}")
    
    # Assert improvement
    assert optimized_loss < random_loss
