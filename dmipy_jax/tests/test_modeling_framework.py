
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel, CompartmentModel

# Dummy Model 1
class DummyStick(CompartmentModel):
    parameter_names = ('diffusivity', 'orientation')
    parameter_cardinality = {'diffusivity': 1, 'orientation': 2}
    parameter_ranges = {'diffusivity': (0.1, 3.0), 'orientation': ([0, jnp.pi], [-jnp.pi, jnp.pi])}
    
    def __init__(self):
        pass
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        return jnp.ones_like(bvals) * 0.5 # Dummy signal

# Dummy Model 2 (Ball)
class DummyBall(CompartmentModel):
    parameter_names = ('diffusivity',)
    parameter_cardinality = {'diffusivity': 1}
    parameter_ranges = {'diffusivity': (0.1, 3.0)}
    
    def __init__(self):
        pass
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        # bvecs = gradient_directions
        return jnp.exp(-bvals * kwargs['diffusivity'])

def test_mcm_construction():
    m1 = DummyStick()
    m2 = DummyBall()
    
    mcm = JaxMultiCompartmentModel([m1, m2])
    
    # Check parameters
    # m1: diffusivity, orientation (mu)
    # m2: diffusivity
    # Collision on 'diffusivity' -> diffusivity_1, diffusivity_2
    
    assert 'diffusivity' in mcm.parameter_names
    # Actually explicit collision handling in code appends _i+1?
    # Let's check logic: if pname in self.parameter_names -> rename.
    # m1 inserts 'diffusivity'.
    # m2 inserts 'diffusivity' -> collision -> 'diffusivity_2'.
    
    assert 'diffusivity' in mcm.parameter_names
    assert 'diffusivity_2' in mcm.parameter_names
    
    # Partial fractions
    assert 'partial_volume_0' in mcm.parameter_names
    assert 'partial_volume_1' in mcm.parameter_names

def test_mcm_call():
    m1 = DummyStick()
    m2 = DummyBall()
    mcm = JaxMultiCompartmentModel([m1, m2])
    
    # Mock params
    params = {
        'diffusivity': 1.0,
        'orientation': jnp.array([0.0, 0.0]),
        'diffusivity_2': 2.0,
        'partial_volume_0': 0.6,
        'partial_volume_1': 0.4
    }
    
    N = 10
    
    class MockAcq:
        bvals = jnp.ones(N)
        bvalues = bvals # Alias
        bvecs = jnp.zeros((N, 3))
        gradient_directions = bvecs # Alias
        delta = 0.01
        Delta = 0.03
        TE = 0.05
        
    acq = MockAcq()
    
    # Run
    signal = mcm(params, acq)
    
    assert signal.shape == (N,)
    # Expected: 0.6 * 0.5 + 0.4 * exp(-1 * 2.0)
    expected = 0.6 * 0.5 + 0.4 * jnp.exp(-2.0)
    assert jnp.allclose(signal, expected)

def test_mcm_fit_interface():
    # Test internal packing
    pass
