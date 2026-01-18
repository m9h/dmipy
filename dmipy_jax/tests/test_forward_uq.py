
import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.core.uncertainty_model import propagate_uncertainty, Uniform, Normal

def test_forward_uq_uniform_linear():
    # Model: y(x) = 2x + 1
    # x ~ U[0, 1]
    # Analytical:
    # Mean[x] = 0.5. Mean[y] = 2(0.5) + 1 = 2.0.
    # Var[x] = (1-0)^2 / 12 = 1/12.
    # Var[y] = 4 * Var[x] = 4/12 = 1/3.
    # Std[y] = sqrt(1/3) approx 0.57735
    
    def model_func(params, acq):
        # params[0] is x
        return params[0] * 2.0 + 1.0
        
    uncertain_params = {'x': Uniform(0.0, 1.0)}
    fixed_params = {}
    parameter_names = ['x']
    acquisition = None
    
    mean, std = propagate_uncertainty(
        model_func, fixed_params, acquisition, parameter_names, uncertain_params, order=2
    )
    
    # Check scalar output (since propagate_uncertainty returns array of size num_meas=1)
    # Actually our model returns scalar, so vmap in propagate makes it (N, 1)? 
    # Or (N,). propagate output is array (n_meas,).
    # model_func returns scalar here?
    # propagate expects model to return array-like?
    # Let's adjust model to return array of size 1
    def model_func_array(params, acq):
        return jnp.array([params[0] * 2.0 + 1.0])
        
    mean, std = propagate_uncertainty(
        model_func_array, fixed_params, acquisition, parameter_names, uncertain_params, order=2
    )
    
    assert jnp.isscalar(mean[0]) or mean[0].shape == ()
    assert jnp.allclose(mean[0], 2.0, atol=1e-2)
    assert jnp.allclose(std[0], jnp.sqrt(1.0/3.0), atol=1e-2)

def test_forward_uq_normal_quadratic():
    # Model: y(x) = x^2
    # x ~ N(0, 1)
    # Analytical:
    # Mean[y] = E[x^2] = Var[x] + E[x]^2 = 1 + 0 = 1.
    # Var[y] = E[x^4] - E[x^2]^2 = 3 - 1 = 2.
    # Std[y] = sqrt(2).
    
    def model_func(params, acq):
        return jnp.array([params[0]**2])
        
    uncertain_params = {'x': Normal(0.0, 1.0)}
    fixed_params = {}
    parameter_names = ['x']
    acquisition = None
    
    mean, std = propagate_uncertainty(
        model_func, fixed_params, acquisition, parameter_names, uncertain_params, order=3
    )
    
    assert jnp.allclose(mean[0], 1.0, atol=1e-2)
    assert jnp.allclose(std[0], jnp.sqrt(2.0), atol=1e-2)

def test_forward_uq_mixed():
    # Model: y = x1 + x2
    # x1 ~ U[0, 1]
    # x2 ~ N(0, 1)
    # Mean = 0.5 + 0 = 0.5
    # Var = 1/12 + 1 = 13/12
    
    def model_func(params, acq):
        # params: [x1, x2]
        return jnp.array([params[0] + params[1]])
        
    uncertain = {'x1': Uniform(0.0, 1.0), 'x2': Normal(0.0, 1.0)}
    params = {} 
    
    mean, std = propagate_uncertainty(
        model_func, params, None, ['x1', 'x2'], uncertain, order=2
    )
    
    assert jnp.allclose(mean[0], 0.5, atol=1e-2)
    assert jnp.allclose(std[0], jnp.sqrt(13.0/12.0), atol=1e-2)
