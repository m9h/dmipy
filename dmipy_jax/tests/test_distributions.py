import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.distributions import DD1Gamma, DistributedModel
from dmipy_jax.signal_models import cylinder_models
from jax.scipy import stats

def test_dd1gamma_pdf():
    """Test if DD1Gamma generates a valid PDF."""
    alpha = 5.0
    beta = 1e-6
    gamma_dist = DD1Gamma()
    
    # Get grid and pdf
    radii, pdf = gamma_dist(alpha=alpha, beta=beta)
    
    # Check shape
    assert radii.shape == (50,)
    assert pdf.shape == (50,)
    assert jnp.all(pdf >= 0)
    
    # Check integral is close to 1
    if hasattr(jnp, 'trapezoid'):
        integral = jnp.trapezoid(pdf, x=radii)
    else:
        integral = jnp.trapz(pdf, x=radii)
    assert jnp.abs(integral - 1.0) < 0.05 # Approximate check due to finite grid
    
    # Check against analytic PDF at a point
    test_r = radii[25]
    expected = stats.gamma.pdf(test_r / beta, alpha) / beta
    assert jnp.allclose(pdf[25], expected)

def test_distributed_model_instantiation():
    """Test constructing an AxCaliber matching distributed model."""
    cylinder = cylinder_models.RestrictedCylinder()
    gamma = DD1Gamma()
    
    # AxCaliber: Cylinder distributed over diameter using Gamma
    axcaliber = DistributedModel(cylinder, gamma, target_parameter='diameter')
    
    # Check parameters
    # Expected: mu, lambda_par, alpha, beta (diameter removed)
    assert 'diameter' not in axcaliber.parameter_names
    assert 'alpha' in axcaliber.parameter_names
    assert 'beta' in axcaliber.parameter_names
    assert 'mu' in axcaliber.parameter_names
    assert 'lambda_par' in axcaliber.parameter_names
    
    # Check ranges
    assert axcaliber.parameter_ranges['alpha'] == (0.1, 30.)

def test_distributed_model_execution():
    """Smoke test for AxCaliber forward pass."""
    cylinder = cylinder_models.RestrictedCylinder()
    gamma = DD1Gamma()
    axcaliber = DistributedModel(cylinder, gamma, target_parameter='diameter')
    
    # Mock acquisition
    N = 5
    bvals = jnp.ones(N) * 1000.0
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0)
    
    params = {
        'lambda_par': 1.7e-9,
        'mu': jnp.array([jnp.pi/2, 0.0]),
        'alpha': 4.0,
        'beta': 1.0e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    # JIT check
    signal = jax.jit(axcaliber)(bvals, bvecs, **params)
    
    assert signal.shape == (N,)
    assert jnp.all(jnp.isfinite(signal))
    assert jnp.all(signal >= 0)
    assert jnp.all(signal <= 1.0)

def test_distributed_model_gradients():
    """Check if gradients propagate through integration."""
    cylinder = cylinder_models.RestrictedCylinder()
    gamma = DD1Gamma()
    axcaliber = DistributedModel(cylinder, gamma, target_parameter='diameter')
    
    N = 2
    bvals = jnp.ones(N) * 2000.0
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0)
    
    params = {
        'lambda_par': 1.7e-9,
        'mu': jnp.array([jnp.pi/2, 0.0]),
        'alpha': 5.0,
        'beta': 1.0e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }

    # Define loss function
    def loss(alpha):
        p = params.copy()
        p['alpha'] = alpha
        s = axcaliber(bvals, bvecs, **p)
        return jnp.sum(s)
        
    grad_fun = jax.grad(loss)
    grad_alpha = grad_fun(5.0)
    
    assert jnp.isfinite(grad_alpha)
    assert grad_alpha != 0.0 # Should be sensitive to alpha changes
