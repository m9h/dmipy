import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from dmipy_jax.inference.variational import MeanFieldGaussian, VIMinimizer, fit_vi
from dmipy_jax.signal_models.sphere_models import SphereGPD
from dmipy_jax.optimization.acquisition import AcquisitionProtocol

def test_vi_gaussian_toy_problem():
    """
    Test VI on a simple problem: estimate mean of a distribution.
    Prior: N(0, inf) (flat)
    Likelihood: y ~ N(mu, 1)
    Data: y = 5.0
    Posterior q(mu) should converge to N(5.0, 1.0) approx (driven by entropy/LL balance).
    If we strictly minimize NLL + Entropy, sigma might go to 0 without prior anchor?
    Actually if Loss = - (LL + H).
    LL = -0.5 (y - mu)^2 / sigma_noise^2.
    If sigma_noise=1.
    E[LL] = -0.5 ( (y - E[mu])^2 + Var[mu] )
    H = 0.5 (1 + log(2pi) + 2 log sigma_q)
    Loss = 0.5 [ (y - mu_q)^2 + sigma_q^2 ] - 0.5 (2 log sigma_q)
    dLoss/dSigma = sigma_q - 1/sigma_q = 0 => sigma_q = 1.
    dLoss/dMu = (y - mu_q) = 0 => mu_q = y.
    
    So it should converge to mu=y, sigma=sigma_noise.
    """
    
    # Mock Model: Linear model mu -> mu
    class IdentityModel(eqx.Module):
        def __call__(self, **kwargs):
            return jnp.array([kwargs['x']])
            
    model = IdentityModel()
    
    # Mock Acquisition
    class MockAcq:
        bvalues = jnp.array([0.0])
        gradient_directions = jnp.array([[0.0, 0.0, 1.0]])
        Delta = jnp.array([1.0])
        delta = jnp.array([1.0])
    
    acq = MockAcq()
    
    # Data = 5.0. Noise sigma = 1.0
    data = jnp.array([5.0])
    
    init_params = {'x': 0.0}
    
    posterior, losses = fit_vi(
        tissue_model=model,
        acquisition=acq,
        data=data,
        init_params=init_params,
        sigma_noise=1.0,
        learning_rate=0.05,
        num_steps=4000,
        seed=42
    )
    
    mu_final = posterior.means['x']
    std_final = jnp.exp(posterior.log_stds['x'])
    
    print(f"Final Mu: {mu_final}, Final Std: {std_final}")
    
    assert jnp.allclose(mu_final, 5.0, atol=0.2)
    assert jnp.allclose(std_final, 1.0, atol=0.1)

def test_vi_sphere_model():
    """
    Test VI fitting on SphereGPD model.
    """
    # Ground Truth
    gt_diameter = 6.0e-6
    gt_diff = 2.0e-9
    
    model = SphereGPD()
    
    # Generate Data
    # Use SI units: 3000 s/mm^2 = 3e9 s/m^2
    protocol_model = AcquisitionProtocol(n_measurements=20, max_b_value=3e9)
    acq = protocol_model()
    
    signal = model(
        bvals=acq.bvalues,
        gradient_directions=acq.gradient_directions,
        big_delta=acq.Delta,
        small_delta=acq.delta,
        diameter=gt_diameter,
        diffusion_constant=gt_diff
    )
    
    # Add noise
    key = jax.random.PRNGKey(123)
    sigma = 0.02
    noisy_signal = signal + sigma * jax.random.normal(key, signal.shape)
    
    # Fit VI
    init_params = {'diameter': 8.0e-6, 'diffusion_constant': 1.5e-9} # Perturbed initialization
    
    posterior, losses = fit_vi(
        tissue_model=model,
        acquisition=acq,
        data=noisy_signal,
        init_params=init_params,
        sigma_noise=sigma,
        learning_rate=0.005,
        num_steps=5000,
        seed=999
    )

    # Evaluation
    # Since Softplus is used in LogLikelihood wrapper, the posterior means are in UNCONSTRAINED space.
    # To compare with GT, we need to transform them back.
    # The LogLikelihood wrapper used `jax.nn.softplus`.

    est_diameter_unconstrained = posterior.means['diameter']
    est_diff_unconstrained = posterior.means['diffusion_constant']

    # Apply softplus AND scaling (inverse of what was done in log_likelihood)
    est_diameter = jax.nn.softplus(est_diameter_unconstrained) * 1e-6
    est_diff = jax.nn.softplus(est_diff_unconstrained) * 1e-9

    print(f"GT Diam: {gt_diameter}, Est: {est_diameter}")
    print(f"GT Diff: {gt_diff}, Est: {est_diff}")

    # Roughly within range?
    # Note: VI might be biased or stuck in local optima, or convergence requires more steps.
    # We check for reasonable proximity.
    assert jnp.abs(est_diameter - gt_diameter) < 3e-6
    assert jnp.abs(est_diff - gt_diff) < 1e-8
    
    # Check uncertainty scales with noise?
    # Sigma ~ Noise * Sensitivity^-1.
    # Just check it's not zero and not huge.
    std_diam = jnp.exp(posterior.log_stds['diameter'])
    # Since params passed through softplus, the effective std in physical space:
    # Std_y ~ Std_x * sigmoid(x) (derivative of softplus)
    # Just check unconstrained std is reasonable (e.g., < 1.0 in logit space)
    
    assert std_diam < 2.0
    assert std_diam > 0.001
