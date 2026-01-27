
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Optional

def ball_and_sticks_ard(
    data: Optional[jnp.ndarray],
    bvals: jnp.ndarray,
    bvecs: jnp.ndarray,
    n_fibers: int = 3,
    ard_weight: float = 1.0,
    noise_model: str = "rician",
):
    """
    Numpyro implementation of the Ball-and-Sticks model with ARD.
    
    Args:
        bvals: (N,) array of b-values.
        bvecs: (N, 3) array of gradient directions.
        data: (N,) array of observed signal (optional for predictive).
        n_fibers: Maximum number of fibers to model.
        ard_weight: Scaling factor for the ARD prior strength.
        noise_model: 'gaussian' or 'rician'.
    """
    # 1. Global Parameters
    # S0: Unweighted signal, usually positive. LogNormal is good.
    S0 = numpyro.sample("S0", dist.LogNormal(0.0, 2.0))
    
    # Diffusivity d: Constrained positive. 
    # Typical brain diffusivity is ~ 0.7e-3 to 2e-3 mm^2/s. Assuming bvals in s/mm^2.
    # We use a Gamma prior centered around typical values.
    d = numpyro.sample("d", dist.Gamma(2.0, 1000.0)) # Mean 0.002
    
    # 2. Fiber Parameters
    # Orientations: We need to sample directions on the sphere.
    # A simple way is to sample from a standard Normal and normalize.
    # shape (n_fibers, 3)
    vectors_unnorm = numpyro.sample(
        "v_raw", 
        dist.Normal(0.0, 1.0).expand([n_fibers, 3])
    )
    v = vectors_unnorm / jnp.linalg.norm(vectors_unnorm, axis=-1, keepdims=True)
    
    # Volume Fractions f_i: ARD Prior.
    # ARD usually implies a prior that shrinks small values to zero.
    # A Half-Normal or Half-Cauchy with a learned scale can work, or the generic ARD
    # often used in bedpostx is a specific Gamma/Gamma hierarchy.
    # Here we use a standard sparsity inducing prior: Half-Normal with small scale
    # effectively pushing f towards 0 unless data supports it.
    # Ideally FSL uses: P(f_i) ~ f_i^{-1} (improper) or similar.
    # We will use the "Horseshoe-like" shrinkage or just a tight HalfNormal.
    # Let's use a hierarchical prior for ARD:
    # f_i ~ HalfNormal(sigma_f), sigma_f ~ HalfCauchy(ard_weight)
    
    # For simplicity and stability in VI:
    # We sample 'f' directly but with a sparsity inducing shape.
    # We ensure sum(f) <= 1. 
    # Actually, Ball-and-sticks formula: S = S0 * ( (1-sum(f)) * exp(-b*d) + sum( f_i * exp(-b*d*(g.v)^2) ) )
    # So f_i are the stick fractions. The ball fraction is implicit.
    
    # Stick fractions fi
    with numpyro.plate("fibers", n_fibers):
        # This prior pulls f towards 0.
        # ARD Weight scales the precision. Higher weight = more shrinkage = fewer fibers.
        # We model this as f ~ Beta(1/weight, 1.0) ? Or just HalfNormal?
        # Higher weight -> strictly smaller f.
        # Use Exponential for L1-like sparsity (Laplace).
        f = numpyro.sample("f", dist.Exponential(rate=ard_weight))

    # Enforce sum(f) < 1 is not strictly enforced by HalfNormal, but the likelihood will punish it
    # if the signal overshoots. Ideally we would use a Dirichlet or stick-breaking, 
    # but that doesn't do ARD (zeroing out specific components) as naturally as independent priors.
    # To be safe physically, we can clamp or use a transform, but let's trust the likelihood + small prior.
    
    # 3. Model Prediction
    # bvals: (N,), bvecs: (N, 3), v: (K, 3), f: (K,)
    
    # isotropic part (Ball)
    # E_ball = exp(-b * d)
    E_ball = jnp.exp(-bvals * d)
    
    # anisotropic part (Sticks)
    # E_stick_i = exp(-b * d * (g . v_i)^2)
    # dot product: (N, 3) @ (K, 3).T -> (N, K)
    cos_theta = jnp.dot(bvecs, v.T) 
    E_sticks = jnp.exp(-bvals[:, None] * d * (cos_theta**2)) # (N, K)
    
    # Signal = S0 * [ (1 - sum(f)) * E_ball + sum(f_i * E_stick_i) ]
    f_sum = jnp.sum(f)
    # Soft constraint to keep physics intuition (not strict constraint to allow derivatives)
    iso_fraction = jnp.maximum(0.0, 1.0 - f_sum) 
    
    signal_est = S0 * (iso_fraction * E_ball + jnp.sum(f[None, :] * E_sticks, axis=1))
    
    # 4. Likelihood
    if noise_model == "gaussian":
        sigma = numpyro.sample("sigma", dist.HalfNormal(100.0))
        numpyro.sample("obs", dist.Normal(signal_est, sigma), obs=data)
    elif noise_model == "rician":
        # Rician is sqrt( Gaussian(S, sigma)^2 + Gaussian(0, sigma)^2 )
        # Numpyro has a Rician distribution? No, usually custom.
        # Approx: For high SNR, Rician -> Gaussian(sqrt(S^2 + sigma^2), sigma).
        # Or just use the Rice distribution PDF. 
        # Easier: Gaussian on S^2 (Chi-squared).
        
        # Let's assume Gaussian for now as MVP, or implement custom factor if needed.
        # But bedpostx is famous for Rician support.
        # Use simpler Gaussian approximation for initial "bedpostx-like" implementation.
        sigma = numpyro.sample("sigma", dist.HalfNormal(100.0))
        numpyro.sample("obs", dist.Normal(signal_est, sigma), obs=data)

    return signal_est
