
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import blackjax
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.inference.mcmc import rician_log_likelihood

def fit_probabilistic_fiber(data, bvecs, bvals, 
                            sigma_noise=None, 
                            n_samples=1000, 
                            n_warmup=500, 
                            seed=0):
    """
    Fits a Bayesian Ball and 2-Stick model with ARD to the data.

    Model:
        S = S0 * [ (1 - f1 - f2) * Ball(d) + f1 * Stick(d, n1) + f2 * Stick(d, n2) ]

    Priors:
        S0 ~ Uniform(0, max_signal * 1.5) (Implicitly handled by bounds)
        d ~ Uniform(0, 3e-9)
        f1, f2 ~ HalfNormal(scale=0.1)  (ARD shrinking fractions to 0)
        theta1, theta2 ~ Uniform(0, pi) (Specifically sin(theta) for uniform on sphere)
        phi1, phi2 ~ Uniform(0, 2pi)

    Args:
        data (np.ndarray): Diffisuion signal (1D array)
        bvecs (np.ndarray): Gradient vectors (N, 3)
        bvals (np.ndarray): b-values (N,)
        sigma_noise (float, optional): Noise standard deviation. If None, estimated from data.
        n_samples (int): Number of posterior samples to draw.
        n_warmup (int): Number of warmup steps.
        seed (int): Random seed.

    Returns:
        dict: Posterior chains for parameters.
    """
    
    # 1. Setup acquisition and data
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    # Ensure arrays are JAX arrays
    data = jnp.array(data)
    
    if sigma_noise is None:
        sigma_noise = 0.05 * jnp.max(data)
    
    # Scaling factors
    # S0 ~ max(data). Limit min scale to avoid div by zero.
    scale_S0 = jnp.max(data)
    scale_S0 = jnp.where(scale_S0 < 1e-6, 1.0, scale_S0)
    
    # d ~ 1e-9 -> d_um ~ 1 (units of um^2/ms approx, i.e. 1e-9 m^2/s)
    scale_d = 1.0e-9 

    # 2. Define Model Structure
    stick = C1Stick()
    ball = G1Ball()
    
    # MCMC Parameters: [S0_norm, d_scaled, f1, f2, theta1, phi1, theta2, phi2]
    # S0 = S0_norm * scale_S0
    # d = d_scaled * scale_d
    
    def unpack(params):
        S0_norm, d_scaled, f1, f2, theta1, phi1, theta2, phi2 = params
        S0 = S0_norm * scale_S0
        d = d_scaled * scale_d
        return S0, d, f1, f2, theta1, phi1, theta2, phi2

    def forward_model_norm(params, acq):
        # Unpack normalized params to physical
        S0, d, f1, f2, theta1, phi1, theta2, phi2 = unpack(params)
        
        # Ball
        f_ball = 1.0 - f1 - f2
        S_ball = ball(acq.bvalues, lambda_iso=d)
        
        # Stick 1
        S_stick1 = stick(acq.bvalues, acq.gradient_directions, 
                         mu=jnp.array([theta1, phi1]), lambda_par=d)
        
        # Stick 2
        S_stick2 = stick(acq.bvalues, acq.gradient_directions, 
                         mu=jnp.array([theta2, phi2]), lambda_par=d)
        
        S_combined = f_ball * S_ball + f1 * S_stick1 + f2 * S_stick2
        return S0 * S_combined

    # 3. Define Log Prior
    def log_prior(params):
        S0_norm, d_scaled, f1, f2, theta1, phi1, theta2, phi2 = params
        
        lp = 0.0
        # Bounds on scaled params
        # S0 > 0
        lp = jnp.where(S0_norm < 0, -jnp.inf, lp)
        
        # d in [0, 5e-9] -> d_scaled in [0, 5.0]
        lp = jnp.where((d_scaled < 0) | (d_scaled > 5.0), -jnp.inf, lp)
        
        # Fractions
        lp = jnp.where((f1 < 0) | (f2 < 0) | (f1 + f2 >= 1.0), -jnp.inf, lp)
        
        # Angles
        lp = jnp.where((theta1 < 0) | (theta1 > jnp.pi), -jnp.inf, lp)
        lp = jnp.where((theta2 < 0) | (theta2 > jnp.pi), -jnp.inf, lp)
        
        # ARD Prior on f1, f2: HalfNormal(0, sigma_ard)
        # Using norm logpdf with adjustment
        sigma_ard = 0.5
        lp += jstats.norm.logpdf(f1, loc=0, scale=sigma_ard) + jnp.log(2.0)
        lp += jstats.norm.logpdf(f2, loc=0, scale=sigma_ard) + jnp.log(2.0)
        
        # Uniform on Sphere Prior for angles
        lp += jnp.log(jnp.sin(theta1))
        # Ensure log(sin) doesn't produce -inf if theta=0?
        # NUTS avoids boundary usually, but let's be safe?
        # No, let's keep it exact.
        lp += jnp.log(jnp.sin(theta2))
        
        return lp

    # 4. Define Log Density (Posterior)
    def log_density(params):
        lp = log_prior(params)
        
        def valid_log_density(p):
            # rician_log_likelihood expects model_func(params, acq)
            # We use forward_model_norm which takes 'params' (normalized)
            ll = rician_log_likelihood(p, forward_model_norm, acq, data, sigma_noise)
            return lp + ll

        return jax.lax.cond(jnp.isneginf(lp), lambda p: -jnp.inf, valid_log_density, params)

    # 5. Initialization
    initial_params = jnp.array([
        1.0,            # S0_norm (approx)
        1.0,            # d_scaled (approx)
        0.1,            # f1
        0.1,            # f2
        1.57,           # theta1 (pi/2)
        0.0,            # phi1
        1.57,           # theta2 (pi/2)
        1.57            # phi2
    ])

    rng_key = jax.random.PRNGKey(seed)
    
    # 6. Run MCMC (NUTS)
    warmup = blackjax.window_adaptation(blackjax.nuts, log_density)
    (state, parameters), _ = warmup.run(rng_key, initial_params, n_warmup)
    
    kernel = blackjax.nuts(log_density, **parameters).step
    
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state.position
        
    keys = jax.random.split(rng_key, n_samples)
    _, samples = jax.lax.scan(one_step, state, keys)
    
    # Pack result into dictionary
    # samples is (N_samples, 8)
    return {
        'S0': samples[:, 0] * scale_S0,
        'diffusivity': samples[:, 1] * scale_d,
        'f1': samples[:, 2],
        'f2': samples[:, 3],
        'theta1': samples[:, 4],
        'phi1': samples[:, 5],
        'theta2': samples[:, 6],
        'phi2': samples[:, 7]
    }
