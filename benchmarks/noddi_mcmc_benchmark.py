
import time
import jax
import jax.numpy as jnp
import blackjax
from dmipy_jax.signal_models import Stick, Ball, Zeppelin
from dmipy_jax.distributions.sphere_distributions import SD1Watson

# -----------------------------------------------------------------------------
# 1. NODDI Implementation (Stick + Zeppelin + Ball with Watson Dispersion)
# -----------------------------------------------------------------------------

class NODDI_Watson:
    def __init__(self, grid_size=200):
        self.stick = Stick()
        self.zeppelin = Zeppelin()
        self.ball = Ball()
        self.watson = SD1Watson(grid_size=grid_size)
    
    def __call__(self, bvals, bvecs, f_iso, f_ic, kappa, mu, d_par, d_iso):
        """
        Computes NODDI signal.
        
        Args:
            bvals: (N,)
            bvecs: (N, 3)
            f_iso: Isotropic volume fraction
            f_ic: Intra-cellular volume fraction (of the non-iso compartment)
            kappa: Concentration parameter for Watson distribution
            mu: (2,) Orientation [theta, phi]
            d_par: Parallel diffusivity
            d_iso: Isotropic diffusivity
        """
        # 1. Get Integration Grid (Orientation Distribution)
        # Returns grid_angles (N_grid, 2) and weights (N_grid,)
        grid_angles, weights = self.watson(mu=mu, kappa=kappa)
        
        # 2. Derive Parameters
        # Global volume fractions
        # f_ic_global = (1 - f_iso) * f_ic
        # f_ec_global = (1 - f_iso) * (1 - f_ic)
        val_f_ic_global = (1.0 - f_iso) * f_ic
        val_f_ec_global = (1.0 - f_iso) * (1.0 - f_ic)
        
        # Tortuosity constraint: d_perp = d_par * (1 - f_ic)
        d_perp = d_par * (1.0 - f_ic)
        
        # 3. Compute Anisotropic Signals (Stick + Zeppelin) at each grid point
        # We need to reshape/broadcast bvals/bvecs for vmap or map over grid
        
        def compute_compartments(orientation):
            # orientation: (2,) [theta, phi]
            s_stick = self.stick(bvals, bvecs, mu=orientation, lambda_par=d_par)
            s_zeppelin = self.zeppelin(bvals, bvecs, mu=orientation, lambda_par=d_par, lambda_perp=d_perp)
            return s_stick, s_zeppelin
            
        # vmap over grid orientations
        s_stick_grid, s_zeppelin_grid = jax.vmap(compute_compartments)(grid_angles) 
        # Output info: (N_grid, N_meas)
        
        # Integrate (Weighted Sum)
        # sum( signal(u) * w(u) )
        S_stick_int = jnp.dot(weights, s_stick_grid)    # (N_meas,)
        S_zeppelin_int = jnp.dot(weights, s_zeppelin_grid) # (N_meas,)
        
        # 4. Compute Isotropic Signal
        S_ball = self.ball(bvals, bvecs, lambda_iso=d_iso)
        
        # 5. Combine
        S_total = (val_f_ic_global * S_stick_int + 
                   val_f_ec_global * S_zeppelin_int + 
                   f_iso * S_ball)
                   
        return S_total

# -----------------------------------------------------------------------------
# 2. Synthetic Data Generation
# -----------------------------------------------------------------------------

def generate_synthetic_data(key, noddi_model):
    # Standard Multi-shell Protocol (HCP-like subset)
    # b=0 (1), b=1000 (30), b=2000 (30)
    b0_val = 0.0
    b1_val = 1000.0
    b2_val = 2000.0
    
    n_b0 = 1
    n_b1 = 30
    n_b2 = 30
    
    bvals = jnp.concatenate([
        jnp.full(n_b0, b0_val),
        jnp.full(n_b1, b1_val),
        jnp.full(n_b2, b2_val)
    ])
    
    # Random directions on sphere
    k1, k2 = jax.random.split(key)
    
    def random_directions(rng, n):
        # crude random directions on sphere
        z = jax.random.uniform(rng, (n,), minval=-1.0, maxval=1.0)
        phi = jax.random.uniform(rng, (n,), minval=0.0, maxval=2*jnp.pi)
        x = jnp.sqrt(1 - z**2) * jnp.cos(phi)
        y = jnp.sqrt(1 - z**2) * jnp.sin(phi)
        return jnp.stack([x, y, z], axis=1)

    bvecs_b0 = jnp.zeros((n_b0, 3)) # b=0 has no direction
    bvecs_b1 = random_directions(k1, n_b1)
    bvecs_b2 = random_directions(k2, n_b2)
    bvecs = jnp.concatenate([bvecs_b0, bvecs_b1, bvecs_b2])
    
    # Ground Truth Parameters
    # f_iso=0.0, f_ic=0.6, kappa=1.0
    gt_params = {
        'f_iso': 0.0,
        'f_ic': 0.6,
        'kappa': 1.0,
        'mu': jnp.array([1.57, 0.0]), # theta=pi/2, phi=0 (Along X axis)
        'd_par': 1.7e-3, # um^2/ms -> using 1e-3 to match b-values in s/mm^2 usually? 
                         # Wait, in the codebase d_par was default 1.7e-9 m^2/s = 1.7 um^2/ms
                         # If bvals are 1000 s/mm^2 = 1e6 s/m^2.
                         # 1.7e-9 m^2/s * 1e6 s/m^2 = 1.7e-3.
                         # So if b=1000, d should be 1.7e-3 or similar scaling.
                         # Let's verify units.
                         # In 'stick.py', range is (0.1e-9, 3e-9). These are SI units (m^2/s).
                         # But bvals in 'sphere_test.py' are 1000, 2000.
                         # exp(-b*d) -> b*d must be unitless.
                         # If b=1000 s/mm^2 = 1000 * 1e6 s/m^2 = 1e9 s/m^2.
                         # Then d=1e-9 m^2/s gives b*d = 1. Correct.
                         # So strictly speaking, b=1000 means standard unit-less b-value if d is in um^2/ms?
                         # Usually: b=1000 s/mm^2, d=0.001 mm^2/s = 1 um^2/ms.
                         # 1000 * 0.001 = 1.
                         # In dmipy-jax code: range is 1e-9.
                         # If I pass b=1000 into model that expects d ~ 1e-9, result is exp(-1000 * 1e-9) = exp(-1e-6) ~= 1.
                         # That's wrong. Diffusion should cause signal decay.
                         # So IF the code expects SI units d (1e-9), THEN bvals MUST be in SI units (1e9).
                         # I need to use bvals = 1e6, 2e6, 3e6? or 1e9? 
                         # 1000 s/mm^2 = 1000 * 10^6 s/m^2 = 10^9 s/m^2.
                         # So bvals should be 1e9, 2e9.
                         # Let's adjust bvals to SI units.
        'd_iso': 3.0e-9
    }
    
    # SI Unit Adjustment
    # b=1000 s/mm^2 = 1e9 s/m^2
    bvals_si = bvals * 1e6 
    gt_params['d_par'] = 1.7e-9
    gt_params['d_iso'] = 3.0e-9
    
    # Generate Signal
    signal_clean = noddi_model(bvals_si, bvecs, 
                               gt_params['f_iso'], gt_params['f_ic'], gt_params['kappa'],
                               gt_params['mu'], gt_params['d_par'], gt_params['d_iso'])
                               
    # Add Noise (Rice or Gaussian? prompt said "Generate synthetic signal...").
    # Let's use Gaussian noise with high SNR for benchmark stability first.
    # SNR = 50
    sigma = 1.0 / 50.0
    noise = jax.random.normal(key, signal_clean.shape) * sigma
    signal_noisy = signal_clean + noise
    
    return bvals_si, bvecs, signal_noisy, gt_params, sigma


# -----------------------------------------------------------------------------
# 3. MCMC Sampling (Blackjax)
# -----------------------------------------------------------------------------

def run_mcmc(bvals, bvecs, data, sigma, noddi_model, num_warmup=500, num_samples=1000, rng_key=None):
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # 1. Define Log Probability
    # Parameters to sample: f_iso, f_ic, kappa, mu(theta, phi)
    # Fixed: d_par, d_iso (Usually fixed in NODDI, or at least d_par/d_iso are)
    # The prompt says "sample the posterior distribution of the parameters".
    # Usually f_iso, f_ic, kappa, theta, phi.
    
    def logprob_fn(params):
        # Unpack
        f_iso = jax.nn.sigmoid(params['logit_f_iso']) # Map real -> [0,1]
        f_ic = jax.nn.sigmoid(params['logit_f_ic'])   # Map real -> [0,1]
        kappa = jax.nn.softplus(params['soft_kappa']) # Map real -> positive
        
        # Orientation: simple parameterization for now (unconstrained angles or mapped)
        # theta in [0, pi], phi in [-pi, pi]
        # Using sigmoid/scaling
        theta = jax.nn.sigmoid(params['logit_theta']) * jnp.pi
        phi = jax.nn.sigmoid(params['logit_phi']) * 2 * jnp.pi - jnp.pi
        mu = jnp.array([theta, phi])
        
        # Fixed
        d_par = 1.7e-9
        d_iso = 3.0e-9
        
        # Prediction
        pred = noddi_model(bvals, bvecs, f_iso, f_ic, kappa, mu, d_par, d_iso)
        
        # Likelihood (Gaussian)
        # log L = -0.5 * sum( (y - pred)^2 / sigma^2 )
        log_likelihood = -0.5 * jnp.sum((data - pred)**2) / (sigma**2)
        
        # Priors
        # Uniform priors implied by transforms?
        # Jacobian adjustments needed for proper priors on transformed space?
        # For benchmark speed test, raw likelihood + standard transforms is often sufficient 
        # unless strictly validating Bayesian coverage.
        # Let's add basic Jacobian corrections for [0,1] transforms to ensure Uniform(0,1).
        # y = sigmoid(x) -> log|dy/dx| = log(y(1-y))
        
        log_jac = 0.0
        log_jac += jnp.log(f_iso * (1 - f_iso))
        log_jac += jnp.log(f_ic * (1 - f_ic))
        
        # Kappa prior? Gamma or Uniform? Let's assume Uniform on positive range or weak prior.
        # softplus correction not strictly uniform, close enough for high SNR.
        
        return log_likelihood + log_jac

    # 2. Initial State
    initial_position = {
        'logit_f_iso': -5.0, # Start low iso
        'logit_f_ic': 0.0,   # Start 0.5
        'soft_kappa': 1.0,   # Start ~1
        'logit_theta': 0.0,
        'logit_phi': 0.0
    }
    
    # 3. Setup NUTS
    warmup = blackjax.window_adaptation(blackjax.nuts, logprob_fn)
    
    # 4. Warmup
    print("Starting Warmup...")
    t0 = time.time()
    (state,parameters), _ = warmup.run(rng_key, initial_position, num_steps=num_warmup)
    t1 = time.time()
    print(f"Warmup done in {t1-t0:.2f}s")
    
    # 5. Sampling
    kernel = blackjax.nuts(logprob_fn, **parameters).step
    
    def inference_loop(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)
        
    print(f"Sampling {num_samples} iterations...")
    t2 = time.time()
    # scan
    keys = jax.random.split(rng_key, num_samples)
    final_state, (states, infos) = jax.lax.scan(inference_loop, state, keys)
    # block until done
    _ = states.position['logit_f_iso'].block_until_ready()
    t3 = time.time()
    
    duration = t3 - t2
    iter_per_sec = num_samples / duration
    print(f"Sampling done in {duration:.2f}s. Speed: {iter_per_sec:.2f} iter/s")
    
    return states, iter_per_sec


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------

def main():
    print("=== NODDI MCMC Benchmark ===")
    
    # Setup
    key = jax.random.PRNGKey(42)
    model = NODDI_Watson(grid_size=100) # 100 points for speed/accuracy trade-off
    
    # Generate Data
    print("Generating Synthetic Data...")
    bvals, bvecs, data, gt, sigma = generate_synthetic_data(key, model)
    print("Ground Truth:", gt)
    
    # Run MCMC
    states, speed = run_mcmc(bvals, bvecs, data, sigma, model, num_warmup=500, num_samples=2000, rng_key=key)
    
    # Transform Back
    f_iso_samples = jax.nn.sigmoid(states.position['logit_f_iso'])
    f_ic_samples = jax.nn.sigmoid(states.position['logit_f_ic'])
    kappa_samples = jax.nn.softplus(states.position['soft_kappa'])

    # Statistics
    print("\n--- Results ---")
    
    def report(name, samples, true_val):
        mean = jnp.mean(samples)
        lower = jnp.percentile(samples, 2.5)
        upper = jnp.percentile(samples, 97.5)
        covered = (true_val >= lower) and (true_val <= upper)
        tag = "[OK]" if covered else "[FAIL]"
        print(f"{name}: True={true_val:.4f} | Est={mean:.4f} [{lower:.4f}, {upper:.4f}] {tag}")
        
    report('f_iso', f_iso_samples, gt['f_iso'])
    report('f_ic', f_ic_samples, gt['f_ic'])
    report('kappa', kappa_samples, gt['kappa'])
    
    print(f"\nMethod Speed: {speed:.2f} iterations/sec")
    
    # Validate
    # Minimal validation logic for CI
    assert speed > 10.0, "Sampling speed too low (< 10 it/s)"
    
    # Check coverage (simple check)
    mean_fic = jnp.mean(f_ic_samples)
    assert abs(mean_fic - gt['f_ic']) < 0.1, "Failed to recover f_ic"

if __name__ == "__main__":
    main()
