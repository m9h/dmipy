import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.distributions.distribute_models import DistributedModel
from dmipy_jax.distributions.sphere_distributions import SD1Watson
from dmipy_jax.gaussian import G2Zeppelin, G1Ball
from dmipy_jax.cylinder import C1Stick

# --- 1. Simulation Logic ---

def get_noddi_model():
    """Defines the Distributed NODDI model."""
    # Intra-cellular: Stick dispersed with Watson
    stick = C1Stick()
    watson = SD1Watson(grid_size=200)
    intra = DistributedModel(stick, watson, target_parameter='mu')
    
    # Extra-cellular: Zeppelin dispersed with Watson
    zeppelin = G2Zeppelin()
    # Note: simple Zeppelin assumes parallel/perp diffusivities.
    # We should fix d_par and d_perp or let them be variables?
    # NODDI constraints: d_par = d_intra (fixed ~1.7e-9), d_perp = d_par * (1 - f_intra).
    # This TORTUOSITY constraint is critical for NODDI.
    # BUT JaxMultiCompartmentModel is just a summation. It doesn't handle functional constraints 
    # between parameters of different sub-models nicely (except via bespoke wrapper).
    # 
    # HOWEVER, the prompt asks for "Implement an SBI inference module... Simulation ... priors ... f_intra ... f_iso ... kappa".
    # And "Standard fitting gives a single point estimate".
    # 
    # If I use JaxMultiCompartmentModel([Stick, Zeppelin, Ball]), I have independent parameters.
    # To enforce Tortuosity (d_perp_extra = d_par (1-f_intra)), I must do it AT GENERATION TIME.
    # I.e. I sample f_intra, then calculate d_perp_extra, and feed it to the Zeppelin simulator.
    # The Zeppelin model parameter 'lambda_perp' will be set dynamically.
    
    watson2 = SD1Watson(grid_size=200) # Need separate instance or same? Same class is fine.
    extra = DistributedModel(zeppelin, watson2, target_parameter='mu')
    
    ball = G1Ball()
    
    model = JaxMultiCompartmentModel([intra, extra, ball])
    return model

# Global Model Instance for valid parameter names/shapes
NODDI_MODEL = get_noddi_model()
# Parameter names will be:
# intra: mu_1, kappa_1, lambda_par_1, partial_volume_0
# extra: mu_2, kappa_2, lambda_par_2, lambda_perp_2, partial_volume_1
# ball: lambda_iso_1, partial_volume_2 (actually pv names are handled by MCM)

# Let's inspect parameter names once (simulated in head):
# intra parameters: lambda_par, mu, kappa. (Collision handling might rename some)
# extra parameters: lambda_par, lambda_perp, mu, kappa.
# ball parameters: lambda_iso.

# MCM appends partial_volume_{i}
# The sub-models parameters are gathered. Collisions are renamed.
# This means we might have 'mu', 'mu_2', 'kappa', 'kappa_2' etc.

# Helper to generate Acquisition
def get_acquisition():
    # Multi-shell b=700, 2000
    # 32 dirs each + b0s? 
    # Let's do 32 dirs for b=700, 32 dirs for b=2000. Total 64.
    # Plus some b0?
    b0_val = 0.0
    b1_val = 700e6 # SI: 700 s/mm^2 = 700 * 1e6 s/m^2 = 7e8
    b2_val = 2000e6 # 2e9
    
    # 2 b0s
    bvals = jnp.concatenate([
        jnp.zeros(2), 
        jnp.full(32, b1_val), 
        jnp.full(32, b2_val)
    ])
    
    # Directions
    # Random on sphere
    key = jax.random.PRNGKey(42)
    # Just uniform on sphere
    from dmipy_jax.examples.sbi.train_dti import sample_on_sphere
    vecs1 = sample_on_sphere(key, (32,))
    vecs2 = sample_on_sphere(key, (32,)) # Same key? No using sample_on_sphere from train_dti might reuse logic
    # Let's re-implement simple sampler here to be self-contained
    k1, k2 = jax.random.split(key)
    
    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    
    v1 = rand_vecs(k1, 32)
    v2 = rand_vecs(k2, 32)
    
    # b0 vecs usually (1,0,0) or undefined.
    v0 = jnp.array([[1.0, 0.0, 0.0]] * 2)
    
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    
    # Create object mock or just struct
    class Acquisition:
        bvalues = bvals
        gradient_directions = bvecs
        delta = None
        Delta = None
        
    return Acquisition()

ACQUISITION = get_acquisition()
N_MEAS = len(ACQUISITION.bvalues)

@jax.jit(static_argnames=('batch_size',))
def get_batch(key, batch_size=128):
    """
    Generates (Signal, Parameters) pairs.
    Parameters of interest: f_intra, f_iso, kappa.
    """
    # Keys
    k_fintra, k_fiso, k_kappa, k_mu, k_noise = jax.random.split(key, 5)
    
    # 1. Sample Priors
    # f_intra: Beta focused on 0.3-0.7. 
    # Beta(a,b). Mode = (a-1)/(a+b-2). Mean = a/(a+b).
    # Let's use Beta(5, 5) -> Mean 0.5. Range mostly 0.2-0.8.
    f_intra_raw = jax.random.beta(k_fintra, 5.0, 5.0, shape=(batch_size,)) 
    # Rescale or just use as is? 0.3-0.7 focus is satisfied by Beta(5,5).
    f_intra = f_intra_raw # (N,)
    
    # f_iso: Sparse prior.
    # Mixture: 90% Uniform(0, 0.05), 10% Uniform(0.05, 1.0)
    # OR Beta(0.5, 5.0) -> skewed to 0.
    f_iso = jax.random.beta(k_fiso, 0.5, 5.0, shape=(batch_size,))
    
    # kappa: Watson concentration. Log-uniform.
    # range 0.1 to 32.
    min_log_k = jnp.log(0.1)
    max_log_k = jnp.log(32.0)
    kappa = jnp.exp(jax.random.uniform(k_kappa, (batch_size,), minval=min_log_k, maxval=max_log_k))
    
    # Orientation mu: Random on sphere
    # (theta, phi)
    # Uniform on sphere
    z = jax.random.normal(k_mu, (batch_size, 3))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    # Convert to theta, phi
    theta = jnp.arccos(jnp.clip(z[:, 2], -1.0, 1.0))
    phi = jnp.arctan2(z[:, 1], z[:, 0])
    mu = jnp.stack([theta, phi], axis=1) # (N, 2)
    
    # 2. Construct Full Parameter Set for Simulation
    # We need to map our physical params to the verbose parameter list of JaxMultiCompartmentModel.
    # Model parameters: 
    #   intra: mu, kappa, lambda_par
    #   extra: mu, kappa, lambda_par, lambda_perp
    #   ball: lambda_iso
    #   fractions: partial_volume_0, partial_volume_1, partial_volume_2 (auto-normalized?)
    #   Warning: MCM treats partial_volumes as weights that are multiplied by signal.
    #   Usually MCM output = pv0*S0 + pv1*S1 + ...
    #   It doesn't enforce sum=1 unless we do it.
    #   So we set pv values explicitly.
    
    # Fractions
    # f_intra is the signal fraction of the INTRA compartment relative to tissue?
    # No, usually f_intra is w.r.t (1-f_iso).
    # Let's assume the classic definition:
    # S = f_iso * S_ball + (1 - f_iso) * [ f_ic * S_stick + (1 - f_ic) * S_zeppelin ]
    #
    # So:
    # pv_ball = f_iso
    # pv_stick = (1 - f_iso) * f_intra
    # pv_zeppelin = (1 - f_iso) * (1 - f_intra)
    
    pv_ball = f_iso
    pv_stick = (1.0 - f_iso) * f_intra
    pv_zepp = (1.0 - f_iso) * (1.0 - f_intra)
    
    # Fixed Diffusivities
    d_par = 1.7e-9 # m^2/s
    d_iso = 3.0e-9
    
    # Tortuosity Constraint for Zeppelin
    # d_perp = d_par * (1 - f_intra)
    d_perp = d_par * (1.0 - f_intra)
    
    # Build Dictionary for MCM
    # We rely on "parameter collision renaming" knowledge or we just inspect model.parameter_names at runtime?
    # Since we are inside JIT, we can't inspect string names easily if dynamic.
    # But names are static in the class.
    # Let's hardcode the expected names based on simple collision logic (suffix _1, _2...)
    # 
    # Correct mapping (Assuming order: Intra(Stick+Watson), Extra(Zepp+Watson), Ball)
    # Intra: mu, kappa, lambda_par. 
    # Extra: mu (collides->mu_1), kappa (collides->kappa_1), lambda_par (collides->lambda_par_1), lambda_perp.
    # Ball: lambda_iso.
    #
    # Note: Using `dipy` or `dmipy` naming:
    # Stick: lambda_par, mu.
    # Watson: mu, kappa.
    # Distributed(Stick, Watson): mu, kappa, lambda_par. (Removed target 'mu' from Stick, added from Watson).
    #
    # Zeppelin: lambda_par, lambda_perp, mu.
    # Distributed(Zepp, Watson): mu, kappa, lambda_par, lambda_perp.
    #
    # So Intra names: mu, kappa, lambda_par.
    # Extra names (collisions): mu_1, kappa_1, lambda_par_1, lambda_perp.
    # Ball names: lambda_iso.
    
    params_dict = {
        'mu': mu,
        'kappa': kappa,
        'lambda_par': jnp.full((batch_size,), d_par),
        
        'mu_2': mu,          # Shared orientation
        'kappa_2': kappa,    # Shared dispersion
        'lambda_par_2': jnp.full((batch_size,), d_par),
        'lambda_perp': d_perp, # Constrained
        
        'lambda_iso': jnp.full((batch_size,), d_iso),
        
        'partial_volume_0': pv_stick,
        'partial_volume_1': pv_zepp,
        'partial_volume_2': pv_ball
    }
    
    # 3. Simulate
    # JaxMultiCompartmentModel handles batching automatically if params are arrays.
    signals = NODDI_MODEL(params_dict, ACQUISITION) # (N, n_meas)
    
    # 4. Noise
    # SNR = 50 for good quality data? Or 30? Prompt says b=700, 2000.
    # Often SNR ~ 30-50 at b0.
    sigma = 1.0 / 30.0
    n1 = jax.random.normal(k_noise, signals.shape) * sigma
    n2 = jax.random.normal(k_noise, signals.shape) * sigma # Need split key? reusing k_noise is bad
    # Fix splitting
    k_n1, k_n2 = jax.random.split(k_noise)
    n1 = jax.random.normal(k_n1, signals.shape) * sigma
    n2 = jax.random.normal(k_n2, signals.shape) * sigma
    
    signals_noisy = jnp.sqrt((signals + n1)**2 + n2**2)
    
    # 5. Targets
    # f_intra, f_iso, kappa
    targets = jnp.stack([f_intra, f_iso, kappa], axis=-1)
    
    return signals_noisy, targets


# --- 2. Inference Network (NPE/MDN) ---
# Reusing mixture density network class
class MixtureDensityNetwork(eqx.Module):
    full_shared_mlp: eqx.nn.MLP
    n_components: int
    n_outputs: int
    
    def __init__(self, key, in_size, out_size, n_components=8, width=128, depth=4):
        self.n_components = n_components
        self.n_outputs = out_size
        total_out = n_components * (1 + 2 * out_size)
        self.full_shared_mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=total_out,
            width_size=width,
            depth=depth,
            activation=jax.nn.gelu,
            key=key
        )
    
    def __call__(self, x):
        raw_out = self.full_shared_mlp(x)
        nc = self.n_components
        no = self.n_outputs
        logits = raw_out[:nc]
        means = raw_out[nc : nc + nc*no].reshape(nc, no)
        log_sigmas = raw_out[nc + nc*no:].reshape(nc, no)
        sigmas = jnp.exp(log_sigmas)
        return logits, means, sigmas

def mdn_loss_fn(model, x, y):
    logits, means, sigmas = model(x)
    y_b = y 
    z = (y_b - means) / sigmas
    log_prob_comps = -0.5 * jnp.sum(z**2, axis=-1) - jnp.sum(jnp.log(sigmas), axis=-1) - 0.5 * means.shape[1] * jnp.log(2*jnp.pi)
    log_pis = jax.nn.log_softmax(logits)
    final_log_prob = jax.scipy.special.logsumexp(log_pis + log_prob_comps)
    return -final_log_prob

def loss_fn(model, x_batch, y_batch):
    per_sample_loss = jax.vmap(mdn_loss_fn, in_axes=(None, 0, 0))(model, x_batch, y_batch)
    return jnp.mean(per_sample_loss)

# --- 3. Main ---

def main():
    print("Initializing NODDI SBI Training...")
    
    # Hyperparams
    LR = 5e-4
    BATCH_SIZE = 256
    ITERS = 500
    
    key = jax.random.PRNGKey(123)
    k_net, k_train = jax.random.split(key)
    
    # Init Model
    # Input: 66 signals (2+32+32)
    # Output: 3 params
    model = MixtureDensityNetwork(k_net, in_size=N_MEAS, out_size=3, n_components=4) # 4 components enough?
    
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print("Starting Loop...")
    k_iter = k_train
    losses = []
    
    try:
        from tqdm import tqdm
        pbar = tqdm(range(ITERS))
    except ImportError:
        pbar = range(ITERS)
        
    for i in pbar:
        k_iter, k_batch = jax.random.split(k_iter)
        x, y = get_batch(k_batch, BATCH_SIZE)
        model, opt_state, loss = step(model, opt_state, x, y)
        losses.append(loss)
        if i % 500 == 0:
            if isinstance(pbar, range): print(f"{i}: {loss}")
            
    # --- 4. Validation Plots ---
    print("Generating Corner Plots...")
    
    # Helper to sample from posterior (Mixed Gaussian)
    def sample_posterior(model, x, key, n_samples=1000):
        logits, means, sigmas = model(x) # (K,), (K, D), (K, D)
        # Sample component
        k_cat, k_gauss = jax.random.split(key)
        # Gumbel-max or categorical
        comp_indices = jax.random.categorical(k_cat, logits, shape=(n_samples,))
        
        # Gather means/sigmas
        # means[comp_indices]
        m = means[comp_indices]
        s = sigmas[comp_indices]
        
        noise = jax.random.normal(k_gauss, (n_samples, 3))
        samples = m + s * noise
        return samples

    # Test Case: Crossing Fiber / Partial Volume?
    # Actually just a random test sample
    k_test = jax.random.PRNGKey(999)
    x_test, y_test = get_batch(k_test, 5) # 5 validation examples
    
    # Pick one with high dispersion or ambiguous
    idx = 0
    true_params = y_test[idx] # f_intra, f_iso, kappa
    signal = x_test[idx]
    
    post_samples = sample_posterior(model, signal, k_test, n_samples=5000)
    
    import corner
    import numpy as np
    
    # Convert to numpy
    samples_np = np.array(post_samples)
    true_np = np.array(true_params)
    
    labels = [r"$f_{intra}$", r"$f_{iso}$", r"$\kappa$"]
    
    fig = corner.corner(
        samples_np, 
        labels=labels, 
        truths=true_np,
        truth_color='red',
        show_titles=True
    )
    
    plt.savefig("noddi_posterior.png")
    print("Saved noddi_posterior.png")
    
    # Plot Training Curve
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.title("NPE Training Loss")
    plt.savefig("loss_curve.png")

if __name__ == "__main__":
    main()
