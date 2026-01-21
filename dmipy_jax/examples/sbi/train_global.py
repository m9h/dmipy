import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import sys
import os
import functools
from typing import Callable, Tuple, Any

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.signal_models import gaussian_models

# --- 1. Global Simulation with Varying Acquisition ---

def sample_random_acquisition(key, max_b=3000.0, max_n=100, min_n=6):
    """Samples a random acquisition protocol."""
    k_n, k_b, k_vec = jax.random.split(key, 3)
    n_meas = jax.random.randint(k_n, (), minval=min_n, maxval=max_n + 1)
    bvals_valid = jax.random.uniform(k_b, (max_n,), minval=0.0, maxval=max_b)
    vecs = jax.random.normal(k_vec, (max_n, 3))
    vecs = vecs / (jnp.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-9)
    
    indices = jnp.arange(max_n)
    mask = indices < n_meas
    
    bvals = bvals_valid * mask
    bvecs = vecs * mask[:, None]
    
    return bvals, bvecs, mask

@functools.partial(jax.jit, static_argnames=['batch_size', 'max_n_dirs'])
def get_global_batch(key, batch_size=128, max_n_dirs=100):
    """
    Generates batch of varying acquisitions and corresponding signals.
    Targets: DTI Lower Triangular Tensor (6 components).
    """
    k_acq, k_params, k_noise = jax.random.split(key, 3)
    
    # 1. Acquisition
    bvals, bvecs, mask = sample_random_acquisition(k_acq, max_n=max_n_dirs)
    
    class RandomAcq:
        bvalues = bvals
        gradient_directions = bvecs
    acq = RandomAcq()
    
    # 2. Parameters (DTI Tensor)
    k_l, k_orient = jax.random.split(k_params)
    l_min, l_max = 0.1e-9, 3.0e-9
    
    k1, k2, k3 = jax.random.split(k_l, 3)
    l1 = jax.random.uniform(k1, (batch_size,), minval=l_min, maxval=l_max)
    l2 = jax.random.uniform(k2, (batch_size,), minval=l_min, maxval=l1)
    l3 = jax.random.uniform(k3, (batch_size,), minval=l_min, maxval=l2)
    
    k_a, k_b, k_g = jax.random.split(k_orient, 3)
    alpha = jax.random.uniform(k_a, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    beta = jax.random.uniform(k_b, (batch_size,), minval=0.0, maxval=jnp.pi)
    gamma = jax.random.uniform(k_g, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    
    # 3. Simulate Signal
    def sim_one(l1, l2, l3, a, b, g):
        model = gaussian_models.Tensor(l1, l2, l3, a, b, g)
        sig = model(acq.bvalues, acq.gradient_directions)
        
        # Manually compute D tensor
        # R matrix from ZYZ Euler angles (alpha, beta, gamma)
        ca, sa = jnp.cos(a), jnp.sin(a)
        cb, sb = jnp.cos(b), jnp.sin(b)
        cg, sg = jnp.cos(g), jnp.sin(g)
        
        # R = Rz(alpha) Ry(beta) Rz(gamma)
        R = jnp.array([
            [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
            [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
            [-sb*cg,           sb*sg,             cb]
        ])
        
        # E = diag(l1, l2, l3)
        E = jnp.diag(jnp.array([l1, l2, l3]))
        
        # D = R E R.T
        D = R @ E @ R.T
        
        # Extract components (SI units) scaled by 1e9
        D_scaled = D * 1e9
        
        # Order: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
        d6 = jnp.stack([
            D_scaled[0,0], D_scaled[0,1], D_scaled[0,2],
            D_scaled[1,1], D_scaled[1,2],
            D_scaled[2,2]
        ])
        
        return sig, d6

    signals, targets = jax.vmap(sim_one)(l1, l2, l3, alpha, beta, gamma)
    
    signals = signals * mask 
    
    # 4. Noise
    sigma = 1.0 / 30.0
    k_n1, k_n2 = jax.random.split(k_noise)
    n1 = jax.random.normal(k_n1, signals.shape) * sigma
    n2 = jax.random.normal(k_n2, signals.shape) * sigma
    
    signals_noisy = jnp.sqrt((signals + n1)**2 + n2**2)
    signals_noisy = signals_noisy * mask
    
    # Prepare Inputs
    bvals_norm = bvals / 3000.0
    s = signals_noisy
    b = jnp.tile(bvals_norm, (batch_size, 1))
    v = jnp.tile(bvecs, (batch_size, 1, 1)) 
    m = jnp.tile(mask, (batch_size, 1))
    
    features = jnp.concatenate([
        s[..., None], 
        b[..., None], 
        v, 
        m[..., None]
    ], axis=-1)
    
    inputs = features.reshape(batch_size, -1)
    
    return inputs, targets # Targets is (B, 6)

# --- 2. Network ---

class GlobalMDN(eqx.Module):
    full_shared_mlp: eqx.nn.MLP
    n_components: int
    n_outputs: int
    
    def __init__(self, key, max_n, n_outputs=6, n_components=8):
        self.n_components = n_components
        self.n_outputs = n_outputs
        in_size = max_n * 6
        total_out = n_components * (1 + 2 * n_outputs)
        
        self.full_shared_mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=total_out,
            width_size=512, # Wider
            depth=5,        # Deeper
            activation=jax.nn.gelu,
            key=key
        )
        
    def __call__(self, x):
        raw = self.full_shared_mlp(x)
        nc = self.n_components
        no = self.n_outputs
        
        logits = raw[:nc]
        means = raw[nc : nc + nc*no].reshape(nc, no)
        log_sigmas = raw[nc + nc*no:].reshape(nc, no)
        sigmas = jnp.exp(log_sigmas)
        
        return logits, means, sigmas

def mdn_loss_fn(model, x, y):
    logits, means, sigmas = model(x)
    y_b = y
    z = (y_b - means) / sigmas
    log_prob_comps = -0.5 * jnp.sum(z**2, axis=-1) - jnp.sum(jnp.log(sigmas), axis=-1) - 0.5 * means.shape[1] * jnp.log(2*jnp.pi)
    log_pis = jax.nn.log_softmax(logits)
    return -jax.scipy.special.logsumexp(log_pis + log_prob_comps)

def loss_fn(model, x_batch, y_batch):
    return jnp.mean(jax.vmap(mdn_loss_fn, in_axes=(None, 0, 0))(model, x_batch, y_batch))

# --- 3. Training ---

def main():
    print("Initializing Global SBI Training (Tensor Mode)...")
    
    MAX_N = 64
    BATCH_SIZE = 256
    ITERS = 2000
    
    key = jax.random.PRNGKey(42)
    k_net, k_loop = jax.random.split(key)
    
    # 6 Outputs = Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
    model = GlobalMDN(k_net, max_n=MAX_N, n_outputs=6, n_components=8)
    
    optimizer = optax.adam(2e-4) # Conservative LR
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Starting Loop...")
    losses = []
    
    try:
        from tqdm import tqdm
        pbar = tqdm(range(ITERS))
    except ImportError:
        pbar = range(ITERS)
        
    for i in pbar:
        k_loop, k_batch = jax.random.split(k_loop)
        x, y = get_global_batch(k_batch, BATCH_SIZE, MAX_N)
        model, opt_state, loss = step(model, opt_state, x, y)
        losses.append(loss)
        
        if i % 100 == 0:
             if isinstance(pbar, range): print(f"{i}: {loss}")
             
    eqx.tree_serialise("global_sbi_model.eqx", model)
    print("Model saved.")
    
    # Validation
    print("Validating with 6-dir subset...")
    k_val = jax.random.PRNGKey(99)
    x_val, y_val = get_global_batch(k_val, batch_size=100, max_n_dirs=MAX_N)
    
    @eqx.filter_jit
    def predict(model, x):
        logits, means, _ = model(x)
        w = jax.nn.softmax(logits)[:, None]
        return jnp.sum(w * means, axis=0) # E[y]
        
    preds = jax.vmap(predict, in_axes=(None, 0))(model, x_val)
    
    mse = jnp.mean((preds - y_val)**2)
    print(f"Mean Squared Error (Tensor Components): {mse}")
    
    plt.plot(losses)
    plt.title("Global SBI Tensor Training")
    plt.savefig("global_loss.png")
    
if __name__ == "__main__":
    main()
