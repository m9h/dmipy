
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import pickle
import numpy as np

# A simple Emulator Network
# Input: (radius, q_magnitude)
# Output: Signal
class EmulatorNet(eqx.Module):
    layers: list
    
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # Inputs: radius, q (2 params)
        # Hidden: 64
        self.layers = [
            eqx.nn.Linear(2, 64, key=k1),
            jax.nn.softplus, # Softplus for smooth activation
            eqx.nn.Linear(64, 64, key=k2),
            jax.nn.softplus,
            eqx.nn.Linear(64, 1, key=k3),
            jax.nn.sigmoid # Signal in [0, 1]
        ]
        
    def __call__(self, radius, q):
        x = jnp.stack([radius, q], axis=-1)
        for layer in self.layers:
            x = layer(x)
        return x[0] # scalar

def train_emulator():
    print("=== Training JAX-MD Emulator ===")
    
    # 1. Load Data
    with open('emulator_train_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    radii = jnp.array(data['radii'])     # (N_samples,)
    qs = jnp.array(data['q_values'])     # (N_q,)
    signals = jnp.array(data['signals']) # (N_samples, N_q)
    
    print(f"Loaded Data: {signals.shape}")
    
    # Flatten for training: (N*Q, 2) -> (N*Q, 1)
    # Inputs: pairs of (r, q)
    # Use broadcasting/meshgrid logic
    
    r_grid, q_grid = jnp.meshgrid(radii, qs, indexing='ij')
    
    X_train = jnp.stack([r_grid.ravel(), q_grid.ravel()], axis=1) # (N*Q, 2)
    Y_train = signals.ravel()[:, None] # (N*Q, 1)
    
    # Normalize Inputs for better training
    # r in [1e-6, 10e-6]. Scale to [0, 1] or similar.
    # q in [0, 5e5]. Scale.
    r_mean = jnp.mean(X_train[:, 0])
    r_std = jnp.std(X_train[:, 0])
    q_mean = jnp.mean(X_train[:, 1])
    q_std = jnp.std(X_train[:, 1])
    
    def normalize(r, q):
        return (r - r_mean)/r_std, (q - q_mean)/q_std
        
    X_norm = jnp.stack(normalize(X_train[:, 0], X_train[:, 1]), axis=1)
    
    # 2. Setup Model
    key = jax.random.PRNGKey(66)
    model = EmulatorNet(key)
    
    optimizer = optax.adam(0.001)
    # Filter params for array-only optimization
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    
    # 3. Optimize
    @eqx.filter_value_and_grad
    def loss_fn(m, x_in, y_target):
        # x_in: (Batch, 2) (normalized)
        # We need to unpack rows. vmap the model.
        # model takes (r, q) scalars.
        preds = jax.vmap(m)(x_in[:, 0], x_in[:, 1])
        return jnp.mean((preds.reshape(-1, 1) - y_target)**2)
        
    print(f"Training on {X_norm.shape[0]} points...")
    
    # Simple Batching? Or Full Batch?
    # 10k points is small enough for full batch on GPU, maybe CPU too.
    # Let's do full batch for demo simplicity.
    
    for i in range(500):
        loss, grads = loss_fn(model, X_norm, Y_train)
        
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        
        if i % 50 == 0:
            print(f"Iter {i}: MSE Loss = {loss:.6f}")
            
    # 4. Inverse Crime Verification
    print("\n=== Inverse Crime Test ===")
    # Can we recover Radius from Signal using the Differentiable Emulator?
    
    # Select a ground truth
    idx = 50 # Random index
    true_r = radii[idx]
    true_signal_vector = signals[idx] # (50,)
    
    print(f"Ground Truth Radius: {true_r:.2e}")
    
    # Optimization to find r
    # We define a loss(r_estim) = mean( (Emulator(r_estim, qs) - true_signal)^2 )
    
    # Initial guess
    r_guess = jnp.array([5.0e-6]) # Middle
    
    # Optimizer for inversion
    inv_optimizer = optax.adam(0.05)
    inv_state = inv_optimizer.init(r_guess)
    
    @jax.value_and_grad
    def inverse_loss(r):
        # Normalize r
        r_n, q_n = normalize(r, qs) 
        # r_n is (1,), q_n is (50,)
        # Broadcast r_n to (50,)
        r_n_broad = jnp.repeat(r_n, q_n.shape[0])
        
        # Predict
        preds = jax.vmap(model)(r_n_broad, q_n)
        return jnp.mean((preds - true_signal_vector)**2)
        
    est_r = r_guess
    for k in range(100):
        loss_inv, grads_inv = inverse_loss(est_r)
        updates, inv_state = inv_optimizer.update(grads_inv, inv_state, est_r)
        est_r = optax.apply_updates(est_r, updates)
        
    print(f"Recovered Radius: {est_r[0]:.2e}")
    error = jnp.abs(est_r[0] - true_r) / true_r * 100
    print(f"Error: {error:.2f}%")
    
    if error < 5.0:
        print("SUCCESS: Emulator enables differentiable inversion of JAX-MD physics.")
    else:
        print("WARNING: Inversion inaccurate.")

if __name__ == "__main__":
    train_emulator()
