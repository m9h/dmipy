
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List
from jaxtyping import Array, Float, PRNGKeyArray

class ICNNLayer(eqx.Module):
    """
    Input Convex Neural Network Layer.
    z_{i+1} = sigma( W_z * z_i + W_y * y + b )
    
    Constraint: W_z >= 0 to maintain convexity w.r.t input y.
    """
    w_z: eqx.nn.Linear
    w_y: eqx.nn.Linear
    
    def __init__(self, in_z: int, in_y: int, out_size: int, key: PRNGKeyArray):
        k_z, k_y = jax.random.split(key)
        
        # Custom Init: Weights should be negative so softplus(w) is small positive.
        # Softplus(0) = 0.69. Softplus(-2) = 0.12.
        # Initialize w ~ N(-1.0, 0.1)
        
        # W_z
        self.w_z = eqx.nn.Linear(in_z, out_size, use_bias=True, key=k_z)
        w_z_init = jax.random.normal(k_z, self.w_z.weight.shape) * 0.5 - 4.0
        self.w_z = eqx.tree_at(lambda l: l.weight, self.w_z, w_z_init)
        
        # W_y
        self.w_y = eqx.nn.Linear(in_y, out_size, use_bias=False, key=k_y)
        # W_y can be negative? Wait, if we use softplus on w_y in call?
        # In current call: y_part = self.w_y(y). Standard linear.
        # So w_y can be standard.
        # But z_part adds to it.
        # If w_y is standard centere 0, it's fine.
        
    def __call__(self, z: Array, y: Array) -> Array:
        # Enforce non-negative W_z
        w_z_pos = jax.nn.softplus(self.w_z.weight)
        z_part = w_z_pos @ z + self.w_z.bias
        y_part = self.w_y(y)
        return jax.nn.relu(z_part + y_part)

class NeuralSignalModel(eqx.Module):
    """
    Represents -log E(q) as a convex function C(q).
    """
    layers: List[ICNNLayer]
    final_layer: eqx.nn.Linear
    
    def __init__(self, data_dim: int = 3, hidden_dim: int = 64, depth: int = 3, key: PRNGKeyArray = None):
        if key is None: key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, depth + 2)
        
        self.layers = []
        for i in range(depth):
            self.layers.append(ICNNLayer(hidden_dim, data_dim, hidden_dim, keys[i]))
            
        self.final_layer = eqx.nn.Linear(hidden_dim, 1, key=keys[-1]) 
        # Init final layer weights to be small positive (approx 0.5 after softplus)
        w_final_init = jax.random.normal(keys[-1], self.final_layer.weight.shape) * 0.5 - 4.0
        self.final_layer = eqx.tree_at(lambda l: l.weight, self.final_layer, w_final_init)
        
    def __call__(self, q: Array) -> Array:
        # q shape (3,)
        
        # z_0 = zeros
        hidden = jnp.zeros((self.layers[0].w_z.in_features,))
        
        for layer in self.layers:
            hidden = layer(hidden, q)
            
        # Final integration
        # Output = W_z * hidden + bias
        # Force positive weights
        w_final_pos = jax.nn.softplus(self.final_layer.weight)
        output = w_final_pos @ hidden + self.final_layer.bias
        
        return output[0] # Scalar

    def parameters(self, q):
        # Enforce Symmetry: C(q) = ICNN(q) + ICNN(-q)
        # Sum of convex functions is convex.
        # Reflection symmetry enforced by construction.
        
        val_pos = self(q)
        val_neg = self(-q)
        raw_val = 0.5 * (val_pos + val_neg)
        
        # Enforce C(0) = 0
        # Compute zero val once (can be optimized, but ok for now)
        # zero_val = self(0) + self(-0) = 2 * self(0)
        # scaled by 0.5 -> self(0).
        
        # For efficiency, we can subtract C(0) if we knew it.
        # Let's compute it dynamically to be safe.
        zero_val = self(jnp.zeros_like(q)) 
        # Note: self(0) is ICNN(0). raw_val(0) = 0.5(ICNN(0)+ICNN(0)) = ICNN(0).
        
        return raw_val - zero_val


if __name__ == "__main__":
    print("Verifying ICNN Signal Model...")
    
    # 1. Generate Fake Signal (Sphere)
    # E(q) ~ exp(- q^2 D) roughly for low q.
    # High q diffraction.
    # Let's train on E(q) = exp(- |q|^2) for simplicity (Gaussian).
    
    key = jax.random.PRNGKey(0)
    
    # Training Data
    # 100 random q vectors
    q_train = jax.random.normal(key, (100, 3))
    # Signal = exp(-|q|^2)
    s_train = jnp.exp(-jnp.sum(q_train**2, axis=-1))
    
    # Log Signal Target: - |q|^2 (which is convex)
    # We train model to predict -log E = |q|^2.
    y_target = -jnp.log(s_train + 1e-6)
    
    model = NeuralSignalModel(data_dim=3, hidden_dim=64, key=jax.random.PRNGKey(42))
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    @eqx.filter_jit
    def step(model, opt_state, q_batch, y_batch):
        def loss_fn(m):
            # Enforce C(0)=0 in prediction
            preds = jax.vmap(m.parameters)(q_batch)
            return jnp.mean((preds - y_batch)**2)
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
        
    print("Training...")
    for i in range(500):
        model, opt_state, loss = step(model, opt_state, q_train, y_target)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
            
    # Verification
    # Check 1: E(0) = 1 (C(0)=0)
    c0 = model.parameters(jnp.zeros(3))
    print(f"\nC(0) (should be 0): {c0:.6f}")
    
    # Check 2: Convexity
    # Check along a ray
    q_test = jnp.array([1.0, 0.0, 0.0])
    vals = []
    alphas = jnp.linspace(-2, 2, 20)
    for a in alphas:
        vals.append(model.parameters(q_test * a))
    vals = jnp.array(vals)
    
    # Second derivative check numerically?
    # Just print values. Should be quadratic-like.
    print(f"Values along ray (should be U-shaped): {vals[::4]}")
    
    # Check if vals are symmetric?
    sym_err = jnp.mean((vals - vals[::-1])**2)
    print(f"Symmetry Error: {sym_err:.6f}")
    
    if loss < 0.1 and abs(c0) < 1e-5:
        print("SUCCESS: ICNN learned convex signal representation.")
    else:
        print("WARNING: Training failed.")
