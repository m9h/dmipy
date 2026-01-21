
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os
from dmipy_jax.core.networks import SIREN, SineLayer

def target_function(x):
    return jnp.sin(10 * x) + jnp.cos(25 * x) * jnp.sin(5 * x)

def train_siren():
    key = jr.PRNGKey(0)
    model_key, train_key = jr.split(key)
    
    # Initialize SIREN
    # 1 input (x), 1 output (y), 64 hidden units, 3 hidden layers
    model = SIREN(
        in_features=1,
        out_features=1,
        hidden_features=64,
        hidden_layers=3,
        key=model_key,
        first_omega_0=30.0,
        hidden_omega_0=30.0
    )
    
    # Verify initialization
    print("Verifying initialization...")
    first_layer = model.layers[0]
    if isinstance(first_layer, SineLayer):
        expected_bound = 1 / 1 # in_features = 1
        w_min, w_max = jnp.min(first_layer.weight), jnp.max(first_layer.weight)
        print(f"Layer 1 weights range: [{w_min:.4f}, {w_max:.4f}] vs bound +/- {expected_bound}")
        assert w_min >= -expected_bound - 1e-5
        assert w_max <= expected_bound + 1e-5
        print("Layer 1 initialization check passed.")
    
    # Training data
    x_train = jr.uniform(train_key, (1000, 1), minval=-1, maxval=1)
    y_train = target_function(x_train)
    
    # Validation data (dense grid for plotting)
    x_val = jnp.linspace(-1, 1, 400).reshape(-1, 1)
    y_val = target_function(x_val)
    
    # Optimization
    optim = optax.adam(1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((pred_y - y) ** 2)
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print("Starting training...")
    for i in range(2000):
        loss, model, opt_state = make_step(model, opt_state, x_train, y_train)
        if i % 200 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")
            
    print(f"Final Loss: {loss:.6f}")
    
    # Prediction
    pred_y = jax.vmap(model)(x_val)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x_val, y_val, label='Ground Truth', linewidth=2)
    plt.plot(x_val, pred_y, '--', label='SIREN Prediction', linewidth=2)
    plt.title('SIREN Function Approximation')
    plt.legend()
    plt.savefig('siren_verification.png')
    print("Verification plot saved to 'siren_verification.png'")
    
    return loss

if __name__ == "__main__":
    final_loss = train_siren()
    if final_loss < 1e-2:
        print("SUCCESS: SIREN converged to low error.")
    else:
        print("WARNING: SIREN did not converge well.")
