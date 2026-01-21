
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import equinox as eqx
import optax
from dmipy_jax.core.networks import SIREN

def verify_siren():
    key = jr.PRNGKey(0)
    
    # 1. Initialize Network
    model = SIREN(
        in_features=1,
        out_features=1,
        hidden_features=256,
        hidden_layers=3,
        key=key
    )
    
    print("Network initialized successfully.")
    
    # 2. Check Activation Statistics (Sitzmann et al. Check)
    # Feed random normal input
    x = jr.normal(key, (1000, 1))
    
    # We need to manually inspect layers for this valid check or just check output distribution
    # For now, let's just check the forward pass works and output is reasonable
    y_pred = jax.vmap(model)(x)
    print(f"Output Mean: {jnp.mean(y_pred):.4f}")
    print(f"Output Std: {jnp.std(y_pred):.4f}")
    
    # 3. Simple Function Fitting Task
    # Target: f(x) = sin(30 * x)
    def target_func(x):
        return jnp.sin(30 * x)
    
    x_train = jnp.linspace(-1, 1, 200)[:, None]
    y_train = target_func(x_train)
    
    @eqx.filter_value_and_grad
    def loss_fn(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y)**2)
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print("\nStarting training loop...")
    for i in range(1001):
        loss, model, opt_state = make_step(model, opt_state, x_train, y_train)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")
            
    print("\nTraining complete.")
    
    # Check final error
    final_loss, _ = loss_fn(model, x_train, y_train)
    if final_loss < 0.01:
        print("SUCCESS: SIREN fitted high frequency sine wave.")
    else:
        print("FAILURE: SIREN failed to fit high frequency sine wave.")

if __name__ == "__main__":
    verify_siren()
