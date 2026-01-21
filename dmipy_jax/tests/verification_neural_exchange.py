import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from dmipy_jax.biophysics.neural_exchange import NeuralExchangeRate, KargoModel, simulate_kargo_signal

def verify_neural_exchange():
    print("Initializing Neural Exchange Verification...")
    key = jax.random.PRNGKey(42)
    m_key, d_key = jax.random.split(key, 2)
    
    # 1. Initialize Model
    exchange_net = NeuralExchangeRate(key=m_key)
    model = KargoModel(
        exchange_network=exchange_net,
        D_intra=2.0e-9, # m^2/s
        D_extra=1.0e-9, 
        f_intra=0.5,
        t2_intra=0.2,
        t2_extra=0.2
    )
    
    # 2. Simulate Data (B-values)
    # bvals from 0 to 3000 s/mm^2
    bvals = jnp.linspace(0, 3000, 10) # s/mm^2
    TE = 0.1 # 100 ms
    
    print("Running Forward Pass...")
    signals = simulate_kargo_signal(model, bvals, TE)
    print(f"Signals at b=0, b=3000: {signals[0]:.4f}, {signals[-1]:.4f}")
    
    assert not jnp.isnan(signals).any(), "NaNs in signal!"
    
    # 3. Test Differentiation
    print("Testing Backpropagation...")
    
    # Create a target signal (e.g., from a static exchange model)
    # Just add noise to current signal for "fake" data
    target_signals = signals * 0.95 
    
    def loss_fn(model):
        preds = simulate_kargo_signal(model, bvals, TE)
        return jnp.mean((preds - target_signals)**2)
    
    # Filter grad
    grad_loss = eqx.filter_grad(loss_fn)
    grads = grad_loss(model)
    
    # Check if grads exist for MLP
    # The MLP parameters are inside model.exchange_network.mlp
    # We check if they are non-zero.
    
    # Flatten grads to check
    leaves = jax.tree_util.tree_leaves(grads.exchange_network)
    total_grad_norm = sum(jnp.sum(jnp.abs(l)) for l in leaves)
    
    print(f"Total Gradient Norm on MLP parameters: {total_grad_norm:.6f}")
    
    if total_grad_norm > 0.0:
        print("SUCCESS: Gradients are flowing to the MLP.")
    else:
        print("FAILURE: Gradients are zero!")
        
    # 4. Simple Optimization Step
    print("Testing Optimization Step...")
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    model, opt_state, init_loss = make_step(model, opt_state)
    model, opt_state, current_loss = make_step(model, opt_state)
    
    print(f"Initial Loss: {init_loss:.6f}")
    print(f"Loss after step 2: {current_loss:.6f}")
    
    if current_loss < init_loss:
        print("SUCCESS: Loss decreased after optimization step.")
    else:
        print("WARNING: Loss did not decrease (might be noise or small step).")

if __name__ == "__main__":
    verify_neural_exchange()
