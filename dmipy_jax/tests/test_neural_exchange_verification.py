
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from dmipy_jax.biophysics.neural_exchange import KargoModel, NeuralExchangeRate, simulate_kargo_signal
import matplotlib.pyplot as plt

def verify_neural_exchange():
    print("Initializing Neural Exchange Verification...")
    
    # 1. Setup Model
    key = jax.random.PRNGKey(42)
    key_nlp, key_init = jax.random.split(key)
    
    neural_exch = NeuralExchangeRate(key=key_nlp)
    
    # KargoModel requires D_intra, D_extra, etc.
    model = KargoModel(
        exchange_network=neural_exch,
        D_intra=2.0e-9, # m^2/s
        D_extra=1.0e-9,
        f_intra=0.5,
        t2_intra=0.1,
        t2_extra=0.1
    )
    
    # 2. Setup Data
    bvals = jnp.array([0.0, 1000.0, 2000.0, 3000.0]) # s/mm^2
    TE = 0.1 # 100 ms
    
    # 3. Forward Pass Check
    print("Running forward pass...")
    # simulate_kargo_signal(model, bvals, TE, delta=..., Delta=...)
    signals = simulate_kargo_signal(model, bvals, TE, delta=0.01, Delta=0.02)
    print("Signals:", signals)
    
    assert signals.shape == bvals.shape
    assert not jnp.any(jnp.isnan(signals))
    
    # 4. Gradient Check
    print("Checking differentiability...")
    
    # Target signal (synthetic decay)
    target = jnp.exp(-bvals * 1e-3) # Just a dummy target
    
    def loss_fn(model):
        pred = simulate_kargo_signal(model, bvals, TE)
        return jnp.mean((pred - target)**2)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    print("Initial Loss:", loss)
    
    # Check if gradients are non-zero/finite
    has_finite_grads = False
    
    # Traverse gradients
    leaves = jax.tree_util.tree_leaves(grads)
    for leaf in leaves:
        if leaf is not None and jnp.sum(jnp.abs(leaf)) > 0:
            has_finite_grads = True
            break
            
    if has_finite_grads:
        print("SUCCESS: Gradients are flowing.")
    else:
        print("WARNING: Gradients might be zero.")
        
    # 5. Simple Optimization Loop
    print("Running optimization loop (5 steps)...")
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for i in range(5):
        model, opt_state, l = step(model, opt_state)
        print(f"Step {i+1}, Loss: {l:.6f}")
        
    print("Verification Complete.")

if __name__ == "__main__":
    verify_neural_exchange()
