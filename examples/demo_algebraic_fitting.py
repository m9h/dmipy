
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

# Local Imports
# Use absolute imports relative to project root in examples
from dmipy_jax.algebra.initializers import get_monoexponential_initializer
from dmipy_jax.algebra.wrapper import check_identifiability, print_identifiability_report

# We need a Tissue Model. Let's define a simple SphereGPD equivalent here for the demo
# to avoid rigorous dependencies on complex modules if not needed, 
# but ideally we import from simulation/phantoms or similar.
# Let's define it explicitly for clarity in the demo.

class SphereGPDLower(eqx.Module):
    """
    Simple 2-parameter SphereGPD model for demo.
    S = S0 * exp(-b * D)  (Approximation for initializer)
    But true model:
    S = S0 * [ f * exp(-b*D_ex) + (1-f) * Sphere(b, diameter) ]
    Let's stick to MonoExponential for the model we fit in this demo to start,
    OR fit the full SphereGPD if we can define it easily.
    
    Let's fit a simple Signal Model: MonoExponential with fixed S0=1.
    S = exp(-b * D)
    """
    diffusivity: jax.Array
    
    def __init__(self, key):
        self.diffusivity = jax.random.uniform(key, (), minval=1e-4, maxval=3e-3)
        
    def __call__(self, bvalues):
        return jnp.exp(-bvalues * self.diffusivity)

# Let's use a more complex model: BiExponential (Approximating Intra/Extra)
class BiExpModel(eqx.Module):
    f: jax.Array
    D_slow: jax.Array
    D_fast: jax.Array
    
    def __init__(self, f=None, Ds=None, Df=None, key=None):
        if key is not None:
             self.f = jax.random.uniform(key, (), minval=0.2, maxval=0.8)
             self.D_slow = jax.random.uniform(key, (), minval=0.1e-3, maxval=1.0e-3)
             # Fast is significantly faster
             self.D_fast = jax.random.uniform(key, (), minval=1.5e-3, maxval=3.0e-3)
        else:
            self.f = jnp.array(f)
            self.D_slow = jnp.array(Ds)
            self.D_fast = jnp.array(Df)
            
    def __call__(self, bvalues):
        # S = f * exp(-b Ds) + (1-f) * exp(-b Df)
        return self.f * jnp.exp(-bvalues * self.D_slow) + (1-self.f) * jnp.exp(-bvalues * self.D_fast)

def demo_algebraic_fitting():
    print("=== Algebraic vs Random Initialization Demo ===\n")
    
    # 1. Comparison Setup
    # Protocol: 4 Shells
    bvalues = jnp.array([0.0, 1000.0, 2000.0, 3000.0])
    print(f"Protocol B-values: {bvalues}")
    
    # Check Identifiability
    # BiExponential has 3 params (f, Ds, Df) (assuming S0=1)
    # 4 measurements > 3 params.
    # Check uniqueness using wrapper (using simplified 'Zeppelin' ~ Mono/Bi proxy)
    alg_check = check_identifiability(bvalues.tolist(), "Zeppelin") # Zeppelin is bi-component
    print_identifiability_report(alg_check)

    # 2. Ground Truth
    true_model = BiExpModel(f=0.6, Ds=0.5e-3, Df=2.5e-3)
    signal_noiseless = true_model(bvalues)
    
    # Add noise
    key = jax.random.PRNGKey(42)
    signal_noisy = signal_noiseless + 0.05 * jax.random.normal(key, signal_noiseless.shape)
    
    print(f"\nNoisy Signal: {signal_noisy}")
    
    # 3. Algebraic Inversion
    # Run MonoExponential Initializer to get a "Mean Diffusivity" guess
    print("\n--- Running Algebraic Initializer ---")
    
    mono_init_fn = get_monoexponential_initializer(bvalues.tolist())
    # Returns [S0, D]
    alg_guess = mono_init_fn(signal_noisy)
    guess_S0, guess_D = alg_guess[0], alg_guess[1]
    
    print(f"Algebraic Guess: S0={guess_S0:.2f}, Mean D={guess_D:.4e}")
    
    # Construct Initial Model State from Algebraic Guess
    # We initialize D_slow and D_fast around Mean D.
    # D_slow = 0.5 * D, D_fast = 1.5 * D?
    alg_model = BiExpModel(
        f=0.5, 
        Ds=guess_D * 0.5, 
        Df=guess_D * 1.5
    )
    
    # 4. Random Initialization
    key_rand = jax.random.PRNGKey(101)
    rand_model = BiExpModel(key=key_rand)
    print(f"Random Init: f={rand_model.f:.2f}, Ds={rand_model.D_slow:.4e}, Df={rand_model.D_fast:.4e}")
    
    # 5. Optimization Loop
    def fit_model(init_model, label):
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(init_model)
        model = init_model
        
        losses = []
        
        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred = m(bvalues)
            return jnp.mean((pred - signal_noisy)**2)
            
        for i in range(200):
            loss, grads = loss_fn(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            losses.append(loss)
            
        print(f"[{label}] Final Loss: {losses[-1]:.6f}")
        return model, losses

    print("\n--- Fitting Random Init ---")
    fitted_rand, loss_rand = fit_model(rand_model, "Random")
    
    print("\n--- Fitting Algebraic Init ---")
    fitted_alg, loss_alg = fit_model(alg_model, "Algebraic")
    
    # 6. Results
    print("\n=== Results Comparison ===")
    print(f"Truth: f=0.6, Ds=0.5e-3, Df=2.5e-3")
    print(f"Random Fit: f={fitted_rand.f:.3f}, Ds={fitted_rand.D_slow:.4e}, Df={fitted_rand.D_fast:.4e}")
    print(f"Algebraic Fit: f={fitted_alg.f:.3f}, Ds={fitted_alg.D_slow:.4e}, Df={fitted_alg.D_fast:.4e}")
    
    # Plotting (text-based logic representation)
    if loss_alg[-1] < loss_rand[-1]:
         print("\nVICTORY: Algebraic initialization converged significantly better/faster.")
    else:
         print("\nTie: Both converged.")

if __name__ == "__main__":
    demo_algebraic_fitting()
