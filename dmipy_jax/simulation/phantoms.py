
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
from typing import Callable, List
from jaxtyping import Array, Float, PRNGKeyArray

class ScoreNetwork(eqx.Module):
    """
    Time-Dependent Score Network for SGM.
    Predicts the score: grad_x log p_t(x).
    """
    mlp: eqx.nn.MLP
    
    def __init__(self, key, data_dim: int, hidden_size: int = 64, depth: int = 3):
        # Input: data_dim + 1 (time)
        self.mlp = eqx.nn.MLP(
            in_size=data_dim + 1,
            out_size=data_dim,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.swish, 
            key=key
        )
        
    def __call__(self, t: Float[Array, ""], x: Float[Array, " D"]) -> Float[Array, " D"]:
        # t is scalar, x is (D,)
        # Broadcast t to match x batch if needed, but here x is (D,)
        t_arr = jnp.array([t])
        net_in = jnp.concatenate([x, t_arr])
        return self.mlp(net_in)

class SGM(eqx.Module):
    """
    Score-Based Generative Model (VP-SDE).
    Wrapper class containing ScoreNetwork and SDE logic.
    """
    score_net: ScoreNetwork
    beta_min: float = 0.1
    beta_max: float = 20.0
    data_dim: int
    
    def __init__(self, key, data_dim: int):
        self.score_net = ScoreNetwork(key, data_dim)
        self.data_dim = data_dim
        
    def beta(self, t):
        """Linear beta schedule"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
        
    def drift(self, t, y, args):
        # VP-SDE Drift: f(x,t) = -0.5 * beta(t) * x
        return -0.5 * self.beta(t) * y
        
    def diffusion(self, t, y, args):
        # VP-SDE Diffusion: g(t) = sqrt(beta(t))
        # Returns scalar g to be broadcasted or diagonal matrix?
        # diffrax.ControlTerm with VirtualBrownianTree(shape=(D,)) needs g of shape (D, D) or scalar compatible.
        # If g is scalar, diffrax broadacsts.
        return jnp.sqrt(self.beta(t))
        
    def loss_fn(self, model: "SGM", x0: Array, t: float, key: PRNGKeyArray):
        """
        Denoising Score Matching Loss.
        """
        # 1. Sample eps
        eps = jax.random.normal(key, x0.shape)
        
        # 2. Perturb data x_t (VP-SDE Marginal)
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = jnp.exp(log_mean_coeff)
        std = jnp.sqrt(1. - jnp.exp(2. * log_mean_coeff))
        
        x_t = mean_coeff * x0 + std * eps
        
        # 3. Predict Score
        score = model.score_net(t, x_t)
        
        # 4. Target Score & Loss
        # || score * std + eps ||^2
        loss = jnp.sum((score * std + eps)**2)
        return loss

    def sample(self, key, n_samples: int, t1: float = 1.0):
        """
        Generate samples by solving Reverse SDE.
        """
        data_dim = self.data_dim
        
        # Initial sample from N(0, I)
        k_init, k_batch = jax.random.split(key)
        y0 = jax.random.normal(k_init, (n_samples, data_dim))
        keys = jax.random.split(k_batch, n_samples)
        
        def single_sample(y_start, k):
            def reverse_drift(t, y, args):
                 # y is (D,)
                s = self.score_net(t, y)
                f = self.drift(t, y, args)
                g = self.diffusion(t, y, args) # scalar
                return f - g**2 * s
                
            def reverse_diffusion(t, y, args):
                g = self.diffusion(t, y, args) # scalar
                return jnp.full_like(y, g) # (D,) vector for WeaklyDiagonal
                
            solver = diffrax.EulerHeun()
            dt0 = -0.01
            
            # Brownian Path: Defined on [0, 1]. Shape (D,)
            brownian = diffrax.VirtualBrownianTree(
                t0=0.0, t1=1.0, tol=1e-3, shape=(data_dim,), key=k
            )
            
            # Use WeaklyDiagonalControlTerm for elementwise diffusion * noise
            terms = diffrax.MultiTerm(
                diffrax.ODETerm(reverse_drift),
                diffrax.WeaklyDiagonalControlTerm(reverse_diffusion, brownian)
            )
            
            sol = diffrax.diffeqsolve(
                terms, 
                solver, 
                t0=t1, 
                t1=1e-3, 
                dt0=dt0, 
                y0=y_start,
                # Use fixed step size for SDE sampling stability
                stepsize_controller=diffrax.ConstantStepSize()
            )
            
            return sol.ys[-1]

        return jax.vmap(single_sample)(y0, keys)

if __name__ == "__main__":
    print("Running Score-Based Phantom Generator Verification...")
    
    # 1. Toy Data: Mixture of Gaussians (2D)
    # Mode 1: [2, 2], Mode 2: [-2, -2]
    batch_size = 128
    data_dim = 2
    
    def get_batch(key, size):
        k1, k2, k_sel = jax.random.split(key, 3)
        d1 = jax.random.normal(k1, (size, 2)) * 0.5 + 2.0
        d2 = jax.random.normal(k2, (size, 2)) * 0.5 - 2.0
        sel = jax.random.bernoulli(k_sel, 0.5, (size, 1))
        return jnp.where(sel, d1, d2)

    # 2. Init Model
    key_init = jax.random.PRNGKey(42)
    model = SGM(key_init, data_dim)
    optimizer = optax.adam(1e-3)
    
    # Filter for learnable parameters (arrays)
    # optax.init expects a PyTree of arrays.
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # 3. Training Loop
    @eqx.filter_jit
    def step(model, opt_state, batch, k):
        def single_loss(x, k_i):
            k_t, k_loss = jax.random.split(k_i)
            t_val = jax.random.uniform(k_t, (), minval=1e-5, maxval=1.0)
            return model.loss_fn(model, x, t_val, k_loss)
            
        loss_val_fn = lambda m: jnp.mean(jax.vmap(single_loss)(batch, jax.random.split(k, len(batch))))
        loss, grads = eqx.filter_value_and_grad(loss_val_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Training SGM...")
    key_train = jax.random.PRNGKey(0)
    for i in range(1000):
        key_batch, key_step, key_train = jax.random.split(key_train, 3)
        batch = get_batch(key_batch, batch_size)
        model, opt_state, loss = step(model, opt_state, batch, key_step)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
            
    # 4. Generate
    print("Generating Samples...")
    key_gen = jax.random.PRNGKey(100)
    gen_samples = model.sample(key_gen, n_samples=200)
    
    # 5. Check Stats
    mean_gen = jnp.mean(gen_samples, axis=0)
    print(f"Generated Mean: {mean_gen} (Expected near 0 for balanced modes)")
    
    pos_samples = jnp.sum((gen_samples[:, 0] > 0) & (gen_samples[:, 1] > 0))
    neg_samples = jnp.sum((gen_samples[:, 0] < 0) & (gen_samples[:, 1] < 0))
    ratio = pos_samples / (pos_samples + neg_samples + 1e-9)
    print(f"Ratio of Positive/Negative Modes: {ratio:.2f} (Expected ~0.5)")
    
    if 0.4 < ratio < 0.6:
        print("SUCCESS: Generated distribution captures multimodal structure.")
    else:
        print("WARNING: Mode collapse or bad training.")
