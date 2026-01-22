import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import flowjax.bijections as bij
import flowjax.distributions as dist
import flowjax.flows as flows
from typing import Callable, Any, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

class SBITrainer(eqx.Module):
    """
    Trainer for Simulation-Based Inference (SBI) using 'Infinite Data'.
    
    This trainer wraps a Normalizing Flow (NPE) and a Simulator.
    It generates training data on-the-fly using JAX's JIT compilation,
    enabling training on effectively infinite unique samples.
    """
    flow: Any
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState
    simulator: Callable[[PRNGKeyArray, Float[Array, "batch theta_dim"]], Float[Array, "batch signal_dim"]] = eqx.field(static=True)
    prior_sampler: Callable[[PRNGKeyArray, int], Float[Array, "batch theta_dim"]] = eqx.field(static=True)
    
    def __init__(
        self,
        flow: Any,
        optimizer: optax.GradientTransformation,
        simulator: Callable,
        prior_sampler: Callable,
        key: PRNGKeyArray
    ):
        self.flow = flow
        self.optimizer = optimizer
        self.simulator = simulator
        self.prior_sampler = prior_sampler
        
        # Initialize optimizer state
        # We need to filter for trainable parameters in the flow
        params, static = eqx.partition(self.flow, eqx.is_array)
        self.opt_state = self.optimizer.init(params)

    @eqx.filter_jit
    def train_step(
        self, 
        key: PRNGKeyArray,
        batch_size: int,
        noise_std: float = 0.0
    ):
        """
        Performs a single training step with on-the-fly simulation.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        # 1. Sample Theta from Prior
        theta = self.prior_sampler(k1, batch_size)
        
        # 2. Simulate Signal (Infinite Data)
        signal = self.simulator(k2, theta)
        
        # 3. Add Noise
        if noise_std > 0:
            noise = jax.random.normal(k3, signal.shape) * noise_std
            signal = signal + noise
            
        # 4. Compute Loss and Update
        # Loss = - E[log q(theta | signal)]
        
        def loss_fn(flow):
            # Condition the flow on the signal (context)
            # FlowJAX flows usually take (y, condition) where y is the target (theta)
            # log_prob returns the log probability of theta given the context (signal)
            log_probs = flow.log_prob(theta, condition=signal)
            return -jnp.mean(log_probs)

        grads = eqx.filter_grad(loss_fn)(self.flow)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, self.flow)
        new_flow = eqx.apply_updates(self.flow, updates)
        
        loss_val = loss_fn(self.flow)
        
        return new_flow, new_opt_state, loss_val

def create_trainer(
    flow_key: PRNGKeyArray,
    theta_dim: int,
    signal_dim: int,
    simulator: Callable,
    prior_sampler: Callable,
    learning_rate: float = 1e-4,
    hidden_dim: int = 64,
    num_layers: int = 4
) -> SBITrainer:
    """
    Factory function to create an SBITrainer with a standard NSF flow.
    """
    # Create a Neural Spline Flow using FlowJAX
    # We want a conditional flow: p(theta | signal)
    
    # Base distribution is standard normal z ~ N(0, I)
    base_dist = dist.StandardNormal((theta_dim,))
    
    # Bijector: Rational Quadratic Spline
    # We stack multiple layers of splines permuted by simple linear layers
    # The conditioner needs to take 'signal' as input.
    # In FlowJAX, we can use `MaskedAutoregressiveFlow` or `CouplingFlow`.
    # For SBI, Coupling Flows are common.
    
    # NOTE: FlowJAX construction might vary slightly by version. 
    # Using a common pattern for conditional flows.
    
    flow = flows.masked_autoregressive_flow(
        key=flow_key,
        base_dist=base_dist,
        cond_dim=signal_dim,
        flow_layers=num_layers,
        nn_width=hidden_dim,
        nn_depth=2
    )
    
    optimizer = optax.adam(learning_rate)
    
    # Hack to initialize opt_state inside __init__ requires specific handling if passed directly
    # We will initialize it inside the class __init__
    
    return SBITrainer(
        flow=flow,
        optimizer=optimizer,
        simulator=simulator,
        prior_sampler=prior_sampler,
        key=flow_key 
    )

# Independent loop helper
def train_loop(
    trainer: SBITrainer,
    key: PRNGKeyArray,
    num_steps: int,
    batch_size: int,
    noise_std: float = 0.0,
    print_every: int = 100
):
    """
    Runs the training loop.
    Could utilize jax.lax.scan for ultra-fast execution if no printing is needed.
    """
    import time
    
    current_trainer = trainer
    curr_key = key
    
    start_time = time.time()
    
    for i in range(num_steps):
        k_step, curr_key = jax.random.split(curr_key)
        
        # We need to manually unpack and repack the trainer to handle the Equinox update pattern
        # SBITrainer is an eqx.Module, so we can treat it functionally.
        
        flow, opt_state, loss = current_trainer.train_step(k_step, batch_size, noise_std)
        
        # Update the trainer with new state
        current_trainer = eqx.tree_at(
            lambda t: (t.flow, t.opt_state),
            current_trainer,
            (flow, opt_state)
        )
        
        if i % print_every == 0:
            print(f"Step {i}: Loss = {loss:.4f}")
            
    end_time = time.time()
    print(f"Training finished. {num_steps} steps in {end_time - start_time:.2f}s.")
    print(f"Throughput: {num_steps / (end_time - start_time):.1f} steps/sec")
    
    return current_trainer
