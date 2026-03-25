"""
Verification Script for High-Performance PINN on GB200 (Equinox Version).
Runs a dummy training loop using the DynamicCollocationSampler and GB200PINNTrainer.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import time

# Import our new modules
from dmipy_jax.pinns.sampling import DynamicCollocationSampler
from dmipy_jax.pinns.trainer import GB200PINNTrainer

# Define a simple MLP using Equinox
class PINN(eqx.Module):
    layers: list
    
    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(4, 128, key=keys[0]),
            eqx.nn.Linear(128, 128, key=keys[1]),
            eqx.nn.Linear(128, 2, key=keys[2]) # Output Real, Imag
        ]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        u = self.layers[-1](x)
        return u[..., 0] + 1j * u[..., 1]

# Define a dummy Loss
# Note: trainer passes (model, batch) to evaluate
class DummyLoss:
    def evaluate(self, model, batch):
        points = batch['pde_domain']
        
        # vmap the model application over the batch
        # Equinox models are not batched by default, one must vmap
        u = jax.vmap(model)(points)
        
        return jnp.mean(jnp.abs(u)**2)

def main():
    print("Setting up GB200 PINN Verification (Equinox)...")
    
    # 1. Mesh Setup
    devices = jax.devices()
    print(f"JAX Devices: {len(devices)}")
    mesh = Mesh(devices, ('data',))
    
    # 2. Sampler
    bounds = ((0.0, -1.0, -1.0, -1.0), (0.1, 1.0, 1.0, 1.0))
    sampler = DynamicCollocationSampler(bounds, interface_fraction=0.1)
    
    # 3. Model & Optimizer
    key = jax.random.PRNGKey(0)
    init_key, train_key = jax.random.split(key)
    
    model = PINN(init_key)
    optimizer = optax.adam(1e-3)
    
    # 4. Trainer (Refactored)
    loss_fn = DummyLoss()
    trainer = GB200PINNTrainer(loss_fn, sampler, optimizer, mesh)
    
    # 5. Initialization
    # Create State (Returns state, static)
    state, static = trainer.create_train_state(key, model)
    
    # Create Step Fn (Requires static)
    step_fn = trainer.make_step_fn(static)
    
    # Run loop
    t0 = time.time()
    n_steps = 100
    
    # Shard keys
    n_devices = len(devices)
    key_batch = jax.random.split(key, n_devices)
    key_sharding = NamedSharding(mesh, P('data'))
    key_batch = jax.device_put(key_batch, key_sharding)
    
    print(f"Starting training on {n_devices} devices...")
    for i in range(n_steps):
        state, loss = step_fn(state, key_batch)
        loss.block_until_ready() # Ensure computation finishes for timing
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss}")
            
    end_time = time.time()
    
    total_points = 4096 * n_devices * N_STEPS
    duration = end_time - start_time
    pps = total_points / duration
    
    print(f"\nVerification Complete.")
    print(f"Total Points Sampled: {total_points}")
    print(f"Duration: {duration:.4f} s")
    print(f"Throughput: {pps:.2e} Points/sec")
    print("Success: GB200 PINN Infrastructure (Equinox) is operational.")

if __name__ == "__main__":
    main()
