# Physics Simulation with Dmipy-JAX

Dmipy-JAX includes a high-performance Monte Carlo simulator powered by JAX. This allows you to simulate diffusion in complex geometries using GPU acceleration.

## 1. Setup

Import the necessary modules from `dmipy_jax.simulation`.

```python
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.simulation.monte_carlo import simulate_ground_truth, cylinder_sdf
```

## 2. Define Geometry

The simulator uses Signed Distance Functions (SDFs) to define geometry. Negative values are inside the fluid, positive values are walls.

```python
# Radius of 5 microns
radius = 5.0e-6 

# Define SDF for a cylinder
# We use partial application to bake in the radius
from functools import partial
geometry_sdf = partial(cylinder_sdf, radius=radius)
```

## 3. Initialize Particles

We need a function to generate initial particle positions. For a cylinder, we can initialize them uniformly inside.

```python
def initialization_func(key, n_particles):
    # Simplified initialization: mostly near center for demo
    # In practice, use rejection sampling to fill circle uniformly
    return jax.random.uniform(key, (n_particles, 3), minval=-radius/2, maxval=radius/2)
```

## 4. Compile Simulator

Generate the optimized simulation function for this geometry.

```python
simulator = simulate_ground_truth(geometry_sdf, initialization_func)
```

## 5. Define Gradient Waveform

Create a simple Pulse Gradient Spin Echo (PGSE) waveform.

```python
# Time resolution
dt = 1e-5 
duration = 0.05 # 50 ms
n_steps = int(duration / dt)

# Simple gradient pulse
G_max = 0.04 # 40 mT/m
gradients = jnp.zeros((n_steps, 3))
# Apply gradient in X direction for first 10ms
gradients = gradients.at[:1000, 0].set(G_max)
```

## 6. Run Simulation

Run the simulation with JAX acceleration.

```python
key = jax.random.PRNGKey(0)
D_water = 3.0e-9  # Diffusivity
N_particles = 100_000

# First run compiles (may be slow)
signal = simulator(gradients, D_water, dt, N_particles, key)

print(f"Simulated Signal Attenuation: {signal:.4f}")
```

## 7. Performance

Because the simulator is JIT-compiled, subsequent runs with different parameters (e.g., different `D` or `gradients`) are extremely fast, especially on GPU.
