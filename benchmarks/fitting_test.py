import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models import c1_stick
from dmipy_jax.fitting import VoxelFitter

# 1. Setup Data
n_voxels = 100_000
print(f"Generating data for {n_voxels} voxels...")
bvals = jnp.array([1000.0] * 30 + [2000.0] * 30)
bvecs = jnp.array(np.random.randn(60, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)

# Generate Truth
true_mus = jnp.array(np.random.randn(n_voxels, 3))
true_mus = true_mus / jnp.linalg.norm(true_mus, axis=1, keepdims=True)
true_lambda = 1.7e-3
# Generate Signal
signal = jax.vmap(c1_stick, in_axes=(None, None, 0, None))(bvals, bvecs, true_mus, true_lambda)
signal = signal + 0.05 * np.random.randn(*signal.shape)

# 2. Define the Model Wrapper for the Fitter
@jax.jit
def model_to_fit(bval, bvec, mu_x, mu_y, mu_z):
    mu = jnp.array([mu_x, mu_y, mu_z])
    mu = mu / (jnp.linalg.norm(mu) + 1e-6)
    return c1_stick(bval, bvec, mu, 1.7e-3)

# 3. Instantiate Fitter
fitter = VoxelFitter(model_to_fit, parameter_ranges=[(-1., 1.), (-1., 1.), (-1., 1.)])

# 4. Vectorize the Fit
fit_batch = jax.vmap(fitter.fit, in_axes=(0, None, None, 0))

# 5. Run Benchmark
init_guess = jnp.array([1.0, 0.0, 0.0])
init_guesses = jnp.tile(init_guess, (n_voxels, 1))

print("Compiling Fitter...")
# FIX: Select [0] (params) to block on the array, not the tuple
fit_batch(signal[:10], bvals, bvecs, init_guesses[:10])[0].block_until_ready()

print("Fitting 100k Voxels...")
start = time.time()
params, state = fit_batch(signal, bvals, bvecs, init_guesses)
params.block_until_ready()
end = time.time()

print(f"🧩 Fitting Speed: {n_voxels / (end - start):.0f} voxels/sec")
