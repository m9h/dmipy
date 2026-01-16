import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models import c1_stick, g1_ball, g2_zeppelin

# 1. Generate Data (Same as before)
n_voxels = 1_000_000
n_dirs = 100
print(f"Generating {n_voxels} voxels with {n_dirs} directions...")
bvals = jnp.array(np.ones(n_dirs) * 3000.0)
bvecs = jnp.array(np.random.randn(n_dirs, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)

# 2. Define Inputs (GPU Arrays)
# We simulate a "Field of Tensors" (e.g. White Matter)
mu_field = jnp.array(np.random.randn(n_voxels, 3))
mu_field = mu_field / jnp.linalg.norm(mu_field, axis=1, keepdims=True)
lambda_par = 1.7e-3
lambda_perp = 0.2e-3

# 3. The "NODDI-like" Kernel (Stick + Ball + Zeppelin mixture)
# This is what actually runs during a fit
@jax.jit
def noddi_forward_pass(mu, bval, bvec):
    # Weights (Fixed for benchmark)
    f_intra = 0.6
    f_iso = 0.1
    f_extra = 0.3
    
    # Calculate components
    s_stick = c1_stick(bval, bvec, mu, lambda_par)
    s_ball = g1_ball(bval, bvec, 3.0e-3)
    s_zep = g2_zeppelin(bval, bvec, mu, lambda_par, lambda_perp)
    
    # Mix
    return f_intra * s_stick + f_iso * s_ball + f_extra * s_zep

# 4. Vectorize over voxels
# in_axes: (0, None, None) -> First arg (mu) is different per voxel, others are shared
batch_noddi = jax.vmap(noddi_forward_pass, in_axes=(0, None, None))

# 5. Run Benchmark
print("Compiling Real Physics...")
batch_noddi(mu_field[:100], bvals, bvecs).block_until_ready()

print("Running 1 Million Voxels...")
start = time.time()
res = batch_noddi(mu_field, bvals, bvecs)
res.block_until_ready()
end = time.time()

print(f"🚀 Real Physics Speed: {n_voxels / (end - start):.0f} voxels/sec")
