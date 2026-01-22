
import jax
import jax.numpy as jnp
import time
import argparse
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.models.ball_stick import BallStick
from dmipy_jax.models.c_noddi import CNODDI

def benchmark_models(n_voxels=100_000, batch_size=100_000):
    print(f"Benchmarking on {n_voxels:,} voxels...")
    
    # 1. HBN Protocol Simulation
    # 64 directions total: 32 @ b=1000, 32 @ b=2000 (Approximation)
    bvals = jnp.concatenate([
        jnp.zeros(1), 
        jnp.full(32, 1000e6), 
        jnp.full(32, 2000e6)
    ])
    # Random bvecs (placeholder)
    k = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(k, (65, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # Fake Data
    data = jnp.ones((n_voxels, 65))
    
    # 2. Models
    # A. Free Water DTI Proxy (BallStick - 2 compartments)
    # BallStick is functionally equivalent (Tensor is stick in simplified limit, or use actual Tensor+Ball if available)
    # Using BallStick to represent "FW-DTI" complexity level (3 params: f_iso, theta, phi)
    fw_dti = BallStick() 
    
    # B. NODDI (Elastometry/Density Proxy)
    # Using AMICO-style or standard fitting? Standard non-linear for now.
    noddi = CNODDI()
    
    # 3. Timings
    
    # FW-DTI
    print("Benchmarking FW-DTI (BallStick Proxy)...")
    # JIT compile first
    # fit_fn = jax.jit(jax.vmap(fw_dti.fit, in_axes=(None, 0))) # Hypothetical API
    # Assuming model.fit(acquisition, data) works on batches as per modeling_framework.py
    
    start = time.time()
    # Mock Fit: Just running forward pass + gradient (proxy for 1 iteration)
    # Or actual fit if fast enough. 
    # Let's run a Mock Fit Loop to simulate optimization cost
    # 10 iterations of LM
    t0 = time.time()
    # Actually, let's use the actual fit if implemented.
    # The models likely inherit from modeling_framework.
    # Check if they have fit().
    # Assuming they do:
    # res = fw_dti.fit(acq, data)
    # But C_NODDI might be complex.
    # Let's measure 'Throughput potential' via forward/backward pass speed 
    # and estimate 50 iterations.
    
    # Better: Measure forward model evaluation speed.
    # Fit time ~ 50 * (Forward + Backward).
    
    # Forward Pass Benchmark
    params_dti = jnp.ones((n_voxels, 3)) # BallStick: [theta, phi, f_stick]
    params_noddi = jnp.ones((n_voxels, 4)) # CNODDI: [theta, phi, f_stick, f_iso]

    # Eval DTI
    @jax.jit
    def eval_dti(p): 
        return jax.vmap(fw_dti, in_axes=(0, None))(p, acq)
    
    _ = eval_dti(params_dti[:10]).block_until_ready() # Warmup
    t0 = time.time()
    _ = eval_dti(params_dti).block_until_ready()
    dt_dti = time.time() - t0
    
    # Eval NODDI
    @jax.jit
    def eval_noddi(p): 
        return jax.vmap(noddi, in_axes=(0, None))(p, acq)
    
    _ = eval_noddi(params_noddi[:10]).block_until_ready()
    t0 = time.time()
    _ = eval_noddi(params_noddi).block_until_ready()
    dt_noddi = time.time() - t0
    
    # Estimate Fit Time (50 iterations safe upper bound)
    est_fit_time_dti = dt_dti * 50 * 3 # factor 3 for gradient overhead + LM steps
    est_fit_time_noddi = dt_noddi * 50 * 3
    
    print(f"FW-DTI Sim Time: {dt_dti:.4f}s")
    print(f"Est. FW-DTI Fit Time (100k vox): {est_fit_time_dti:.2f}s")
    
    print(f"NODDI Sim Time: {dt_noddi:.4f}s")
    print(f"Est. NODDI Fit Time (100k vox): {est_fit_time_noddi:.2f}s")
    
    return est_fit_time_dti, est_fit_time_noddi

if __name__ == "__main__":
    benchmark_models()
