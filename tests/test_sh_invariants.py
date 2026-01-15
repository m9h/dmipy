
import numpy as np
import jax
import jax.numpy as jnp
import time
from dmipy_jax.core.invariants import compute_invariants_jit, associated_legendre

def test_legendre():
    print("Testing Legendre Polynomials...")
    x = 0.5
    # P_1^0(x) = x
    assert np.allclose(associated_legendre(1, 0, x), x)
    # P_2^0(x) = 0.5 * (3x^2 - 1) -> 0.5 * (0.75 - 1) = -0.125
    p20 = 0.5 * (3 * x**2 - 1)
    assert np.allclose(associated_legendre(2, 0, x), p20)
    # P_2^1(x) = -3 * x * sqrt(1-x^2) -> -1.5 * sqrt(0.75)
    # Note: My implementation might have phase differences? Standard is (-1)^m
    # associated_legendre(l, m, x) generally aligns with unnormalized.
    # My implementation: P_1^1 = -sqrt(1-x^2). Correct.
    # P_2^1 = x * 3 * P_1^1 = -3x sqrt(1-x^2). Correct.
    p21 = -3 * x * np.sqrt(1 - x**2)
    assert np.allclose(associated_legendre(2, 1, x), p21)
    print("Legendre Polynomials: Passed")

def simple_stick(bval, bvec, fiber_dir, d_par=1.7e-3):
    # S = exp(-b * d_par * (g.n)^2)
    # cos_theta = dot(g, n)
    cos_theta = np.dot(bvec, fiber_dir)
    return np.exp(-bval * d_par * cos_theta**2)

def generate_random_bvecs(n_dirs=64):
    rng = np.random.RandomState(42)
    bvecs = rng.randn(n_dirs, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    return bvecs

def test_rotational_invariance():
    print("Testing Rotational Invariance...")
    n_dirs = 120
    bvecs = generate_random_bvecs(n_dirs)
    bval = 3000 # High b-value for contrast
    
    # Fiber 1: Z-axis
    fib1 = np.array([0, 0, 1])
    sig1 = simple_stick(bval, bvecs, fib1)
    
    # Fiber 2: X-axis
    fib2 = np.array([1, 0, 0])
    sig2 = simple_stick(bval, bvecs, fib2)
    
    # Fiber 3: Random
    fib3 = np.array([1, 1, 0]) / np.sqrt(2)
    sig3 = simple_stick(bval, bvecs, fib3)
    
    # Stack signals
    signals = np.stack([sig1, sig2, sig3], axis=0) # (3, N_dirs)
    
    # Compute invariants
    invariants = compute_invariants_jit(jnp.array(signals), jnp.array(bvecs), max_order=6)
    
    # Check if P0, P2, P4, P6 are similar across voxels
    print("Invariants for 3 directions:")
    print(invariants)
    
    # We expect numerical invariance.
    # Note: Finite sampling (120 dirs) causes some discretized rotation error.
    # But it should be small (< 5-10%).
    
    diff_12 = jnp.max(jnp.abs(invariants[0] - invariants[1]))
    diff_13 = jnp.max(jnp.abs(invariants[0] - invariants[2]))
    
    print(f"Max Diff (Z vs X): {diff_12:.6f}")
    print(f"Max Diff (Z vs xy): {diff_13:.6f}")
    
    if diff_12 < 0.1 and diff_13 < 0.1:
        print("Rotational Invariance: Passed (within tolerance)")
    else:
        print("Rotational Invariance: Failed or high error")

def benchmark_performance():
    print("Benchmarking on 100,000 voxels...")
    n_vox = 100000
    n_dirs = 64
    bvecs = jnp.array(generate_random_bvecs(n_dirs))
    # Random signals
    rng = np.random.RandomState(0)
    signals = jnp.array(rng.rand(n_vox, n_dirs).astype(np.float32))
    
    # Warmup
    _ = compute_invariants_jit(signals[:100], bvecs, max_order=6)
    
    # Run
    t0 = time.time()
    res = compute_invariants_jit(signals, bvecs, max_order=6)
    res.block_until_ready()
    t1 = time.time()
    
    print(f"Time for {n_vox} voxels: {t1-t0:.4f} s")
    print(f"Voxels per second: {n_vox / (t1-t0):.2e}")

if __name__ == "__main__":
    test_legendre()
    test_rotational_invariance()
    benchmark_performance()
