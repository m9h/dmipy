import pytest
import jax
import jax.numpy as jnp
import numpy as np
import chex
from dmipy_jax.tractography.tracking import track, evaluate_odf, step_fn, WalkerState

def get_sphere_vectors(n=64):
    """Generate random unit vectors on sphere."""
    # Simple Fibonacci sphere or just random for testing
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return jnp.array(vecs)

def test_evaluate_odf_shapes():
    """Test that evaluate_odf returns correct shapes."""
    batch_size = 10
    n_dirs = 16
    n_coeffs = 6
    x, y, z = 5, 5, 5
    
    sh_coeffs = jnp.ones((x, y, z, n_coeffs))
    pos = jnp.zeros((batch_size, 3)) + 2.0 # Center of volume
    sphere_dirs = get_sphere_vectors(n_dirs)
    sh_basis = jnp.ones((n_dirs, n_coeffs))
    
    amps, gfa = evaluate_odf(sh_coeffs, pos, sphere_dirs, sh_basis)
    
    assert amps.shape == (batch_size, n_dirs)
    assert gfa.shape == (batch_size, 1)

def test_track_compilation_and_shapes():
    """Test that track function compiles and returns correct history shape."""
    batch_size = 5
    n_steps = 20
    x, y, z, n_coeffs = 10, 10, 10, 15
    n_dirs = 32
    
    key = jax.random.PRNGKey(0)
    sh_coeffs = jax.random.normal(key, (x, y, z, n_coeffs))
    seeds = jnp.array([[5.0, 5.0, 5.0]] * batch_size)
    seed_dirs = jnp.array([[1.0, 0.0, 0.0]] * batch_size)
    
    sphere_dirs = get_sphere_vectors(n_dirs)
    sh_basis = jax.random.normal(key, (n_dirs, n_coeffs))
    
    # JIT compile the track function
    jit_track = jax.jit(track, static_argnames=['max_steps'])
    
    history = jit_track(
        sh_coeffs, seeds, seed_dirs, sphere_dirs, sh_basis, 
        key, step_size=0.5, max_steps=n_steps
    )
    
    # Shape should be (n_steps + 1, batch, 3)
    assert history.shape == (n_steps + 1, batch_size, 3)
    # Check for NaNs
    assert not jnp.any(jnp.isnan(history))

def test_gradients_propagate():
    """Test that gradients flow from endpoint back to SH coefficients."""
    # We define a loss function depending on the final position
    # and check if gradient w.r.t sh_coeffs is non-zero.
    
    x, y, z, n_coeffs = 8, 8, 8, 6
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)
    
    input_sh = jax.random.normal(k1, (x, y, z, n_coeffs)) # Variable to differeniate
    
    seeds = jnp.array([[4.0, 4.0, 4.0]])
    seed_dirs = jnp.array([[1.0, 0.0, 0.0]])
    sphere_dirs = get_sphere_vectors(12)
    sh_basis = jax.random.normal(k2, (12, n_coeffs)) # Random basis allows all dirs to depend on coeffs
    
    def loss_fn(coeffs):
        history = track(
            coeffs, seeds, seed_dirs, sphere_dirs, sh_basis, 
            k3, step_size=0.5, max_steps=10, temperature=1.0 
        )
        final_pos = history[-1]
        # Target: maximize X coordinate
        return -jnp.mean(final_pos[..., 0])
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(input_sh)

    # Gradient should exist (non-zero/non-nan)
    assert not jnp.any(jnp.isnan(grads))
    
    # Check that gradient is non-zero
    grad_mag = jnp.sum(jnp.abs(grads))
    print(f"Gradient magnitude: {grad_mag}")
    
    # We check if *some* gradient is non-zero.
    # Note: If the walker doesn't move or steps are independent of coeffs, it would be zero.
    # But step direction -> logits -> amplitudes -> coeffs. So chain rule applies.
    
    assert jnp.sum(jnp.abs(grads)) > 0.0

def test_forward_cone():
    """Test that walker doesn't turn backwards."""
    # Evaluate a single step.
    # Current dir: [1, 0, 0]
    # Sphere dirs includes [-1, 0, 0] (backward) and [1, 0, 0] (forward)
    
    state = WalkerState(
        pos=jnp.array([[0.0, 0.0, 0.0]]),
        dir=jnp.array([[1.0, 0.0, 0.0]]),
        alive=jnp.array([[1.0]])
    )
    
    # Amplitudes favour backward to tempt it
    # sphere_dirs: index 0 is forward, index 1 is backward
    sphere_dirs = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    # Make backward amplitude huge
    amplitudes_mock = jnp.array([[1.0, 1000.0]]) 
    # We override evaluate_odf or pass coeffs that produce this?
    # Easier to just verify step_fn logic logic if we mock things, but integration testing step_fn:
    
    # We can design SH/basis such that index 1 gets huge value.
    sh_coeffs = jnp.zeros((1, 1, 1, 2)) # X,Y,Z irrelevant for this mock if we are careful
    sh_coeffs = sh_coeffs.at[0,0,0, :].set(jnp.array([1.0, 100.0]))
    sh_basis = jnp.eye(2) # coeffs directly become amplitudes
    
    key = jax.random.PRNGKey(42)
    
    # With masking, it should NOT pick backward (idx 1), even if amplitude is high.
    new_state = step_fn(
        state, sh_coeffs, sphere_dirs, sh_basis, key, 
        max_angle_deg=45.0, temperature=0.1
    )
    
    # New dir should be close to forward [1, 0, 0]
    # Dot product with forward should be high
    dot = jnp.dot(new_state.dir, jnp.array([1.0, 0.0, 0.0]))
    assert dot > 0.9 # Should stay roughly forward
