import jax
import jax.numpy as jnp
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.fitting.physics_consistency import physics_loss

def test_physics_loss_gradients():
    """
    Verify that physics_loss is differentiable and gradients flow back to sr_signal.
    """
    
    # 1. Setup Ground Truth
    bvals = jnp.array([1000.0, 1000.0, 1000.0, 1000.0])
    bvecs = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.707, 0.707, 0.0]
    ])
    
    model = Stick()
    
    # Ground truth parameters: mu along X-axis, diffusivity 2e-9
    mu_gt = jnp.array([jnp.pi/2, 0.0]) # Theta, Phi (Cartesian [1, 0, 0])
    lambda_par_gt = 2e-9
    
    # Generate "SR" signal (perfectly physical)
    # Stick model call signature: (bvals, bvecs, mu=..., lambda_par=...)
    gt_signal = model(bvals, bvecs, mu=mu_gt, lambda_par=lambda_par_gt)
    
    # 2. Perturb Signal (Simulate Neural Net output that is slightly unphysical)
    # We want to optimize this signal to be physical
    sr_signal = gt_signal + 0.1 # Add offset
    
    # 3. Define Loss Function wrt sr_signal
    def loss_wrapper(signal):
        return physics_loss(signal, bvals, bvecs, model)
        
    # 4. Compute Gradient
    grad_fn = jax.grad(loss_wrapper)
    gradients = grad_fn(sr_signal)
    
    print("SR Signal:", sr_signal)
    print("Gradients:", gradients)
    
    # Check if gradients are non-zero and finite
    assert jnp.all(jnp.isfinite(gradients))
    assert not jnp.allclose(gradients, 0.0)
    
    # 5. Check if taking a step improves physics consistency
    # (Simple Gradient Descent step)
    new_signal = sr_signal - 0.1 * gradients
    loss_old = loss_wrapper(sr_signal)
    loss_new = loss_wrapper(new_signal)
    
    print(f"Loss Old: {loss_old}, Loss New: {loss_new}")
    
    assert loss_new < loss_old

if __name__ == "__main__":
    test_physics_loss_gradients()
