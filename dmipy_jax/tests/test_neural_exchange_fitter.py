import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest
from dmipy_jax.biophysics.neural_exchange import KargoModel, NeuralExchangeRate, simulate_kargo_signal
from dmipy_jax.fitting.neural_exchange_fitter import NeuralExchangeTrainer

def test_neural_exchange_fitter_convergence():
    key = jax.random.PRNGKey(42)
    key_gt, key_init = jax.random.split(key)
    
    # 1. Ground Truth Model
    # Let's say we have a known exchange network (randomly initialized but fixed)
    gt_network = NeuralExchangeRate(key_gt, width_size=32, depth=2)
    
    gt_model = KargoModel(
        exchange_network=gt_network,
        D_intra=2.0e-9,
        D_extra=1.0e-9,
        f_intra=0.5,
        t2_intra=0.050,
        t2_extra=0.050
    )
    
    # 2. Simulate Data
    # 20 b-values from 0 to 3000
    bvals = jnp.linspace(0, 3000, 20)
    TE = 0.100 # 100ms
    
    target_signals = simulate_kargo_signal(gt_model, bvals, TE)
    
    # 3. Initialize Student Model
    # Different random initialization for network
    init_network = NeuralExchangeRate(key_init, width_size=32, depth=2)
    
    # We can start with somewhat close scalar params to avoid local minima issues if solving everything
    # But let's try to learn them too (or at least f_intra)
    student_model = KargoModel(
        exchange_network=init_network,
         D_intra=1.8e-9, # Slightly off
        D_extra=1.2e-9, # Slightly off
        f_intra=0.4,    # Slightly off
    )
    
    # 4. Setup Trainer
    # Use Adam with a larger learning rate for faster convergence testing
    optimizer = optax.adam(learning_rate=0.01)
    trainer = NeuralExchangeTrainer(student_model, optimizer)
    
    # 5. Determine initial loss
    initial_loss, _ = trainer.compute_loss(student_model, bvals, TE, target_signals)
    print(f"Initial Loss Output: {initial_loss}")
    print(f"Initial D_intra: {student_model.D_intra:.3e}")

    # 6. Fit
    # Run for 500 steps
    n_epochs = 500
    trainer = trainer.fit(bvals, TE, target_signals, n_epochs=n_epochs, print_every=100)
    
    # 7. Check final loss
    final_loss, _ = trainer.compute_loss(trainer.model, bvals, TE, target_signals)
    print(f"Final Loss Output: {final_loss}")
    print(f"Final D_intra: {trainer.model.D_intra:.3e}")
    
    assert final_loss < initial_loss, "Loss did not decrease"
    # Ideally efficient fitting should get it quite low
    # But 200 steps might not be enough for perfect convergence of an MLP
    # We just check for significant improvement
    assert final_loss < 0.1 * initial_loss, "Loss did not improve significantly"

if __name__ == "__main__":
    test_neural_exchange_fitter_convergence()
