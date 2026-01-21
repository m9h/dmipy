import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple, Any

from dmipy_jax.biophysics.neural_exchange import KargoModel, simulate_kargo_signal

class NeuralExchangeTrainer(eqx.Module):
    """
    Trainer for the Neural Exchange Model.
    Optimizes the KargoModel parameters (including the NeuralExchangeRate MLP)
    to match target signals.
    """
    model: KargoModel
    optimizer: optax.GradientTransformation
    opt_state: Any

    def __init__(
        self,
        model: KargoModel,
        optimizer: optax.GradientTransformation,
    ):
        self.model = model
        self.optimizer = optimizer
        # Filter for learnable parameters (inexact arrays)
        # In a real scenario, we might want to freeze D_intra/D_extra and only learn the MLP.
        # But here we initialize optimization state for all diff/float params provided in model.
        # Users can freeze parts using eqx.filter_grad or by constructing a custom filter.
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

    @staticmethod
    @eqx.filter_value_and_grad
    def compute_loss(
        model: KargoModel,
        bvals: jnp.ndarray,
        TE: float,
        target_signals: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes MSE loss between simulated and target signals.
        """
        preds = simulate_kargo_signal(model, bvals, TE)
        loss = jnp.mean((preds - target_signals)**2)
        return loss

    @eqx.filter_jit
    def train_step(
        self,
        bvals: jnp.ndarray,
        TE: float,
        target_signals: jnp.ndarray
    ) -> Tuple['NeuralExchangeTrainer', float]:
        """
        Performs one gradient descent step.
        """
        loss, grads = self.compute_loss(self.model, bvals, TE, target_signals)
        
        # Filter params to match optimizer init structure
        params = eqx.filter(self.model, eqx.is_inexact_array)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, params)
        
        new_model = eqx.apply_updates(self.model, updates)
        
        return eqx.tree_at(
            lambda t: (t.model, t.opt_state),
            self,
            (new_model, new_opt_state)
        ), loss
    
    def fit(
        self,
        bvals: jnp.ndarray,
        TE: float,
        target_signals: jnp.ndarray,
        n_epochs: int = 1000,
        print_every: int = 100
    ) -> 'NeuralExchangeTrainer':
        """
        Runs the training loop for n_epochs.
        """
        trainer = self
        for i in range(n_epochs):
            trainer, loss = trainer.train_step(bvals, TE, target_signals)
            if (i + 1) % print_every == 0:
                # We need to rely on host callback or just print in non-jit context.
                # Since fit is not jitted, we can print.
                print(f"Epoch {i+1}/{n_epochs}, Loss: {loss:.6e}")
        return trainer
