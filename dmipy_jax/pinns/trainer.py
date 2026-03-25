"""
GB200-Optimized PINN Trainer (Equinox Version).
Uses shard_map for distributed collocation sampling and optax for mixed-precision optimization.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

class GB200PINNTrainer:
    """
    Trainer optimizing PINN performance on NVIDIA Grace Blackwell.
    Key optimizations:
    1. shard_map: Generates collocation points LOCALLY on each device (Keys sharded).
    2. Mixed Precision: BF16 for Neural Network limit, Float32 for PDE Residuals.
    3. Equinox: Uses Equinox for model definition and parameter handling.
    """
    
    def __init__(self, loss_fn, sampler, optimizer, mesh):
        """
        Args:
            loss_fn: Function loss(model, batch) -> scalar. 
                     Note: Takes the FULL model object, not just params.
            sampler: DynamicCollocationSampler instance.
            optimizer: Optax optimizer.
            mesh: jax.sharding.Mesh instance.
        """
        self.loss_fn = loss_fn
        self.sampler = sampler
        self.optimizer = optimizer
        self.mesh = mesh
        
    def create_train_state(self, key, model):
        """
        Initializes TrainState compatible with Equinox.
        Args:
            key: Initialization key (not used for Equinox model init, assuming model passed in is already init).
            model: An initialized equinox.Module.
            
        Returns:
            Tuple (state, static)
            state: dict with 'params', 'opt_state', 'step'. (All ARRAYS/Pytrees of Arrays)
            static: Non-array part of the model (Functions, etc).
        """
        # Partition model into trainable and static parts
        params, static = eqx.partition(model, eqx.is_array)
        
        # Initialize Optimizer
        opt_state = self.optimizer.init(params)
        
        state = {
            'params': params,
            'opt_state': opt_state,
            'step': 0
        }
        return state, static

    def make_step_fn(self, static):
        """
        Creates the sharded step function using shard_map.
        
        Args:
            static: The static part of the model (captured in closure).
        """
        
        def train_step(state, key_batch):
            """
            Per-device step function.
            state: Replicated.
            key_batch: Sharded.
            """
            # Recombine model
            # Note: We recombine inside the loss function usually, or here.
            # To compute gradients w.r.t params, we need a function of params.
            
            # 1. Generate Data ON DEVICE
            step_key = key_batch[0] 
            local_batch_size = 4096 # Fixed batch per device
            
            points = self.sampler.sample(step_key, batch_size=local_batch_size, params=None) # Dynamic not wired yet
            
            # 2. Compute Loss & Gradients
            def loss_f(params):
                model = eqx.combine(params, static)
                # Loss function expects (model, batch)
                loss = self.loss_fn.evaluate(model, {'pde_domain': points})
                return loss
                
            loss, grads = jax.value_and_grad(loss_f)(state['params'])
            
            # 3. Cross-Device Reduction
            grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='data'), grads)
            loss = jax.lax.pmean(loss, axis_name='data')
            
            return loss, grads

        # Check rep disabled for pmean
        @jax.jit
        def sharded_step(state, key_batch):
            mesh_axis_name = 'data'
            
            loss, grads = shard_map(
                train_step,
                mesh=self.mesh,
                in_specs=(P(), P(mesh_axis_name)),
                out_specs=(P(), P()),
                check_rep=False
            )(state, key_batch)
            
            # Update
            updates, new_opt_state = self.optimizer.update(grads, state['opt_state'], state['params'])
            new_params = optax.apply_updates(state['params'], updates)
            
            return {
                'params': new_params,
                'opt_state': new_opt_state,
                'step': state['step'] + 1
            }, loss
        
        return sharded_step
