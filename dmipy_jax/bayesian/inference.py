
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide
from dmipy_jax.bayesian.models import ball_and_sticks_ard
from functools import partial

def fit_voxel_vi(
    key, 
    data, 
    bvals, 
    bvecs, 
    n_fibers=3, 
    ard_weight=1.0, 
    num_steps=1000, 
    learning_rate=1e-2
):
    """
    Fit a single voxel using Variational Inference.
    """
    model = partial(
        ball_and_sticks_ard, 
        bvals=bvals, 
        bvecs=bvecs, 
        n_fibers=n_fibers, 
        ard_weight=ard_weight
    )
    
    # Guide: AutoDelta gives MAP estimate (Point estimate). 
    # Use AutoNormal or AutoLowRankMultivariateNormal for full VI.
    guide = autoguide.AutoDelta(model)
    
    optimizer = numpyro.optim.Adam(step_size=learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # --- SVI Training Loop ---
    # We use jax.lax.scan for the compilation-friendly optimization loop
    def step_fn(svi_state, _):
        svi_state, loss = svi.update(svi_state, data)
        return svi_state, loss

    svi_state = svi.init(key, data)
    
    svi_state, losses = jax.lax.scan(step_fn, svi_state, None, length=num_steps)
    
    # Get posterior samples (or parameters)
    # We draw samples from the guide to estimate the means/stds
    params = svi.get_params(svi_state)
    
    # We can either return the raw params (means/sigmas of the guide) 
    # or draw samples to compute stats.
    # Let's return the median of the guide components.
    median_params = guide.median(params)
    
    return median_params, losses

@partial(jax.jit, static_argnames=['n_fibers', 'num_steps'])
def fit_batch_vi(
    key, 
    data, 
    bvals, 
    bvecs, 
    n_fibers=3, 
    ard_weight=1.0, 
    num_steps=1000
):
    """
    Fit a batch of voxels (N_voxels, N_diff) using vmap.
    """
    # data: (B, N_diff)
    # keys: (B,)
    batch_size = data.shape[0]
    keys = jax.random.split(key, batch_size)
    
    # vmap over the batch dimension (axis 0 of data and keys)
    # bvals/bvecs are constant across voxels
    return jax.vmap(
        partial(
            fit_voxel_vi, 
            bvals=bvals, 
            bvecs=bvecs, 
            n_fibers=n_fibers, 
            ard_weight=ard_weight,
            num_steps=num_steps
        )
    )(keys, data)
