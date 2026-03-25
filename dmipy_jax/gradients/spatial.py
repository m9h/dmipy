
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from typing import Optional, Union, Tuple, Callable, Any

def compute_spatial_gradient(
    image_volume: Float[Array, "X Y Z *N"],
    voxel_size: Optional[Union[float, Tuple[float, float, float]]] = None
) -> Float[Array, "3 X Y Z *N"]:
    """
    Computes standard central difference gradient of the image volume.
    
    Args:
        image_volume: 3D (X,Y,Z) or 4D (X,Y,Z,N) array.
        voxel_size: Size of voxels. If None, assumes 1.0. 
                    Can be scalar or (dx, dy, dz).
    
    Returns:
        Gradient array where first dimension is direction (dx, dy, dz).
        Shape: (3, X, Y, Z, [N])
    """
    if voxel_size is None:
        voxel_size = (1.0, 1.0, 1.0)
    elif isinstance(voxel_size, (int, float)):
        voxel_size = (float(voxel_size),) * 3
    
    # We only compute gradients along the first 3 dimensions (spatial)
    # jnp.gradient returns [g_dim0, g_dim1, g_dim2, ...]
    grads = jnp.gradient(image_volume, *voxel_size, axis=(0, 1, 2))
    
    return jnp.stack(grads, axis=0)


def predict_spatial_gradient(
    forward_model_func: Callable[[Any, Any], Float[Array, "N"]],
    params: Any,
    param_gradients: Any,
    acquisition_scheme: Any
) -> Float[Array, "3 N"]:
    """
    Predicts spatial gradient of the diffusion signal using the chain rule:
    dS/dr = (dS/dTheta) * (dTheta/dr).
    
    Allows derivation of signal gradients from underlying parameter map gradients,
    as proposed in Aganj et al. (2026).
    
    This function operates on a SINGLE VOXEL. To apply to a volume, use vmap.
    
    Args:
        forward_model_func: Function f(params, acquisition) -> signal (N,)
        params: PyTree of parameters at a specific voxel.
        param_gradients: PyTree of spatial gradients of parameters. 
                         Structure must match params, but each leaf has an extra 
                         leading dimension of size 3 (dx, dy, dz).
        acquisition_scheme: The acquisition scheme object.
        
    Returns:
        Analytic gradient vector for the signal at this voxel.
        Shape: (3, N)
    """
    
    # 1. Compute Jacobian of Signal w.r.t Parameters: dS/dTheta (Shape: N x n_params)
    # We use eqx.filter_jacfwd to handle the PyTree structure of params.
    jac_S_Theta = eqx.filter_jacfwd(lambda p: forward_model_func(p, acquisition_scheme))(params)
    
    # 2. Compute Dot Product: (dS/dTheta) * (dTheta/dr)
    
    def leaf_dot_product(jac_leaf, grad_leaf):
        # jac_leaf: (N_measurements, *param_shape)
        # grad_leaf: (3, *param_shape)
        # We want output: (3, N_measurements)
        
        param_ndim = grad_leaf.ndim - 1 # first dim is spatial (3)
        
        # dimensions to contract: From 1 to param_ndim in grad, and 1 to param_ndim in jac
        # (effectively dotting all parameter dimensions)
        contract_dims = tuple(range(1, 1 + param_ndim))
        
        # Contract grad[1:] with jac[1:]
        # tensordot(grad, jac, axes=(contract_dims, contract_dims))
        # grad is (3, p...), jac is (N, p...)
        # Result will be (3, N)
        return jnp.tensordot(grad_leaf, jac_leaf, axes=(contract_dims, contract_dims))

    # Calculate contribution from each parameter
    contributions = jax.tree_util.tree_map(leaf_dot_product, jac_S_Theta, param_gradients)
    
    # Sum up all contributions
    leaves = jax.tree_util.tree_leaves(contributions)
    total_gradient = sum(leaves)
    
    return total_gradient
