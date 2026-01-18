
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from typing import Tuple, Optional, Callable, Union
from jaxtyping import Array, Float, Int

# Constants
FARADAY_CONSTANT = 96485.33212  # C/mol
GAS_CONSTANT = 8.314462618  # J/(mol*K)


def create_electrode_masks(
    positions: Float[Array, "n_electrodes 3"],
    shape: Tuple[int, int, int],
    affine: Optional[Float[Array, "4 4"]] = None,
    result_type: str = 'float'
) -> Union[Float[Array, "n_electrodes nx ny nz"], Int[Array, "n_electrodes nx ny nz"]]:
    """
    Rasterizes point electrode positions into voxel masks.
    
    Args:
        positions: Physical coordinates (x, y, z) of electrodes.
        shape: Shape of the target volume (nx, ny, nz).
        affine: Affine transformation matrix (4x4) mapping voxels to physical coordinates.
                If None, assumes identity (positions are already in voxel coordinates).
        result_type: 'float' or 'int' for the output mask.
        
    Returns:
        Stack of masks, one per electrode. 1.0 at the nearest voxel, 0.0 elsewhere.
    """
    n_electrodes = positions.shape[0]
    nx, ny, nz = shape
    
    if affine is None:
        # Positions are voxel indices
        indices = positions
    else:
        # Map physical to voxel: inv(affine) * [x, y, z, 1]
        # inv_affine @ [x,y,z,1].T
        inv_affine = jnp.linalg.inv(affine)
        
        # Homogeneous coordinates
        ones = jnp.ones((n_electrodes, 1))
        pos_h = jnp.hstack([positions, ones])
        
        # Transform
        indices_h = jnp.dot(inv_affine, pos_h.T).T # (N, 4)
        indices = indices_h[:, :3]
        
    # Round to nearest integer
    indices = jnp.round(indices).astype(int)
    
    # Clip to bounds
    indices = jnp.clip(indices, 0, jnp.array([nx-1, ny-1, nz-1]))
    
    # Create masks
    masks = jnp.zeros((n_electrodes, *shape))
    
    # Can use jax.ops.index_update or .at
    # Iterate or vmap? For small N electrodes, loop is fine.
    # Actually, we can use scatter.
    
    for i in range(n_electrodes):
        idx = tuple(indices[i])
        masks = masks.at[(i,) + idx].set(1.0)
        
    if result_type == 'int':
        return masks.astype(int)
    return masks

def nernst_einstein_conductivity(
    diffusivity: Float[Array, "... 3 3"], 
    concentration: Float[Array, "..."], 
    temperature: float = 310.15,
    charge_number: int = 1
) -> Float[Array, "... 3 3"]:
    """
    Converts diffusion tensor to electrical conductivity tensor using the Nernst-Einstein relation.
    
    sigma = (z^2 * F^2 * C * D) / (R * T)
    
    Args:
        diffusivity: Diffusion tensor in m^2/s. Shape (..., 3, 3).
        concentration: Ion concentration in mol/m^3. Shape (...).
        temperature: Temperature in Kelvin. Default is 310.15 K (37 C).
        charge_number: Valency of the ion (z). Default is 1 (e.g., Na+, K+).
        
    Returns:
        Conductivity tensor in S/m. Shape (..., 3, 3).
    """
    
    prefactor = (charge_number**2 * FARADAY_CONSTANT**2) / (GAS_CONSTANT * temperature)
    
    # Ensure concentration broadcasts correctly against diffusivity
    # If concentration is (N,), diffusivity is (N, 3, 3), we need (N, 1, 1) or automatic broadcasting
    # concentration * diffusivity works if shapes are compatible.
    
    # Expand dims of concentration to match (..., 3, 3) if needed by multiplying scalar-wise
    # But JAX broadcasting usually handles scalar * tensor
    
    # We assume concentration has shape (...) matching the batch dims of D.
    # To multiply properly, we convert C to shape (..., 1, 1)
    
    if concentration.ndim == diffusivity.ndim - 2:
        c_expanded = concentration[..., None, None]
    else:
        c_expanded = concentration
        
    sigma = prefactor * c_expanded * diffusivity
    return sigma

def _apply_variable_coefficient_laplacian(
    potential: Float[Array, "nx ny nz"],
    sigma: Float[Array, "nx ny nz 3 3"],
    voxel_size: float
) -> Float[Array, "nx ny nz"]:
    """
    Computes Div(sigma * Grad(potential)) using a compact Finite Volume scheme on value-centered grid.
    Neumann boundary conditions (zero flux) are implicitly enforced at the boundaries of the domain.
    """
    h = voxel_size
    # 1. Compute central gradients at centers for cross-terms
    # Shape (nx, ny, nz, 3)
    grad_V_center = jnp.stack(jnp.gradient(potential, h), axis=-1)
    
    div_J = jnp.zeros_like(potential)
    
    # Iterate over axes x, y, z (0, 1, 2)
    for axis in range(3):
        # We compute Flux at interior faces.
        # If dimension is N, we have N-1 interior faces.
        # Face k is between voxel k and k+1.
        
        # Create slices
        sl_i = [slice(None)] * 3
        sl_ip1 = [slice(None)] * 3
        sl_i[axis] = slice(0, -1)
        sl_ip1[axis] = slice(1, None)
        
        # V_i, V_ip1 at centers adjacent to face
        V_i = potential[tuple(sl_i)]
        V_ip1 = potential[tuple(sl_ip1)]
        
        # Compact Normal Derivative at face
        grad_V_normal = (V_ip1 - V_i) / h
        
        # Average sigma at face
        S_i = sigma[tuple(sl_i)]
        S_ip1 = sigma[tuple(sl_ip1)]
        S_face = 0.5 * (S_i + S_ip1)
        
        # Estimate full Gradient at face
        # Normal component: from compact diff
        # Tangential components: from average of central diffs
        G_center_i = grad_V_center[tuple(sl_i)]
        G_center_ip1 = grad_V_center[tuple(sl_ip1)]
        G_face = 0.5 * (G_center_i + G_center_ip1)
        
        # Overwrite normal component with compact estimate
        G_face = G_face.at[..., axis].set(grad_V_normal)
        
        # Compute Current Density J = -sigma * Grad V at face
        J_face = -jnp.einsum('...ij,...j->...i', S_face, G_face)
        
        # Extract component normal to face (flux)
        J_flux = J_face[..., axis]
        
        # Update Divergence
        # Cell i (left of face) loses flux: -J_flux/h (Note: J is vector, flux out is J.n)
        # Vector points +axis direction. 
        # Out of i (right face): +J_flux
        # Into i+1 (left face): +J_flux
        # Div = (Flux_out - Flux_in)/h
        # Check sign: Div(J) at i.
        # Flux Right (i+1/2) corresponds to J_flux. 
        # Contribution to Div[i]: + J_flux / h  (Wait, J is vector field. Div J = dJx/dx)
        # dJx/dx at i ~ (Jx(i+1/2) - Jx(i-1/2))/h
        # So J_flux(i+1/2) adds positive to i? NO. J(i+1/2) is at Right boundary.
        # It is the "Forward" term.
        # J(i-1/2) is the "Backward" term.
        # So Div[i] += J_flux[i] / h (if J_flux[i] is defined as face i+1/2? No no.)
        
        # Let's align indices:
        # J_flux calculated above corresponds to face between i (slice 0:-1) and i+1 (slice 1:None).
        # Let's call this face index k (0 to N-2). Face k is at i=k+0.5.
        # It is the RIGHT face of voxel k.
        # It is the LEFT face of voxel k+1.
        
        # Contribution to Voxel k (Left of face):
        # It is the Right Face. Term in (Right - Left).
        # So it contributes +1/h to Divergence of Voxel k.
        
        # Contribution to Voxel k+1 (Right of face):
        # It is the Left Face. Term in (Right - Left).
        # So it contributes -1/h to Divergence of Voxel k+1.
        
        # Map back to full grid
        # sl_i corresponds to Voxel k
        div_J = div_J.at[tuple(sl_i)].add(J_flux / h)
        # sl_ip1 corresponds to Voxel k+1
        div_J = div_J.at[tuple(sl_ip1)].add(-J_flux / h)
        
    return div_J 

def solve_voltage_field(
    sigma: Float[Array, "nx ny nz 3 3"], 
    source_map: Float[Array, "nx ny nz"], 
    voxel_size: float = 0.001,
    maxiter: int = 1000,
    tol: float = 1e-5
) -> Float[Array, "nx ny nz"]:
    """
    Solves the Poisson equation: Div(sigma * Grad(V)) = -I_source
    
    Args:
        sigma: Conductivity tensor field.
        source_map: Source current density scalar field (entering current positive).
                    Units: A/m^3.
        voxel_size: Voxel side length in meters.
        
    Returns:
        Voltage field V.
    """
    shape = source_map.shape
    size = source_map.size
    
    # Operator A(x) -> b
    # A(V) = Div(sigma * Grad(V))
    # b = -I_source
    
    # Flatten inputs for CG
    b = -source_map.ravel()
    
    def matvec_flattener(v_flat):
        v = v_flat.reshape(shape)
        # Apply operator
        res = _apply_variable_coefficient_laplacian(v, sigma, voxel_size)
        return res.ravel()
        
    # Initial guess: zeros
    x0 = jnp.zeros(size)
    
    # Preconditioning? 
    # For now, no preconditioner. 
    
    # Jax CG
    # Note: A is symmetric negative semi-definite? 
    # Div(sigma Grad) is negative semi-definite. CG works for SPD (Positive Definite).
    # We solve -Div(sigma Grad V) = I_source.
    # Then operator is Positive Semi-Definite (Neumann allows arbitrary const bias).
    # We should fix one node or ensure sum(source) == 0.
    
    def positive_operator(v_flat):
        return matvec_flattener(v_flat)
        
    b_pos = -b # Because we multiplied LHS by -1
    
    # Run CG
    # Use jax.scipy.sparse.linalg.cg
    v_flat, _ = cg(positive_operator, b_pos, x0=x0, maxiter=maxiter, tol=tol)
    
    # Center potential (remove constant DC offset drift)
    v_flat = v_flat - jnp.mean(v_flat)
    
    return v_flat.reshape(shape)

def tdcs_objective_function(
    injected_current_pattern: Float[Array, "n_electrodes"],
    electrode_masks: Float[Array, "n_electrodes nx ny nz"],
    target_roi_mask: Float[Array, "nx ny nz"],
    target_direction: Float[Array, "3"], 
    sigma: Float[Array, "nx ny nz 3 3"],
    voxel_size: float = 0.001,
    lambda_reg: float = 0.01
) -> float:
    """
    Objective function to optimize injected currents for steering current to a target.
    
    Loss = || (J_local . target_direction) - J_desired ||^2 + lambda ||I_injected||^2
    Or maximizing current in direction: - sum(J . target_direction) in ROI + reg
    
    Here implementing a "Maximization" form as a minimization of negative component.
    
    Args:
        injected_current_pattern: Current values at each electrode.
        electrode_masks: Boolean/Float masks indicating voxel locations of electrodes.
        target_roi_mask: Mask for the region of interest.
        target_direction: Desired direction of current flow (normalized vector).
        sigma: Conductivity map.
        
    Returns:
        Scalar loss value.
    """
    
    # 1. Construct Source Map from electrodes
    # source_map = sum(I_k * Mask_k)
    # Normalize masks so they integrate to 1? Or integral is Volume.
    # Assuming masks are 1 inside electrode, 0 outside.
    # Current density source I_source term: A/m^3.
    # Total current I_k = integral (source) dV = source_val * V_electrode
    # So source_val = I_k / (count(Mask_k) * voxel_size^3)
    
    voxel_vol = voxel_size**3
    source_map = jnp.zeros_like(electrode_masks[0])
    
    for i in range(len(injected_current_pattern)):
        mask = electrode_masks[i]
        vol_electrode = jnp.sum(mask) * voxel_vol + 1e-12
        density = injected_current_pattern[i] / vol_electrode
        source_map = source_map + mask * density
        
    # 2. Solve Forward Model
    voltage = solve_voltage_field(sigma, source_map, voxel_size)
    
    # 3. Compute Current Density J = -sigma Grad V
    grad_V = jnp.stack(jnp.gradient(voltage, voxel_size), axis=-1)
    J = -jnp.einsum('...ij,...j->...i', sigma, grad_V)
    
    # 4. Compute Loss
    # We want to maximize J in target_direction inside ROI.
    # Loss = - sum( dot(J, target_direction) ) in ROI
    
    J_proj = jnp.dot(J, target_direction) # (nx, ny, nz)
    
    # Sum over ROI
    roi_current_sum = jnp.sum(J_proj * target_roi_mask)
    
    # Regularization (minimize total power/current)
    reg_term = lambda_reg * jnp.sum(injected_current_pattern**2)
    
    return -roi_current_sum + reg_term

