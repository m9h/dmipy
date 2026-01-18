"""
Differentiable Streamline Integrator (Phase B)

This module implements a purely functional, differentiable tractography kernel.
It uses Gumbel-Softmax to sample directions in a differentiable manner and
employs a "soft survival" mechanism to handle termination without breaking
the computational graph.
"""

import jax
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, Bool
from typing import Tuple, NamedTuple
import jax.scipy.ndimage as ndimage

@chex.dataclass
class WalkerState:
    """State of the streamline walker."""
    pos: Float[Array, "batch 3"]
    dir: Float[Array, "batch 3"]
    alive: Float[Array, "batch 1"]


def evaluate_odf(
    sh_coeffs: Float[Array, "X Y Z N_coeffs"],
    pos: Float[Array, "batch 3"],
    sphere_dirs: Float[Array, "N_dirs 3"],
    sh_basis: Float[Array, "N_dirs N_coeffs"] = None # Optional precomputed basis
) -> Tuple[Float[Array, "batch N_dirs"], Float[Array, "batch 1"]]:
    """
    Interpolates SH coefficients at continuous coordinates and computes ODF amplitudes.
    
    Also returns Generalized Fractional Anisotropy (GFA) for the current location.
    
    Args:
        sh_coeffs: 4D tensor of Spherical Harmonic coefficients.
        pos: Continuous coordinates (batch, 3). Note: map_coordinates expects (3, batch).
        sphere_dirs: Directions on the sphere to evaluate the ODF.
        sh_basis: Precomputed SH basis matrix (N_dirs, N_coeffs). 
                  If None, this function assumes it can't compute it (would need l_max). 
                  For now, let's assume the user passes projected ODF amplitudes or we do simple dot product if basis is provided.
                  Actually, usually one projects SH to ODF amplitudes via matrix multiplication.
                  Let's assume sh_basis is provided or we can just do the dot product if we view sh_coeffs as weights for the basis.
                  
                  Wait, standard SH to amplitude is just: sum(c_lm * Y_lm(theta, phi)).
                  So if sh_basis corresponds to Y_lm(sphere_dirs), then amplitudes = coeffs @ basis.T
                  
                  Let's assume sh_basis is passed for efficiency, or we strictly require it.
                  The prompt says: "Interpolate SH coefficients ... Compute amplitudes along sphere_directions."
                  
    Returns:
        amplitudes: ODF amplitudes along sphere_dirs (batch, N_dirs).
        gfa: Generalized Fractional Anisotropy at the interpolated locations (batch, 1).
    """
    
    # map_coordinates expects coordinates in shape (ndim, n_points)
    # Our pos is (batch, 3) corresponding to (x, y, z) indices typically.
    # map_coordinates uses independent variables for dimensions.
    
    # Transpose pos to (3, batch)
    coords = pos.T
    
    # We need to interpolate each coefficient channel.
    # sh_coeffs is (X, Y, Z, N_coeffs). 
    # We can treat N_coeffs as the batch dimension for map_coordinates if we permit it, 
    # but map_coordinates usually interpolates a single channel 3D volume.
    # To handle multiple channels efficiently, we can use vmap over the last dimension of sh_coeffs.
    
    # Expected sh_coeffs shape for map_coordinates: (X, Y, Z)
    # We want to run this for each coefficient.
    
    def interpolate_channel(vol_channel):
        return ndimage.map_coordinates(vol_channel, coords, order=1, mode='nearest')
    
    # sh_coeffs: (X, Y, Z, N_coeffs) -> (N_coeffs, X, Y, Z) for vmapping
    sh_coeffs_transposed = jnp.moveaxis(sh_coeffs, -1, 0)
    
    # interpolated_coeffs: (N_coeffs, batch)
    interpolated_coeffs = jax.vmap(interpolate_channel)(sh_coeffs_transposed)
    
    # Transpose back to (batch, N_coeffs)
    interpolated_coeffs = interpolated_coeffs.T
    
    # Compute amplitudes: (batch, N_coeffs) @ (N_coeffs, N_dirs) -> (batch, N_dirs)
    # Assuming sh_basis is (N_dirs, N_coeffs)
    # If sh_basis is not provided, we can't easily compute amplitudes from general SH coeffs without re-evaluating SH basis.
    # For this implementation, I will assume sh_basis is provided.
    
    amplitudes = jnp.dot(interpolated_coeffs, sh_basis.T)
    
    # Compute GFA
    # GFA = std(ODF) / rms(ODF)
    # sqrt(N * sum((vals - mean)^2)) / sqrt((N-1) * sum(vals^2))
    # Or simpler definition: GFA = sqrt(1 - (mean(ODF)^2 / mean(ODF^2))) ? 
    # Standard definition: GFA = std(ODF) / rms(ODF) = sqrt(N * sum((ODF - mean)^2) / ((N-1) * sum(ODF^2)))
    
    # Let's use the definition: GFA = std(Psi) / rms(Psi)
    # where Psi are the ODF amplitudes.
    
    rms = jnp.sqrt(jnp.mean(amplitudes**2, axis=-1, keepdims=True))
    std = jnp.std(amplitudes, axis=-1, keepdims=True)
    
    # Avoid division by zero
    gfa = jnp.where(rms > 1e-6, std / rms, 0.0)
    
    return amplitudes, gfa

def step_fn(
    state: WalkerState,
    sh_coeffs: Float[Array, "X Y Z N_coeffs"],
    sphere_dirs: Float[Array, "N_dirs 3"],
    sh_basis: Float[Array, "N_dirs N_coeffs"],
    key: jax.random.PRNGKey,
    temperature: float = 0.5,
    step_size: float = 0.5,
    min_gfa: float = 0.2,
    max_angle_deg: float = 45.0
) -> WalkerState:
    """
    Performs one step of the differentiable tracking.
    
    1. Interpolates SH at state.pos.
    2. Computes ODF amplitudes.
    3. Applies valid cone mask (dot product with previous dir).
    4. Samples new direction using Gumbel-Softmax.
    5. Updates position and alive status.
    """
    
    # 1. & 2. Interpolate and Amplitude
    amplitudes, gfa = evaluate_odf(sh_coeffs, state.pos, sphere_dirs, sh_basis)
    
    # 3. Apply Forward Cone Mask
    # We want to penalize directions that deviate too much from state.dir
    # Cosine similarity: dot(dir, sphere_dirs)
    # state.dir: (batch, 3), sphere_dirs: (N_dirs, 3)
    
    # cos_sim: (batch, N_dirs)
    cos_sim = jnp.dot(state.dir, sphere_dirs.T)
    
    # Threshold for max angle
    min_cos = jnp.cos(jnp.deg2rad(max_angle_deg))
    
    # Masking: We want to set logits of invalid directions to -inf (or very large negative number)
    # But for a "soft" walker, maybe a soft penalty?
    # Requirement says: "Apply a 'Forward Cone' mask (penalize turning > 45 degrees)"
    # A hard mask is okay for the logits before softmax.
    
    # large negative number for masking
    MASK_VAL = -1e9
    
    # logits = amplitudes. We assume ODF amplitudes are positive-ish probability mass,
    # but for Softmax we usually work with log-probs or raw scores.
    # ODF values can be negative due to truncation, so usually we clip them or take absolute?
    # Or just treat them as logits directly?
    # Usually ODF is a probability density function on sphere.
    # If we treat ODF amplitude as probability p, then logit = log(p).
    # If we treat ODF amplitude as "energy", we can use it directly?
    # Let's clip negative values to small epsilon and take log for logits, 
    # OR if ODF can be negative, maybe we just use the raw amplitude as the logit?
    # Let's take `logits = amplitudes` for now, assuming they are scaled appropriately.
    # Better yet, let's clip at 0 and normalize?
    # "uses Gumbel-Softmax to compute a weighted 'soft direction' vector"
    # Typically input to softmax are logits. 
    # If amplitudes represents probability density then logits = log(amplitudes).
    # Let's use log(ReLU(amplitudes) + eps).
    
    safe_amplitudes = jnp.maximum(amplitudes, 1e-6)
    logits = jnp.log(safe_amplitudes)
    
    # Apply cone mask
    # If cos_sim < min_cos, mask it out.
    logits = jnp.where(cos_sim >= min_cos, logits, MASK_VAL)
    
    # 4. Gumbel-Softmax
    # weights = softmax((logits + gumbel_noise) / temp)
    
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    weights = jax.nn.softmax((logits + gumbel_noise) / temperature, axis=-1)
    
    # Compute new direction as weighted sum of sphere directions
    # weights: (batch, N_dirs)
    # sphere_dirs: (N_dirs, 3)
    # new_dir: (batch, 3)
    new_dir = jnp.dot(weights, sphere_dirs)
    
    # Normalize new_dir to ensure unit length (soft average might shorten it)
    new_dir = new_dir / (jnp.linalg.norm(new_dir, axis=-1, keepdims=True) + 1e-6)
    
    # 5. Update Position
    # new_pos = pos + step_size * new_dir
    # IMPORTANT: If the walker is dead, it shouldn't move? 
    # "The streamline continues computing but contributes zero weight once 'dead.'"
    # The requirement says "Use a continuous alive probability".
    # So we update position regardless? Or scaling step by alive?
    # Usually we update position fully, but the weight of this segment is modulated by alive.
    # But if we update position fully for dead particles, they might wander into garbage areas.
    # Let's update position.
    
    new_pos = state.pos + step_size * new_dir
    
    # 6. Update Alive Status
    # "Use a complex alive probability ... decay based on local anisotropy (GFA)"
    # Sigmoid on GFA: alive_new = alive_old * sigmoid((GFA - threshold) * sharpness)
    # This keeps it between 0 and 1 and differentiable.
    
    # gfa: (batch, 1)
    # soft step: sigmoid(k * (x - x0))
    # Let's make it fairly sharp.
    # If GFA < min_gfa, prob drops.
    
    survival_prob = jax.nn.sigmoid(20.0 * (gfa - min_gfa))
    
    # Also kill if we couldn't find ANY valid direction (all logits masked)?
    # Softmax handles MASK_VAL by giving 0 weight, but if all are MASK_VAL, it gives uniform?
    # If max(logits) is MASK_VAL, then we are stuck.
    # Check if max(cos_sim) < min_cos?
    
    valid_cone_prob = jnp.where(jnp.max(cos_sim, axis=-1, keepdims=True) >= min_cos, 1.0, 0.0)
    # Soft version of cone validity? 
    # Maybe just rely on GFA drop if we turn around? Usually GFA doesn't drop just because we turn.
    # But strictly, if we hit the mask, we should stop.
    # Let's multiply survival_prob by valid_cone_prob.
    
    new_alive = state.alive * survival_prob * valid_cone_prob
    
    return WalkerState(pos=new_pos, dir=new_dir, alive=new_alive)


def track(
    sh_coeffs: Float[Array, "X Y Z N_coeffs"],
    seeds: Float[Array, "batch 3"],
    seed_dirs: Float[Array, "batch 3"],
    sphere_dirs: Float[Array, "N_dirs 3"],
    sh_basis: Float[Array, "N_dirs N_coeffs"],
    key: jax.random.PRNGKey,
    step_size: float = 0.5,
    max_steps: int = 100,
    temperature: float = 0.1,
    min_gfa: float = 0.2
) -> Float[Array, "steps batch 3"]:
    """
    Main entry point for differentiable tractography.
    Uses jax.lax.scan to unroll integration steps.
    
    Args:
        sh_coeffs: SH coefficients volume.
        seeds: Starting positions.
        seed_dirs: Initial tracking directions.
        sphere_dirs: Sphere discretization directions.
        sh_basis: Basis for specific sphere_dirs.
        key: RNG key.
    
    Returns:
        History of positions (max_steps + 1, batch, 3).
    """
    
    # Initialize state
    initial_state = WalkerState(
        pos=seeds,
        dir=seed_dirs / (jnp.linalg.norm(seed_dirs, axis=-1, keepdims=True) + 1e-6),
        alive=jnp.ones((seeds.shape[0], 1))
    )
    
    keys = jax.random.split(key, max_steps)
    
    def scan_fn(carry_state, step_key):
        new_state = step_fn(
            carry_state, 
            sh_coeffs, 
            sphere_dirs, 
            sh_basis, 
            step_key, 
            temperature=temperature,
            step_size=step_size,
            min_gfa=min_gfa
        )
        # We record the new position
        # Optionally we can record the whole state if needed, but requested pos history.
        return new_state, new_state.pos

    # Run scan
    final_state, pos_history = jax.lax.scan(scan_fn, initial_state, keys)
    
    # Prepend initialization?
    # pos_history corresponds to steps 1 to N.
    # User might want step 0.
    
    full_history = jnp.concatenate([initial_state.pos[None, ...], pos_history], axis=0)
    
    return full_history
