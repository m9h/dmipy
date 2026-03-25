"""
Mesh Operations for PINNs in JAX.
Provides differentiable (or JIT-compatible) surface sampling and geometry checks.
"""

import jax
import jax.numpy as jnp
from functools import partial
from dmipy_jax.pinns.sampling import DynamicCollocationSampler

def compute_face_areas(vertices, faces):
    """
    Computes area of each triangle face.
    Area = 0.5 * || (B-A) x (C-A) ||
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    cross = jnp.cross(v1 - v0, v2 - v0)
    area = 0.5 * jnp.linalg.norm(cross, axis=1)
    return area

@partial(jax.jit, static_argnums=(3,))
def sample_surface_uniform(key, vertices, faces, n_points):
    """
    Samples n_points uniformly from the mesh surface.
    1. Select face with probability proportional to area.
    2. Sample point in triangle (Random Barycentric coords).
    """
    areas = compute_face_areas(vertices, faces)
    total_area = jnp.sum(areas)
    probs = areas / total_area
    
    k1, k2 = jax.random.split(key)
    
    # 1. Select Faces
    face_indices = jax.random.choice(k1, len(faces), shape=(n_points,), p=probs)
    
    # 2. Sample in Triangle (P = (1-sqrt(r1))A + (sqrt(r1)(1-r2))B + (sqrt(r1)r2)C)
    # Standard method: r1, r2 uniform [0,1].
    # if r1+r2 > 1, reflect.
    # Barycentric: u, v, w. u+v+w=1.
    
    # Different formulation:
    # A + r1*(B-A) + r2*(C-A). if r1+r2 > 1, reflect.
    
    r = jax.random.uniform(k2, shape=(n_points, 2))
    r1 = r[:, 0:1] # (n,1)
    r2 = r[:, 1:2]
    
    # Reflect if outside
    mask = (r1 + r2) > 1.0
    r1 = jnp.where(mask, 1.0 - r1, r1)
    r2 = jnp.where(mask, 1.0 - r2, r2)
    # Actually reflection logic is slightly more complex, 
    # simpler: sort?
    # Correct method: u = 1 - sqrt(r1), v = sqrt(r1) * (1-r2), w = sqrt(r1) * r2
    
    u = 1.0 - jnp.sqrt(r1)
    v = jnp.sqrt(r1) * (1.0 - r2) # Note: r2 here should be separate random var?
    # The formulation depends on distribution.
    # Kraemer approach: 
    # u = 1 - sqrt(r1)
    # v = r2 * sqrt(r1)
    # w = 1 - u - v
    
    # Let's use the robust one:
    # r1, r2 uniform.
    # if r1+r2 > 1: r1=1-r1, r2=1-r2
    # P = A + r1(B-A) + r2(C-A)
    
    r1_raw = jax.random.uniform(k2, shape=(n_points, 1))
    # need new key? split k2? No, just slice
    # Actually just generate (n,2)
    r = jax.random.uniform(k2, shape=(n_points, 2))
    r1 = r[:, 0:1]
    r2 = r[:, 1:2]
    
    mask = (r1 + r2) > 1.0
    # Reflection: (1-r1, 1-r2) maps triangle (1,0)-(0,1)-(1,1) back to (0,0)-(1,0)-(0,1)
    r1_final = jnp.where(mask, 1.0 - r1, r1)
    r2_final = jnp.where(mask, 1.0 - r2, r2)
    
    selected_faces = faces[face_indices]
    A = vertices[selected_faces[:, 0]]
    B = vertices[selected_faces[:, 1]]
    C = vertices[selected_faces[:, 2]]
    
    P_sampled = A + r1_final * (B - A) + r2_final * (C - A)
    return P_sampled

def get_mesh_sampler(vertices, faces, bounds=((-1,-1,-1), (1,1,1))):
    """
    Returns a DynamicCollocationSampler configured with a mesh interface sampler.
    """
    
    def interface_sampler_fn(key, n):
        return sample_surface_uniform(key, vertices, faces, n)
        
    return DynamicCollocationSampler(
        domain_bounds=bounds,
        interface_fn=interface_sampler_fn,
        interface_fraction=0.5 # Bias towards mesh surface
    )
