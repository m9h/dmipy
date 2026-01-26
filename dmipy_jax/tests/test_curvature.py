import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.geometry import (
    compute_shape_index,
    discrete_mean_curvature_cotangent,
    discrete_gaussian_curvature_angle_deficit,
    compute_face_areas,
    compute_vertex_areas
)

def create_sphere_mesh(radius=1.0, subdivisions=3):
    """
    Create an icosphere mesh.
    """
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    vertices = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ])
    
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None] * radius
    
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])
    
    for _ in range(subdivisions):
        new_faces = []
        new_vertices = vertices.tolist()
        midpoint_cache = {}
        
        def get_midpoint(v1_idx, v2_idx):
            key = tuple(sorted((v1_idx, v2_idx)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            
            p1 = vertices[v1_idx]
            p2 = vertices[v2_idx]
            midpoint = (p1 + p2) / 2.0
            midpoint = midpoint / np.linalg.norm(midpoint) * radius
            
            new_vertices.append(midpoint)
            idx = len(new_vertices) - 1
            midpoint_cache[key] = idx
            return idx
            
        for tri in faces:
            v0, v1, v2 = tri
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            
            new_faces.append([v0, a, c])
            new_faces.append([v1, b, a])
            new_faces.append([v2, c, b])
            new_faces.append([a, b, c])
            
        vertices = np.array(new_vertices)
        faces = np.array(new_faces)
        
    return jnp.array(vertices), jnp.array(faces)

def test_sphere_curvature():
    radius = 2.0
    vertices, faces = create_sphere_mesh(radius=radius, subdivisions=4)
    
    # Expected values
    # Mean curvature H = 1/R = 0.5
    # Gaussian curvature K = 1/R^2 = 0.25
    # Shape index SI = 1.0 (Cap)
    
    H, _ = discrete_mean_curvature_cotangent(vertices, faces)
    K = discrete_gaussian_curvature_angle_deficit(vertices, faces)
    SI = compute_shape_index(vertices, faces)
    
    print(f"Mean Curvature (Min/Max/Mean): {jnp.min(H):.4f}, {jnp.max(H):.4f}, {jnp.mean(H):.4f}")
    print(f"Gaussian Curvature (Min/Max/Mean): {jnp.min(K):.4f}, {jnp.max(K):.4f}, {jnp.mean(K):.4f}")
    print(f"Shape Index (Min/Max/Mean): {jnp.min(SI):.4f}, {jnp.max(SI):.4f}, {jnp.mean(SI):.4f}")
    
    # Allow some tolerance for discrete approximation
    assert jnp.allclose(jnp.mean(H), 1.0/radius, rtol=0.05)
    assert jnp.allclose(jnp.mean(K), 1.0/(radius**2), rtol=0.05)
    
    # Shape index for a convex sphere (Cap) is 1.0
    # Values might fluctuate due to discretization
    assert jnp.allclose(jnp.mean(SI), 1.0, atol=0.1)

def test_differentiability():
    radius = 1.0
    vertices, faces = create_sphere_mesh(radius=radius, subdivisions=2)
    
    def loss_fn(v):
        si = compute_shape_index(v, faces)
        return jnp.sum(si**2)
        
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(vertices)
    
    assert jnp.all(jnp.isfinite(grads))
    assert grads.shape == vertices.shape

if __name__ == "__main__":
    test_sphere_curvature()
    test_differentiability()
    print("All tests passed!")
