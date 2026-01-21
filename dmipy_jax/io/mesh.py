import jax.numpy as jnp
from typing import Tuple, Optional
import os

def load_obj(filename: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load a wavefront OBJ file.
    
    Args:
        filename: Path to the OBJ file.
        
    Returns:
        vertices: (N, 3) array of floats.
        faces: (M, 3) array of integers (indices).
    """
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # Face
                parts = line.strip().split()
                # Handle obj 1-based indexing and possible texture/normal indices (v/vt/vn)
                face_indices = []
                for p in parts[1:]:
                    # Take the first part of "v/vt/vn"
                    v_idx = int(p.split('/')[0]) - 1
                    face_indices.append(v_idx)
                faces.append(face_indices)
                
    return jnp.array(vertices), jnp.array(faces, dtype=jnp.int32)

def save_obj(filename: str, vertices: jnp.ndarray, faces: jnp.ndarray):
    """
    Save a wavefront OBJ file.
    
    Args:
        filename: Path to the OBJ file.
        vertices: (N, 3) array of floats.
        faces: (M, 3) array of integers.
    """
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # 1-based indexing
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def create_sphere_mesh(radius: float = 1.0, level: int = 2) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a geodesic sphere mesh (approximation).
    Starting from an icosahedron and subdividing.
    
    For now, a simple placeholder returning an icosahedron.
    """
    # 1. Icosahedron vertices (unit sphere)
    phi = (1.0 + jnp.sqrt(5.0)) / 2.0
    verts = jnp.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ])
    verts = verts / jnp.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = jnp.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=jnp.int32)
    
    # Subdivision
    for _ in range(level):
        # Current edges
        # We need a way to map edges to new vertices to avoid duplicates.
        # Ideally using an indexed structure.
        # But for JAX, dynamic shapes are hard.
        # Since this is "io/mesh", we can use numpy for construction!
        pass
        
    return verts * radius, faces

def subdivide_mesh_numpy(verts, faces):
    """
    Subdivide triangular mesh using midpoint splitting (1 -> 4 triangles).
    Uses numpy.
    """
    import numpy as np
    
    # Edges
    edges = {}
    
    new_faces = []
    
    # Helper to get/create midpoint
    midpoints = {} # (v1, v2) -> index
    
    new_verts = list(verts)
    
    def get_midpoint(v1, v2):
        key = tuple(sorted((v1, v2)))
        if key in midpoints:
            return midpoints[key]
        
        # Create new
        p1 = new_verts[v1]
        p2 = new_verts[v2]
        pm = (np.array(p1) + np.array(p2)) / 2.0
        pm = pm / np.linalg.norm(pm) # Project to unit sphere
        
        idx = len(new_verts)
        new_verts.append(pm)
        midpoints[key] = idx
        return idx
        
    for tri in faces:
        v0, v1, v2 = tri
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)
        
        # 4 new faces
        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])
        
    return np.array(new_verts), np.array(new_faces)

# Redefine create_sphere_mesh to use numpy subdivision
def create_sphere_mesh(radius: float = 1.0, level: int = 2) -> Tuple[jnp.ndarray, jnp.ndarray]:
    import numpy as np
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    for _ in range(level):
        verts, faces = subdivide_mesh_numpy(verts, faces)
        
    return jnp.array(verts * radius), jnp.array(faces)
