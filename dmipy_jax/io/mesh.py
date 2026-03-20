"""
Standardized Mesh IO Module.
Supports loading surface meshes for PINN training.
Specialized support for FSL BET outputs (.off, .vtk).
"""

import os
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Union

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

def load_mesh(path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Loads a surface mesh from file.
    
    Args:
        path: Absolute path to mesh file.
        
    Returns:
        vertices: (N, 3) float array.
        faces: (M, 3) int array (zero-indexed).
        
    Raises:
        ValueError: If format not supported or file parsing fails.
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.off':
        return _read_off(path)
    elif ext == '.vtk':
        return _read_vtk_legacy(path)
    elif TRIMESH_AVAILABLE:
        # Fallback to trimesh for .obj, .stl, etc.
        try:
            mesh = trimesh.load(path, process=False)
            # Ensure it's a Trimesh object
            if isinstance(mesh, trimesh.Scene):
                # If scene, try to get first geometry or concatenate
                if len(mesh.geometry) == 0:
                    raise ValueError("Empty scene")
                # Grab first
                key = list(mesh.geometry.keys())[0]
                mesh = mesh.geometry[key]
                
            return jnp.array(mesh.vertices), jnp.array(mesh.faces)
        except Exception as e:
            raise ValueError(f"Trimesh failed to load {path}: {e}")
    else:
        raise ValueError(f"Unsupported format {ext} (and trimesh not installed).")

def _read_off(path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reads Geomview OFF format (Standard FSL BET output).
    Format:
    OFF
    N_verts N_faces N_edges
    x y z
    ...
    N_verts_in_face v1 v2 v3 ...
    ...
    """
    with open(path, 'r') as f:
        # Read header
        header = f.readline().strip()
        if header != 'OFF':
            # Sometimes parsing involves skipping comments
            # Or OFF is on same line as counts
            pass
            
        # Read counts
        # Skip empty lines
        line = f.readline()
        while not line.strip() or line.startswith('#'):
            line = f.readline()
            
        counts = line.strip().split()
        n_verts = int(counts[0])
        n_faces = int(counts[1])
        
        # Read Vertices
        verts = []
        for _ in range(n_verts):
            line = f.readline()
            verts.append([float(x) for x in line.split()])
            
        # Read Faces
        faces = []
        for _ in range(n_faces):
            line = f.readline()
            # FSL output: N v1 v2 v3 (usually 3 for triangles)
            # Color might follow
            parts = [int(x) for x in line.split()]
            n_v = parts[0]
            if n_v != 3:
                # Triangulation needed if not triangles?
                # For now assume triangles as FSL BET produces triangles usually
                pass
            faces.append(parts[1:4])
            
    return jnp.array(verts), jnp.array(faces)

def _read_vtk_legacy(path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reads legacy ASCII VTK format (FSL BET -e mesh option).
    Only supports POLYDATA with POINTS and POLYGONS.
    """
    vertices = []
    faces = []
    
    with open(path, 'r') as f:
        # Simple state machine
        state = 'header'
        n_points = 0
        n_polys = 0
        
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line: 
                i+=1
                continue
                
            if 'POINTS' in line:
                # POINTS N type
                parts = line.split()
                n_points = int(parts[1])
                # Read N lines
                # Sometimes points are space separated on multiple lines
                # We assume standard formatting 3 per line or flat?
                # FSL VTK usually x y z per line
                v_count = 0
                i += 1
                while v_count < n_points and i < len(lines):
                    vals = [float(x) for x in lines[i].split()]
                    # Can have multiple points per line?
                    # VTK legacy allows 3n floats.
                    
                    # Flatten logic
                    for k in range(0, len(vals), 3):
                        if v_count < n_points:
                            vertices.append(vals[k:k+3])
                            v_count += 1
                    i += 1
                continue
                
            if 'POLYGONS' in line:
                # POLYGONS N size
                parts = line.split()
                n_polys = int(parts[1])
                p_count = 0
                i += 1
                while p_count < n_polys and i < len(lines):
                    # N v1 v2 v3 ...
                    vals = [int(x) for x in lines[i].split()]
                    n_v = vals[0]
                    faces.append(vals[1:1+n_v]) # Keep as list, might be mixed
                    p_count += 1
                    i += 1
                continue
                
            i += 1
            
    # Assuming triangles
    faces_array = np.array(faces)
    if faces_array.shape[1] != 3:
        # Just warn or take first 3 if needed
        pass
        
    return jnp.array(vertices), jnp.array(faces_array)
