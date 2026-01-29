"""
Mesh Input/Output Utilities for JAX/Jinns

This module provides utilities to load mesh files (using meshio) and convert them
into JAX-compatible arrays (vertices, faces, cells) suitable for use with
PINN libraries like `jinns` or FEM solvers.
"""

import jax.numpy as jnp
import numpy as np
import meshio
import os
from pathlib import Path

def load_mesh_to_jax(file_path, mesh_format=None):
    """
    Load a mesh file and return JAX arrays for vertices and faces.
    
    Args:
        file_path (str or Path): Path to the mesh file.
        mesh_format (str, optional): Format of the mesh file (e.g., 'stl', 'vtk'). 
                                     If None, inferred from extension.
                                     
    Returns:
        dict: A dictionary containing:
            - 'points': jnp.ndarray of shape (N_points, 3)
            - 'cells': dict of {cell_type: jnp.ndarray of shape (N_cells, vertices_per_cell)}
    """
    file_path = str(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    try:
        mesh = meshio.read(file_path, file_format=mesh_format)
    except Exception as e:
        raise ValueError(f"Failed to read mesh file {file_path}: {e}")

    # Convert points to JAX array
    points = jnp.array(mesh.points, dtype=jnp.float32)
    
    # Process cells
    # mesh.cells is often a list of CellBlock objects in recent meshio versions
    # We want to consolidate them if possible or return a structured dict
    
    cells_dict = {}
    
    # Check if modern meshio (list of CellBlocks)
    if isinstance(mesh.cells, list):
        for cell_block in mesh.cells:
            cell_type = cell_block.type
            data = jnp.array(cell_block.data, dtype=jnp.int32)
            
            if cell_type in cells_dict:
                # Concatenate if multiple blocks of same type
                cells_dict[cell_type] = jnp.concatenate([cells_dict[cell_type], data], axis=0)
            else:
                cells_dict[cell_type] = data
    elif isinstance(mesh.cells, dict):
        # Older meshio or condensed format
        for cell_type, data in mesh.cells.items():
            cells_dict[cell_type] = jnp.array(data, dtype=jnp.int32)
            
    return {
        "points": points,
        "cells": cells_dict,
        "point_data": {k: jnp.array(v) for k, v in mesh.point_data.items()},
        "cell_data": {k: [jnp.array(arr) for arr in v] for k, v in mesh.cell_data.items()}
    }

def get_triangle_faces(mesh_jax):
    """
    Helper to extract only triangle faces from the loaded mesh dict.
    Useful for surface meshes.
    """
    if 'triangle' in mesh_jax['cells']:
        return mesh_jax['cells']['triangle']
    else:
        # Check for quads and warn/decompose if needed?
        # For now, return None or empty
        return None

def get_tetra_cells(mesh_jax):
    """
    Helper to extract only tetrahedral cells from the loaded mesh dict.
    Useful for volumetric FEM.
    """
    if 'tetra' in mesh_jax['cells']:
        return mesh_jax['cells']['tetra']
    else:
        return None
