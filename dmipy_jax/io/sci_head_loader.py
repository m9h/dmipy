"""
SCI Head Model Loader

This module provides utilities to load the SCI Head Model (High-Resolution Head and Brain Computer Model)
from the standardized .mat format provided by the SCI Institute.

Reference:
A. Warner, J. Tate, B. Burton, and C.R. Johnson. 2019. 
A High-Resolution Head and Brain Computer Model for Forward and Inverse EEG Simulation. 
bioRxiv doi: 10.1101/552190
"""

import jax.numpy as jnp
import numpy as np
import scipy.io
import h5py
import os
from pathlib import Path

def load_sci_head_mesh(file_path):
    """
    Load the SCI Head Model from a .mat file (SCIRun/Matlab format).
    
    The expected structure in the .mat file is a 'tetmesh' struct containing:
    - node: (3, N_vertices) array of floats (transposed)
    - cell: (4, N_cells) array of indices (1-based, transposed)
    - field: (1, N_cells) array of tissue labels (transposed)
    
    Args:
        file_path (str or Path): Path to the .mat file (e.g., HeadMesh.mat)
        
    Returns:
        dict: A JAX-compatible mesh dictionary containing:
            - 'points': jnp.ndarray of shape (N_vertices, 3)
            - 'cells': dict {'tetra': jnp.ndarray of shape (N_cells, 4)} 0-indexed
            - 'cell_data': dict {'tissue': jnp.ndarray of shape (N_cells,)}
    """
    file_path = str(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SCI Mesh file not found: {file_path}")

    try:
        # Try standard scipy load (for v5/v6 mat files)
        mat = scipy.io.loadmat(file_path)
        is_hdf5 = False
    except Exception:
        # Try h5py (for v7.3 mat files)
        is_hdf5 = True

    if is_hdf5:
        return _load_sci_head_hdf5(file_path)
    else:
        return _load_sci_head_scipy(mat)

def _load_sci_head_scipy(mat):
    """Helper to parse scipy-loaded mat dict."""
    if 'tetmesh' not in mat:
        raise ValueError("Invalid SCI Mesh: 'tetmesh' key not found in .mat file")
    
    tetmesh = mat['tetmesh']
    
    # Access fields from structured array
    # dim 0 is usually 0,0 because it's a 1x1 struct array
    nodes = tetmesh['node'][0, 0]   # Shape (3, N)
    cells = tetmesh['cell'][0, 0]   # Shape (4, M)
    field = tetmesh['field'][0, 0]  # Shape (1, M) or (M, 1) usually (1, M)

    # Convert to JAX-friendly format
    
    # 1. Vertices: Transpose (3, N) -> (N, 3)
    points = jnp.array(nodes.T, dtype=jnp.float32)
    
    # 2. Cells: Transpose (4, M) -> (M, 4) AND Convert 1-based to 0-based
    cells_array = jnp.array(cells.T, dtype=jnp.int32) - 1
    
    # 3. Tissue Labels: Transpose/Flatten
    tissue_labels = jnp.array(field.flatten(), dtype=jnp.int32)
    
    return {
        "points": points,
        "cells": {"tetra": cells_array},
        "cell_data": {"tissue": tissue_labels}
    }

def _load_sci_head_hdf5(file_path):
    """Helper to parse HDF5 (v7.3) mat file."""
    with h5py.File(file_path, 'r') as f:
        if 'tetmesh' not in f:
            raise ValueError("Invalid SCI Mesh: 'tetmesh' group not found in HDF5 file")
            
        # Inspect structure - depends on how struct is saved in HDF5
        # Usually it's a Group or References.
        # Based on previous inspection output, 'tetmesh' is likely a Dataset of Refs or a Group.
        # But previous inspection output showed:
        # tetmesh: shape=(1, 1), dtype=[('node', 'O'), ('cell', 'O'), ('field', 'O')]
        # This implies it was loaded via scipy.io (which supports struct arrays).
        # HDF5 structure is usually different (groups).
        
        # If we are here, it means scipy failed, so it IS v7.3.
        # In v7.3, structs are groups.
        tetmesh_grp = f['tetmesh']
        
        # In Matlab v7.3 HDF5, arrays are stored as (N, 3) if they were (3, N) in Matlab due to C/F order reversal?
        # Actually H5py reads them 'as is' but transposed relative to Matlab?
        # Let's assume standard Matlab v7.3 behavior:
        # If Matlab has (3, N), H5py sees (N, 3) often.
        
        # However, let's look at key names.
        # Checks keys inside tetmesh group
        # If it's a struct, keys should be 'node', 'cell', 'field'.
        pass 
        # CAUTION: Implementing full v7.3 support without verifying the specific structure is risky.
        # But our inspection output earlier worked with scipy.io.loadmat (or implied it did).
        # Wait, I ran `inspect_sci_mesh.py` and it outputted:
        # "tetmesh: shape=(1, 1), dtype=[('node', 'O'), ('cell', 'O'), ('field', 'O')]"
        # This format matches scipy.io.loadmat output (structured numpy array).
        # So scipy.io.loadmat DID work in the inspection step (after I fixed module not found).
        # So I probably don't need HDF5 fallback for the current file.
        # I will leave this as a stub that raises NotImplementedError if scipy fails,
        # unless I specifically need it.
        
        raise NotImplementedError("HDF5 v7.3 loading details need verification. Use scipy.io compatible files.")

