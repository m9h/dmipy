import jax.numpy as jnp
from dmipy_jax.io.sci_head_loader import load_sci_head_mesh
import numpy as np

def verify_loader():
    file_path = 'data/SCI_headmodel/extracted/HeadMesh.mat'
    print(f"Loading mesh from: {file_path}")
    
    mesh = load_sci_head_mesh(file_path)
    
    print("Mesh loaded successfully.")
    print(f"Keys: {mesh.keys()}")
    
    points = mesh['points']
    cells = mesh['cells']['tetra']
    tissues = mesh['cell_data']['tissue']
    
    print(f"Points shape: {points.shape}, dtype: {points.dtype}")
    print(f"Cells shape: {cells.shape}, dtype: {cells.dtype}")
    print(f"Tissue Labels shape: {tissues.shape}, dtype: {tissues.dtype}")
    
    print("-" * 20)
    print(f"Points Range: min={points.min(axis=0)}, max={points.max(axis=0)}")
    print(f"Cells Range: min={cells.min()}, max={cells.max()}")
    print(f"Tissue Values: {jnp.unique(tissues)}")
    
    if cells.min() < 0:
        print("ERROR: Negative cell indices found!")
    if cells.max() >= points.shape[0]:
        print(f"ERROR: Cell index {cells.max()} exceeds point count {points.shape[0]}")
    else:
        print("Index check passed.")

if __name__ == "__main__":
    verify_loader()
