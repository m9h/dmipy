import scipy.io
import numpy as np

file_path = 'data/SCI_headmodel/extracted/HeadMesh.mat'
try:
    mat = scipy.io.loadmat(file_path)
    print(f"Keys in {file_path}: {list(mat.keys())}")
    
    for key in mat:
        if key.startswith('__'): continue
        val = mat[key]
        if isinstance(val, np.ndarray):
            print(f"{key}: shape={val.shape}, dtype={val.dtype}")
            if val.size < 20:
                print(f"  Sample: {val}")
        else:
            print(f"{key}: {type(val)}")
            
except Exception as e:
    print(f"Failed to load .mat: {e}")
    # Try h5py just in case it's v7.3
    try:
        import h5py
        with h5py.File(file_path, 'r') as f:
            print(f"Keys in {file_path} (HDF5): {list(f.keys())}")
            for key in f.keys():
                print(f"{key}: shape={f[key].shape}")
    except Exception as e2:
        print(f"Failed h5py load: {e2}")
