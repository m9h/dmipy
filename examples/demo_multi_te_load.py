
import os
import sys
import jax.numpy as jnp
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dmipy_jax.io.multi_te import MultiTELoader

def main():
    base_path = "/home/mhough/dev/dmipy/data/MultiTE/MTE-dMRI"
    subject = "sub-03"
    
    loader = MultiTELoader(base_path, subject)
    
    print(f"Checking dataset for {subject} in {base_path}...")
    tes = loader.get_available_tes()
    print(f"Found TEs: {tes}")
    
    for te in tes:
        print(f"\n--- Loading TE={te} ---")
        try:
            data, bvals, bvecs = loader.load_data(te)
            affine = loader.load_image_affine(te)
            
            print(f"  Data shape: {data.shape}")
            print(f"  Bvals shape: {bvals.shape}, range: [{bvals.min():.1f}, {bvals.max():.1f}]")
            print(f"  Bvecs shape: {bvecs.shape}")
            print(f"  Affine:\n{affine}")
            
        except FileNotFoundError as e:
            print(f"  [WARNING] Could not load complete set for TE={te}: {e}")
        except Exception as e:
            print(f"  [ERROR] Failed to load TE={te}: {e}")

if __name__ == "__main__":
    main()
