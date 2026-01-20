import argparse
import time
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import nibabel as nib
from dmipy_jax.io.datasets import load_bigmac_mri
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import Stick, Ball, Zeppelin

def main():
    parser = argparse.ArgumentParser(description="BigMac BAS Analysis using dmipy-jax")
    parser.add_argument("--data_path", type=str, required=True, help="Path to BigMac dataset folder")
    parser.add_argument("--out_path", type=str, default="bigmac_results", help="Output directory")
    parser.add_argument("--slice", type=str, default=None, help="Slice 'z_start:z_end' for testing")
    args = parser.parse_args()
    
    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse slice
    voxel_slice = None
    if args.slice:
        z_s, z_e = map(int, args.slice.split(':'))
        # Assuming typical BigMac dimensions, we crop X/Y too? Or just Z.
        # Let's crop center XY for speed if slicing Z.
        voxel_slice = (slice(None), slice(None), slice(z_s, z_e))
        
    print(f"Loading BigMac data from {args.data_path}...")
    t0 = time.time()
    try:
        results = load_bigmac_mri(args.data_path, voxel_slice=voxel_slice)
    except FileNotFoundError as e:
        print(e)
        return

    data = results['dwi']
    scheme = results['scheme']
    mask = results['mask']
    affine = results.get('affine', np.eye(4))
    
    print(f"Data shape: {data.shape}")
    print(f"Scheme measurements: {len(scheme.bvalues)}")
    print(f"Loading time: {time.time()-t0:.2f}s")
    
    # Define Model: BAS (Ball and Stick)
    # Stick (Intra-axonal), Ball (Extra-axonal / Isotropic)
    # Usually we assume Stick means restricted diffusion along mu.
    # Ball means hindered diffusion (isotropic).
    
    stick = Stick()
    ball = Ball()
    
    # We can fix diffusivity for stability if data is noisy or limited shells.
    # BigMac is post-mortem: diffusivities are lower (~0.6e-9?).
    # Let's set ranges appropriate for fixed tissue.
    # stick.lambda_par range default (0.1e-9, 3e-9) covers it.
    # ball.lambda_iso range default (0, 3e-9) covers it.
    
    # Typically BAS fixes d_par = d_iso? Or fits them.
    # dmipy-jax default fits them.
    
    model = JaxMultiCompartmentModel([stick, ball])
    
    print("Fitting model (this may take time)...")
    t1 = time.time()
    
    # Masking
    # Flatten data within mask
    if mask is not None:
        mask_bool = np.array(mask, dtype=bool)
        data_masked = data[mask_bool]
        # scheme is same
    else:
        # Flatten all
        mask_bool = np.ones(data.shape[:-1], dtype=bool)
        data_masked = data.reshape(-1, data.shape[-1])
        
    print(f"Voxels to fit: {data_masked.shape[0]}")
    
    fitted_params_flat = model.fit(scheme, data_masked)
    
    # Unpack and Map back to volume
    # parameters are dictionaries of flat arrays
    
    print(f"Fitting finished in {time.time()-t1:.2f}s")
    
    # Save results
    # Helper to save NIfTI
    def save_nii(arr, name):
        # Embed in full volume
        vol = np.zeros(mask_bool.shape + arr.shape[1:], dtype=np.float32)
        vol[mask_bool] = np.array(arr)
        img = nib.Nifti1Image(vol, affine)
        nib.save(img, out_dir / f"{name}.nii.gz")
        print(f"Saved {out_dir / name}.nii.gz")

    # Keys in fitted_params_flat
    # Stick parameters: mu, lambda_par
    # Ball parameters: lambda_iso
    # Fractions: partial_volume_0 (stick), partial_volume_1 (ball)
    
    if 'mu' in fitted_params_flat:
        save_nii(fitted_params_flat['mu'], 'mu') # (N_vox, 2) spherical coordinates
        # Convert to cartesian for visualization?
        theta = fitted_params_flat['mu'][:, 0]
        phi = fitted_params_flat['mu'][:, 1]
        st = jnp.sin(theta); ct = jnp.cos(theta)
        sp = jnp.sin(phi); cp = jnp.cos(phi)
        vecs = jnp.stack([st*cp, st*sp, ct], axis=-1)
        save_nii(vecs, 'mu_vec')
        
    if 'lambda_par' in fitted_params_flat:
        save_nii(fitted_params_flat['lambda_par'][:, None], 'lambda_par')
        
    if 'lambda_iso' in fitted_params_flat:
         save_nii(fitted_params_flat['lambda_iso'][:, None], 'lambda_iso')
         
    if 'partial_volume_0' in fitted_params_flat:
         save_nii(fitted_params_flat['partial_volume_0'][:, None], 'f_stick')
         
    if 'partial_volume_1' in fitted_params_flat:
         save_nii(fitted_params_flat['partial_volume_1'][:, None], 'f_ball')

    print("Done.")

if __name__ == "__main__":
    main()
