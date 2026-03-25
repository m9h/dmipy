import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from dmipy_jax.io.frdr import FRDRMouseLoader
from dmipy_jax.io.allen import AllenCCFLoader
from dmipy_jax.pipelines.rodent import RodentPreprocessing
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues

# Optional: Set Seaborn theme
sns.set_theme(style="whitegrid")

def fit_dti_ols(dwi_data, bvals, bvecs, mask):
    """
    Fits DTI using Ordinary Least Squares on log-signal.
    Returns FA and MD maps.
    """
    # 1. Prepare Design Matrix B (N_meas, 7)
    # ln(S) = ln(S0) - b * gT D g
    #       = ln(S0) - b * (g_x^2 Dxx + g_y^2 Dyy + g_z^2 Dzz + 2 g_x g_y Dxy + ...)
    # Regressors: [1, -b*gx^2, -b*gy^2, -b*gz^2, -2b*gx*gy, -2b*gx*gz, -2b*gy*gz]
    # Params: [ln(S0), Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    
    N = len(bvals)
    B = np.zeros((N, 7))
    B[:, 0] = 1.0
    bx, by, bz = bvecs.T
    
    B[:, 1] = -bvals * bx**2
    B[:, 2] = -bvals * by**2
    B[:, 3] = -bvals * bz**2
    B[:, 4] = -2 * bvals * bx * by
    B[:, 5] = -2 * bvals * bx * bz
    B[:, 6] = -2 * bvals * by * bz
    
    # 2. Linear Fit
    # S_log = B @ X
    # X = pinv(B) @ S_log
    
    S_data = dwi_data[mask] # (N_vox, N_meas)
    if S_data.shape[0] == 0:
        return np.zeros(dwi_data.shape[:-1]), np.zeros(dwi_data.shape[:-1])
        
    S_log = np.log(np.maximum(S_data, 1e-9)).T # (N_meas, N_vox)
    
    # Pinv
    B_pinv = np.linalg.pinv(B)
    X = B_pinv @ S_log # (7, N_vox)
    
    # 3. Extract Tensor & Metrics
    # D_tens:
    # [[Dxx, Dxy, Dxz],
    #  [Dxy, Dyy, Dyz],
    #  [Dxz, Dyz, Dzz]]
    
    Dxx = X[1, :]
    Dyy = X[2, :]
    Dzz = X[3, :]
    Dxy = X[4, :]
    Dxz = X[5, :]
    Dyz = X[6, :]
    
    # Eigendecomposition (vectorized via numpy or loop)
    # Using a loop for clarity/safety on standard CPU
    
    vals = np.zeros((3, X.shape[1]))
    
    # Construct symmetric matrices (N_vox, 3, 3)
    # This can be heavy, so we might skip full eig if just MD/FA needed approx?
    # No, need eigenvalues for FA.
    
    tensors = np.zeros((X.shape[1], 3, 3))
    tensors[:, 0, 0] = Dxx
    tensors[:, 1, 1] = Dyy
    tensors[:, 2, 2] = Dzz
    tensors[:, 0, 1] = Dxy; tensors[:, 1, 0] = Dxy
    tensors[:, 0, 2] = Dxz; tensors[:, 2, 0] = Dxz
    tensors[:, 1, 2] = Dyz; tensors[:, 2, 1] = Dyz
    
    evals = np.linalg.eigvalsh(tensors) # (N_vox, 3)
    
    # MD
    md_flat = np.mean(evals, axis=1)
    
    # FA
    # sqrt(3/2) * sqrt( sum((lam - md)^2) ) / sqrt( sum(lam^2) )
    num = np.sqrt(np.sum((evals - md_flat[:, None])**2, axis=-1))
    denom = np.sqrt(np.sum(evals**2, axis=-1))
    fa_flat = np.sqrt(1.5) * num / (denom + 1e-9)
    fa_flat = np.clip(fa_flat, 0, 1)
    
    # Reshape back
    fa_map = np.zeros(dwi_data.shape[:-1])
    md_map = np.zeros(dwi_data.shape[:-1])
    
    fa_map[mask] = fa_flat
    md_map[mask] = md_flat
    
    return fa_map, md_map

def generate_mock_data(output_dir, n_subjects=2):
    """
    Generates a mock FRDR dataset structure for verification.
    """
    print(f"Generating MOCK FRDR data in {output_dir}...")
    
    subjects = [f"{i:02d}" for i in range(1, n_subjects+1)]
    
    for sub in subjects:
        # Create structure
        sess_dir = os.path.join(output_dir, f"sub-{sub}", "ses-01")
        dwi_dir = os.path.join(sess_dir, "dwi")
        anat_dir = os.path.join(sess_dir, "anat")
        os.makedirs(dwi_dir, exist_ok=True)
        os.makedirs(anat_dir, exist_ok=True)
        
        # Geometry
        shape = (64, 64, 30)
        affine = np.eye(4)
        affine[0,0] = affine[1,1] = 0.1 # 100um mock
        affine[2,2] = 0.1
        
        # 1. Create Anat (T2w) - Simple sphere
        anat_data = np.zeros(shape)
        x, y, z = np.ogrid[:64, :64, :30]
        # Sphere mask
        mask = (x-32)**2 + (y-32)**2 + (z-15)**2 < 25**2
        anat_data[mask] = 1000.0 + np.random.randn(*anat_data[mask].shape)*50
        
        anat_path = os.path.join(anat_dir, f"sub-{sub}_ses-01_T2w.nii.gz")
        nib.save(nib.Nifti1Image(anat_data, affine), anat_path)
        
        # 2. Create DWI
        n_meas = 10
        dwi_data = np.zeros(shape + (n_meas,))
        # b0
        dwi_data[..., 0] = anat_data # b0 looks like T2
        # dwi
        dwi_data[..., 1:] = anat_data[..., None] * 0.5 # attenuated
        
        dwi_path = os.path.join(dwi_dir, f"sub-{sub}_ses-01_dwi.nii.gz")
        nib.save(nib.Nifti1Image(dwi_data, affine), dwi_path)
        
        # bvals/bvecs
        bvals = np.concatenate([[0], np.ones(n_meas-1)*1000])
        np.savetxt(dwi_path.replace(".nii.gz", ".bval"), bvals)
        
        bvecs = np.random.randn(n_meas, 3)
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        np.savetxt(dwi_path.replace(".nii.gz", ".bvec"), bvecs.T)
        
    print("Mock data generation complete.")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Full FRDR Mouse Analysis with Allen Integration")
    parser.add_argument("--frdr-root", required=False, help="Path to FRDR Dataset (or mock output)")
    parser.add_argument("--output-dir", default="results_rodent_analysis", help="Output directory")
    parser.add_argument("--resolution", type=int, default=50, help="Allen CCF resolution (um)")
    parser.add_argument("--mock-data", action="store_true", help="Generate and use mock data for verification")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mock_data:
        # Generate mock data in a subdirectory of output
        mock_root = os.path.join(args.output_dir, "mock_frdr_data")
        generate_mock_data(mock_root)
        args.frdr_root = mock_root
        
    if not args.frdr_root:
        print("Error: Must specify --frdr-root OR --mock-data")
        return
    
    # 1. Initialize Loaders
    frdr_loader = FRDRMouseLoader(args.frdr_root)
    allen_loader = AllenCCFLoader()
    
    try:
        ccf_paths = allen_loader.fetch_ccfv3(args.resolution)
        print(f"Allen CCFv3 Ready: {ccf_paths['template']}")
    except Exception as e:
        print(f"Skipping Allen steps: {e}")
        return

    subjects = frdr_loader.get_subjects()
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    results = []
    
    for sub in subjects:
        print(f"\nProcessing Subject {sub}...")
        sub_out = os.path.join(args.output_dir, f"sub-{sub}")
        pipeline = RodentPreprocessing(sub_out)
        
        try:
            # Load Data (Session 01)
            data = frdr_loader.load_subject(sub, "01")
            
            # A. N4 Bias Correction
            t2w_n4 = pipeline.run_n4_bias_correction(data['anat_nii'])
            
            # B. Register Anat to Allen (Affine)
            # This enables group analysis in standard space OR mapping atlas to subject
            reg_res = pipeline.register_to_allen(t2w_n4, ccf_paths['template'])
            print(f"  Registration complete: {reg_res['warped_anat']}")
            
            # C. Register DWI to Anat (Rigid)
            # Needed to link DWI space to T2w space
            # Compute Mean b0 first
            img_dwi = nib.load(data['dwi_nii'])
            dwi_arr = img_dwi.get_fdata()
            bvals = np.loadtxt(data['bval'])
            bvecs = np.loadtxt(data['bvec']).T
            
            mean_b0 = np.mean(dwi_arr[..., bvals < 50], axis=-1)
            mean_b0_path = os.path.join(sub_out, "mean_b0.nii.gz")
            nib.save(nib.Nifti1Image(mean_b0, img_dwi.affine), mean_b0_path)
            
            dwi_to_anat_tx = pipeline.register_dwi_to_anat(mean_b0_path, t2w_n4)
            
            # D. Warp Allen Labels to Subject DWI Space
            # Chain: Allen -> Anat (Inverse Affine) -> DWI (Inverse Rigid)
            # For simplicity in this demo, we assume DWI and Anat are aligned (or close enough after rigid)
            # and just map Allen -> Anat.
            # Ideally: Compose transforms. SimpleITK can composite.
            
            # Warp Allen -> Anat
            labels_in_anat = pipeline.warp_labels_to_subject(
                ccf_paths['annotation'], 
                reg_res['inverse_transform_path'], 
                t2w_n4
            )
            
            # E. Fit Microstructure (FA)
            # Simple OLS DTI for demo
            mask = mean_b0 > 0.1 * np.max(mean_b0)
            fa_map, md_map = fit_dti_ols(dwi_arr, bvals, bvecs, mask)
            
            # F. Extract ROI Stats
            label_img = nib.load(labels_in_anat).get_fdata()
            
            # Example ROIs (IDs from Allen Ontology)
            # 382: Field CA1 (Hippocampus)
            # 672: Caudoputamen (Striatum)
            # 997: Root (Whole Brain)
            # Note: IDs depend on the specific annotation version.
            
            rois = {
                "Hippocampus": 382,
                "Striatum": 672,
                "WholeBrain": 997
            }
            
            # Since annotation is often hierarchical, we might need masks.
            # Assuming simple match for demo.
            for roi_name, roi_id in rois.items():
                roi_mask = (label_img == roi_id)
                if np.sum(roi_mask) > 0:
                    mean_fa = np.mean(fa_map[roi_mask])
                    results.append({
                        "Subject": sub,
                        "Region": roi_name,
                        "FA": mean_fa
                    })
                    
            # G. QC Visualization (Intermediary)
            # Overlay edges of Warped Anat on Allen Template
            # We skip this for specific plotting tool but print path
            print(f"  QC: Warped Anat saved to {reg_res['warped_anat']}")
            
        except Exception as e:
            print(f"Failed processing subject {sub}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. Group Statistics Visualization
    if results:
        df = pd.DataFrame(results)
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="Region", y="FA", palette="muted")
        plt.title("Group Analysis: FA by Region (Allen CCFv3)")
        plt.savefig(os.path.join(args.output_dir, "group_stats_fa.png"))
        print(f"\nGroup Analysis Plot saved to {os.path.join(args.output_dir, 'group_stats_fa.png')}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
