import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def load_vol(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    img = nib.load(path)
    data = img.get_fdata()
    # Handle 4D
    if data.ndim == 4:
        data = data[..., 0]
    # Normalize 0-1 for display
    p99 = np.percentile(data, 99)
    if p99 > 0:
        data = data / p99
    return np.rot90(data) # Rotate for axial view if needed

def main():
    base_dir = "/home/mhough/datasets/ds001957-study/derivatives/super_resolution/sub-01/dwi"
    t1_path = "/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/sub-01/anat/t1.nii.gz"
    
    # Define files
    files = [
        ("T1w (Reference)", t1_path),
        ("Baseline (Sinc)", os.path.join(base_dir, "sub-01_desc-baseline_dwi.nii.gz")),
        ("MMORF (Affine)", os.path.join(base_dir, "sub-01_desc-mmorf_dwi.nii.gz")),
        ("JTV (SCICO)", os.path.join(base_dir, "sub-01_desc-jtv_dwi.nii.gz")),
        ("INR (Equinox)", os.path.join(base_dir, "sub-01_desc-inr_dwi.nii.gz")),
    ]
    
    # Load data
    loaded = []
    for name, path in files:
        vol = load_vol(path)
        if vol is not None:
            loaded.append((name, vol))
            
    if not loaded:
        print("No images loaded!")
        return

    # Load Mask
    # Use the officially staged T1 mask
    mask_path = "/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/sub-01/anat/mask.nii.gz"
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found at {mask_path}")
        return

    mask_img = nib.load(mask_path)
    t1_mask = mask_img.get_fdata() > 0
    
    # Load T1 for shape reference
    t1_vol = load_vol(t1_path)
    if t1_vol is None: return

    # Choose a slice
    # Use T1 shape to find center
    t1_shape = t1_vol.shape
    z_slice = int(t1_shape[2] * 0.55)
    
    # Helper to apply mask
    def apply_mask(vol, mask):
        if vol.shape != mask.shape:
             return vol
        v = vol.copy()
        v[~mask] = 0
        return v


    
    # Create 2x4 Plot
    # We have 5 images. We can show:
    # Row 1: T1, Baseline, MMORF, JTV
    # Row 2: INR, (Empty/Zoom?), (Empty), (Empty)
    # Or maybe show 2 different slices?
    # Let's do 2 Slices (Axial and Coronal?) to fill 2x4?
    # That would require 5 cols. 
    # User asked 2x4. That's 8 slots. 
    # Let's fill remaining slots with Zoomed Views of the first 3 (T1, MMORF, JTV) to highlight differences.
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Row 1: Full Axial Slices
    for i in range(5):
        if i >= len(loaded): break
        name, vol = loaded[i]
        
        # Apply T1 Mask
        vol = apply_mask(vol, t1_mask)

        
        # Check bounds
        sl = min(z_slice, vol.shape[2]-1)
        ax = axes[i]
        ax.imshow(vol[:, :, sl], cmap='gray', clim=(0, 1))
        ax.set_title(name)
        ax.axis('off')
        
    # Row 2 (Remaining 3 slots): Zoomed regions
    # Zoom coordinates (approx center of brain)
    rows, cols = t1_shape[0], t1_shape[1]
    r_start, r_len = int(rows*0.3), int(rows*0.4)
    c_start, c_len = int(cols*0.3), int(cols*0.4)
    
    # We used 5 slots (0,1,2,3,4). Left: 5,6,7. (3 slots)
    # Let's show Zoomed T1, Zoomed MMORF, Zoomed JTV
    zoom_candidates = [0, 2, 3] # Indices in loaded list
    
    current_ax_idx = 5
    for idx in zoom_candidates:
        if idx >= len(loaded): continue
        if current_ax_idx >= 8: break
        
        name, vol = loaded[idx]
        vol = apply_mask(vol, t1_mask)

        sl = min(z_slice, vol.shape[2]-1)
        
        # Crop
        crop = vol[r_start:r_start+r_len, c_start:c_start+c_len, sl]
        
        ax = axes[current_ax_idx]
        ax.imshow(crop, cmap='gray', clim=(0, 1))
        ax.set_title(f"Zoom: {name}")
        ax.axis('off')
        current_ax_idx += 1

    plt.tight_layout()
    out_path = os.path.join(base_dir, "comparison_grid.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
