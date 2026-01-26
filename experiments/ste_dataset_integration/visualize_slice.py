import nibabel as nib
import numpy as np
import sys
import os

def ascii_show(image, cols=80):
    """
    Renders a 2D numpy array as an ASCII image.
    """
    h, w = image.shape
    scale = w / cols
    rows = int(h / scale * 0.5) # Correction for aspect ratio of fonts
    
    if rows == 0: rows = 1
    
    # Resize
    # Simple nearest neighbor
    from skimage.transform import resize
    # We can't assume skimage.
    # Let's just subsample
    
    step_x = max(1, w // cols)
    step_y = max(1, h // rows)
    
    small_img = image[::step_y, ::step_x]
    
    chars = np.asarray(list(' .:-=+*#%@'))
    # Normalize 0-1
    mn = small_img.min()
    mx = small_img.max()
    if mx == mn:
        norm_img = np.zeros_like(small_img)
    else:
        norm_img = (small_img - mn) / (mx - mn)
    
    indices = (norm_img * (len(chars) - 1)).astype(int)
    lines = ["".join(chars[row]) for row in indices]
    return "\n".join(lines)

def main():
    path = "/home/mhough/Downloads/STE/STE00_ExVivo/STE/STE.nii.gz"
    print(f"Loading {path}")
    img = nib.load(path)
    data = img.get_fdata()
    
    # Find b0
    bvals = np.loadtxt("/home/mhough/Downloads/STE/STE00_ExVivo/STE/bvals.txt")
    b0_idx = np.argmin(bvals)
    
    vol = data[..., b0_idx]
    x, y, z = vol.shape
    center_z = z // 2
    
    slice_img = vol[:, :, center_z]
    
    print(f"Slice {center_z} ({x}x{y})")
    print(f"Range: {slice_img.min():.2f} - {slice_img.max():.2f}")
    
    print(ascii_show(slice_img))

    # Also check histogram
    print("\nCorner Stats (20x20):")
    corner = vol[0:20, 0:20]
    print(f"Mean: {np.mean(corner)}")
    print(f"Std:  {np.std(corner)}")

if __name__ == "__main__":
    main()
