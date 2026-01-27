import os
import glob
import numpy as np
import nibabel as nib
import jax.numpy as jnp
from typing import Generator, Tuple

class DeepFCDLoader:
    def __init__(self, data_path: str, slice_neighbors: int = 3):
        """
        Args:
            data_path: Root of BIDS dataset (e.g. ~/Data/ds001957).
            slice_neighbors: Number of slices to include on each side (total 2*N+1 context).
        """
        self.data_path = os.path.expanduser(data_path)
        self.slice_neighbors = slice_neighbors
        self.subjects = sorted([d for d in os.listdir(self.data_path) if d.startswith('sub-') and os.path.isdir(os.path.join(self.data_path, d))])
        print(f"Found {len(self.subjects)} subjects in {self.data_path}")

    def load_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and normalizes T1 and FLAIR volumes for a subject."""
        anat_path = os.path.join(self.data_path, subject_id, 'anat')
        
        # Robust globbing for T1w and FLAIR
        t1_files = glob.glob(os.path.join(anat_path, '*T1w.nii.gz'))
        flair_files = glob.glob(os.path.join(anat_path, '*FLAIR.nii.gz'))
        
        if not t1_files or not flair_files:
            print(f"Skipping {subject_id}: Missing T1 or FLAIR.")
            return None, None
            
        t1_img = nib.load(t1_files[0])
        flair_img = nib.load(flair_files[0])
        
        # Get data (H, W, D)
        t1_data = t1_img.get_fdata().astype(np.float32)
        flair_data = flair_img.get_fdata().astype(np.float32)
        
        # Ensure dimensions match. If not, we might need registration/resampling.
        # DeepFCD assumes registered data.
        if t1_data.shape != flair_data.shape:
            # Simple crop/pad to match? Or just warn?
            # For now, warn and skip slices that don't match or crop to min.
            print(f"Warning: Shape mismatch for {subject_id}: {t1_data.shape} vs {flair_data.shape}")
            # Min crop
            min_shape = np.minimum(t1_data.shape, flair_data.shape)
            t1_data = t1_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            flair_data = flair_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            
        # Normalize to [0, 1]
        t1_data = (t1_data - t1_data.min()) / (t1_data.max() - t1_data.min() + 1e-8)
        flair_data = (flair_data - flair_data.min()) / (flair_data.max() - flair_data.min() + 1e-8)
        
        # Pad to be divisible by 16 (for 4 downsamples/U-Net)
        t1_data = self.pad_to_divisible(t1_data, divisor=16)
        flair_data = self.pad_to_divisible(flair_data, divisor=16)
        
        # DEBUG: Print shapes
        print(f"Padded subject {subject_id} to {t1_data.shape}")
        
        return t1_data, flair_data

    def pad_to_divisible(self, data: np.ndarray, divisor: int=16) -> np.ndarray:
        """Pads spatial dimensions (H, W) to be divisible by divisor."""
        h, w = data.shape[:2]
        new_h = int(np.ceil(h / divisor) * divisor)
        new_w = int(np.ceil(w / divisor) * divisor)
        
        if new_h == h and new_w == w:
            return data
            
        pad_h = new_h - h
        pad_w = new_w - w
        
        # Pad H and W with zeros (background/min value)
        # Data is (H, W, D)
        return np.pad(data, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    def get_generator(self, batch_size: int = 8, shuffle: bool = True):
        # Infinite generator
        while True:
            # Pick random subject
            if shuffle:
                subj_idx = np.random.randint(0, len(self.subjects))
            else:
                subj_idx = 0 # Sequential not implemented for infinite loop yet
            
            sub = self.subjects[subj_idx]
            t1, flair = self.load_subject(sub)
            
            if t1 is None:
                continue
                
            # Pick random slices (Axial? usually index 2 for (H,W,D))
            # Assuming (H, W, D) layout. Slices are along D (index 2).
            # DeepFCD usually works on transversal slices.
            
            depth = t1.shape[2]
            # Valid range for center slice
            min_z = self.slice_neighbors
            max_z = depth - self.slice_neighbors
            
            if max_z <= min_z:
                continue
                
            # Sample batch_size indices
            indices = np.random.randint(min_z, max_z, size=batch_size)
            
            # Prepare batch
            context_batch = []
            target_batch = []
            
            for z in indices:
                # Extract 2.5D stack
                # Context: [z-3, ..., z, ..., z+3] -> 7 slices
                # Shape: (H, W, 7) or (7, H, W)?
                # FlowUNet expects (C, H, W).
                
                # Slicing: t1[:, :, z-3:z+4] -> (H, W, 7)
                ctx_vol = t1[:, :, z-self.slice_neighbors : z+self.slice_neighbors+1]
                
                # Transpose to (7, H, W)
                ctx_vol = np.transpose(ctx_vol, (2, 0, 1))
                
                # Target: (1, H, W)
                tgt_slice = flair[:, :, z] # (H, W)
                tgt_slice = tgt_slice[np.newaxis, :, :] # (1, H, W)
                
                context_batch.append(ctx_vol)
                target_batch.append(tgt_slice)
                
            yield np.array(context_batch), np.array(target_batch)

# Wrapper for verify script
def real_data_loader(data_path: str, batch_size: int = 4, in_channels: int = 7) -> Generator:
    loader = DeepFCDLoader(data_path, slice_neighbors=(in_channels-1)//2)
    return loader.get_generator(batch_size=batch_size)
