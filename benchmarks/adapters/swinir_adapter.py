import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# Add SwinIR to path
SWINIR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'benchmarks', 'external', 'SwinIR')
if SWINIR_PATH not in sys.path:
    sys.path.append(SWINIR_PATH)

try:
    from models.network_swinir import SwinIR
except ImportError as e:
    raise ImportError(f"Could not import SwinIR from {SWINIR_PATH}. Make sure the submodule is cloned.") from e

class SwinIRPredictor:
    def __init__(self, model_path=None, scale=4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scale = scale
        
        # Default configuration for 'classical_sr' (lightweight/middle confiugration often used)
        # These params arguably should be configurable, but we start with standard SwinIR-M
        # depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6]
        self.model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        
        if model_path:
            self.load_weights(model_path)
        
        self.model.eval()
        self.model = self.model.to(self.device)

    def load_weights(self, path):
        if not os.path.exists(path):
            print(f"Warning: Model weights not found at {path}. Using random initialization.")
            return
            
        pretrained_model = torch.load(path)
        param_key_g = 'params'
        
        # Handle dict wrapping if present
        if param_key_g in pretrained_model.keys():
            state_dict = pretrained_model[param_key_g]
        else:
            state_dict = pretrained_model
            
        self.model.load_state_dict(state_dict, strict=True)
        print(f"Loaded SwinIR weights from {path}")

    def predict_slice(self, slice_2d):
        """
        Runs inference on a single 2D slice.
        Args:
            slice_2d: (H, W) numpy array, normalized [0, 1]
        Returns:
            sr_slice: (H*scale, W*scale) numpy array
        """
        # SwinIR expects 3 channels. We replicate the single channel 3 times.
        img_lq = np.stack([slice_2d]*3, axis=2) # (H, W, 3)
        img_lq = np.transpose(img_lq, (2, 0, 1)) # (3, H, W)
        
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device) # (1, 3, H, W)
        
        with torch.no_grad():
            # Pad if necessary (simple padding for now, could use the tiling logic from main_test_swinir if needed)
            _, _, h_old, w_old = img_lq.size()
            window_size = 8
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = self.model(img_lq)
            
            # Crop back
            output = output[..., :h_old * self.scale, :w_old * self.scale]
            
        output = output.data.squeeze().float().cpu().numpy() # (3, H_sr, W_sr)
        
        # Average channels back to 1 (they should be identical if input was identical, but averaging is safer)
        output = np.mean(output, axis=0)
        
        return np.clip(output, 0, 1)

    def predict_volume(self, volume_4d):
        """
        Runs inference on a 4D dMRI volume (X, Y, Z, B).
        Iterates over Z (axial slices) and B (gradient directions).
        """
        H, W, Z, B = volume_4d.shape
        H_sr, W_sr = H * self.scale, W * self.scale
        
        output_vol = np.zeros((H_sr, W_sr, Z, B), dtype=np.float32)
        
        # Max-min normalization per volume or per slice? 
        # Usually MRI is normalized by max intensity of the volume.
        max_val = np.max(volume_4d)
        if max_val == 0:
            return output_vol
            
        norm_vol = volume_4d / max_val
        
        total_slices = Z * B
        pbar = tqdm(total=total_slices, desc="SwinIR Inference")
        
        for z in range(Z):
            for b in range(B):
                slice_data = norm_vol[:, :, z, b]
                sr_slice = self.predict_slice(slice_data)
                output_vol[:, :, z, b] = sr_slice
                pbar.update(1)
                
        pbar.close()
        
        # Denormalize
        return output_vol * max_val
