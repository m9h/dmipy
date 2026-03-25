
import numpy as np
import scipy.optimize
import os
import argparse
from dmipy_jax.io.ds004024 import Ds004024Loader
from dmipy_jax.biophysics.network.connectome_mapper import ConnectomeMapper

def main():
    parser = argparse.ArgumentParser(description="Demo TMS-EEG Velocity Calibration on ds004024")
    parser.add_argument("--data-root", type=str, default="data/ds004024", help="Path to dataset root")
    parser.add_argument("--subject", type=str, default="CON001", help="Subject ID (e.g. 'CON001')")
    args = parser.parse_args()
    
    print(f"--- TMS-EEG Velocity Calibration Agent (ds004024) ---")
    print(f"Dataset Root: {args.data_root}")
    print(f"Subject: sub-{args.subject}")
    
    loader = Ds004024Loader(args.data_root)
    
    # 1. Load Data
    print("Loading DWI and TMS Latencies...")
    data, affine, bvals, bvecs = loader.load_dwi(args.subject)
    latencies = loader.load_tms_latencies(args.subject)
    
    print(f"DWI Shape: {data.shape}")
    print(f"Empirical Latencies: {latencies}")
    
    # 2. Fit Microstructure (Simplified Axon Diameter Proxy)
    # In reality: Fit AxCaliber or similar.
    # Here: We use a placeholder 'diameter_map' derived from the data mean for demo.
    print("Fitting Microstructure Model (Proxy)...")
    diameter_map = np.mean(data[..., 1:], axis=-1) # Just signal intensity as proxy
    # Normalize to uM range (0.5 - 5.0)
    diameter_map = (diameter_map - diameter_map.min()) / (diameter_map.max() - diameter_map.min() + 1e-9)
    diameter_map = diameter_map * 4.0 + 0.5
    
    # 3. Tractography / Parcellation (Mock)
    # We need a parcellation and streamlines to map.
    # For this demo, we generate synthetic streamlines if we don't have real ones.
    print("Generating Streamlines (Mock for Demo)...")
    shape = data.shape[:3]
    streamlines = []
    # Create a streamline from middle to middle+5
    start = np.array(shape) // 2
    end = start + 5
    # Interpolate
    points = np.linspace(start, end, 10).astype(float)
    streamlines.append(points)
    
    # ROI definitions
    # Map L_M1 and R_M1 to these endpoints for the demo math to close loop
    parcellation = np.zeros(shape, dtype=int)
    s_idx = start.astype(int)
    e_idx = end.astype(int)
    
    # Check bounds
    s_idx = np.clip(s_idx, 0, np.array(shape)-1)
    e_idx = np.clip(e_idx, 0, np.array(shape)-1)
    
    parcellation[tuple(s_idx)] = 1 # Src
    parcellation[tuple(e_idx)] = 2 # Tgt
    
    region_names = {1: "L_M1", 2: "R_M1"}
    
    # 4. Calibration Loop
    # We want to find 'k' such that simulated latency matches empirical.
    # Target: L_M1 -> R_M1 (Mock latency = 12.5 ms)
    target_lat = latencies.get(("L_M1", "R_M1"), 12.5)
    
    def objective(k):
        if k <= 0: return 1e6
        # n_regions = 3 (0,1,2)
        delays = ConnectomeMapper.map_microstructure_to_velocity(
            streamlines, diameter_map, affine, parcellation, n_regions=3, base_velocity=k
        )
        # Latency between 1 and 2
        sim = delays[0, 1] # Index 0=ROI 1, Index 1=ROI 2
        return (sim - target_lat)**2
        
    print(f"Calibrating 'k' to match target latency {target_lat} ms...")
    res = scipy.optimize.minimize_scalar(objective, bounds=(1.0, 20.0), method='bounded')
    
    if res.success:
        print(f"Calibration Successful! Optimal k = {res.x:.4f} m/s/um")
    else:
        print("Calibration Failed.")

if __name__ == "__main__":
    main()
