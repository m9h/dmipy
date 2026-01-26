import torch
import numpy as np
import time
import os
from dipy.core.gradients import gradient_table

# Import simulator from the cloned repo
# PYTHONPATH must include /app/SBI_dMRI
from models.ball_and_sticks.simulator import BallAndSticksAttenuation, GradientTable

def generate_complex_data():
    print("Starting Complex Oracle Simulation (1M samples)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================================
    # 1. Define Acquisition Protocol
    # ==========================================
    # b=1000 (100 dirs), b=2000 (100 dirs)
    n_dirs = 100
    
    def random_directions(n):
        v = np.random.randn(n, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    # Using fixed seed for directions to allow comparison if needed, 
    # though strict "Oracle" usually implies we just match distribution.
    np.random.seed(42)  
    bvecs_1000 = random_directions(n_dirs)
    bvals_1000 = np.ones(n_dirs) * 1000.0

    bvecs_2000 = random_directions(n_dirs)
    bvals_2000 = np.ones(n_dirs) * 2000.0

    bvals = np.concatenate([bvals_1000, bvals_2000])
    bvecs = np.concatenate([bvecs_1000, bvecs_2000])

    bvals_t = torch.tensor(bvals, dtype=torch.float32, device=device)
    bvecs_t = torch.tensor(bvecs.T, dtype=torch.float32, device=device) 
    gtab = GradientTable(bvals=bvals_t, bvecs=bvecs_t)
    
    # ==========================================
    # 2. Simulator Setup
    # ==========================================
    simulator = BallAndSticksAttenuation(gtab=gtab, device=device)
    
    # ==========================================
    # 3. Large Scale Parameters (1M Samples)
    # ==========================================
    n_samples = 1_000_000
    nfib = 1
    modelnum = 2 
    
    # Batched generation to avoid OOM if VRAM is limited? 
    # 1M floats * 200 dirs * 4 bytes ≈ 800MB. 
    # 1M params * 5 * 4 bytes ≈ 20MB.
    # DGX Spark has plenty of memory. We can do it in one shot.
    
    print(f"Allocating tensors for {n_samples} samples...")
    theta = torch.zeros((n_samples, 5), dtype=torch.float32, device=device)
    
    # Diffusivity d = 1.7e-3 (fixed)
    theta[:, 0] = 1.7e-3
    
    # Fraction f ~ U(0.1, 0.9)
    theta[:, 1] = torch.rand(n_samples, device=device) * 0.8 + 0.1
    
    # Orientation ~ Uniform Sphere
    z = torch.rand(n_samples, device=device) * 2 - 1
    theta[:, 2] = torch.acos(z) # theta
    theta[:, 3] = torch.rand(n_samples, device=device) * 2 * np.pi # phi
    
    # d_std = 0
    theta[:, 4] = 0.0
    
    # ==========================================
    # 4. Simulation & Timing
    # ==========================================
    torch.cuda.synchronize()
    start_time = time.time()
    
    print("Running simulation...")
    # Signal generation (Noise free)
    signals = simulator(theta, nfib=nfib, modelnum=modelnum)
    
    # Rician Noise Additon (SNR=30)
    # Rician noise: sqrt( (S + N1)^2 + N2^2 ) where N1, N2 ~ N(0, sigma)
    # sigma = S0 / SNR. S0=1.0 here (normalized).
    snr = 30.0
    sigma = 1.0 / snr
    
    noise1 = torch.randn_like(signals) * sigma
    noise2 = torch.randn_like(signals) * sigma
    
    signals_noisy = torch.sqrt((signals + noise1)**2 + noise2**2)
    
    torch.cuda.synchronize()
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Simulation complete in {duration:.4f} seconds.")
    print(f"Throughput: {n_samples / duration:,.0f} samples/sec")
    
    # ==========================================
    # 5. Save
    # ==========================================
    out_file = "/data/complex_oracle_1M.npz"
    print(f"Saving to {out_file}...")
    
    np.savez(out_file, 
             signals=signals_noisy.cpu().numpy(),
             signals_clean=signals.cpu().numpy(),
             theta=theta.cpu().numpy(),
             bvals=bvals,
             bvecs=bvecs,
             time_seconds=duration)
    
    print("Done!")

if __name__ == "__main__":
    generate_complex_data()
