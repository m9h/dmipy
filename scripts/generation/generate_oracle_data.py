import torch
import numpy as np
import os
from dipy.core.gradients import gradient_table

# Import simulator from the cloned repo
# PYTHONPATH must include /app/SBI_dMRI
from models.ball_and_sticks.simulator import BallAndSticksAttenuation, GradientTable

def generate_data():
    print("Generating Oracle Data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================================
    # 1. Define Acquisition Protocol (b=1000, 2000)
    # ==========================================
    # We will generate 100 directions for each shell + some b0s if needed, 
    # but the simulator expects NO b0s in the GradientTable for the attenuation calculation itself.
    # The user requirements said: "b=1000 and b=2000 s/mm^2 (100 directions)".
    
    n_dirs = 100
    # Generate random directions on sphere
    # Using dipy to help generate well distributed points if we wanted, but random is fine for "Oracle" usually.
    # Let's use simple random normalization for self-containment/simplicity or numpy.
    
    def random_directions(n):
        v = np.random.randn(n, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    bvecs_1000 = random_directions(n_dirs)
    bvals_1000 = np.ones(n_dirs) * 1000.0

    bvecs_2000 = random_directions(n_dirs)
    bvals_2000 = np.ones(n_dirs) * 2000.0

    bvals = np.concatenate([bvals_1000, bvals_2000])
    bvecs = np.concatenate([bvecs_1000, bvecs_2000])

    # Create Torch tensors for GradientTable
    # The SBI_dMRI GradientTable expects bvals (G,) and bvecs (3, G)
    bvals_t = torch.tensor(bvals, dtype=torch.float32, device=device)
    bvecs_t = torch.tensor(bvecs.T, dtype=torch.float32, device=device) 

    gtab = GradientTable(bvals=bvals_t, bvecs=bvecs_t)
    
    # ==========================================
    # 2. Instantiate Simulator
    # ==========================================
    simulator = BallAndSticksAttenuation(gtab=gtab, device=device)
    
    # ==========================================
    # 3. Define Parameters (Theta)
    # ==========================================
    # Model 2 (Ball & Sticks Multi-shell) expected format:
    # [d, f1, th1, ph1, ..., fN, thN, phN, d_std]
    # The user requested "Ball & Stick" (singular usually implies 1 stick, but let's check repo default).
    # Repo default `simulate_data.py` uses --nfib 3 by default. 
    # Let's assume 1 stick for "Ball & Stick" unless "Ball & Sticks" implies multiple.
    # User said "Ball & Stick model" (singular). I will generate 1 stick.
    
    n_samples = 1000
    nfib = 1
    modelnum = 2 # Multi-shell formulation
    
    # Parameters:
    # d: diffusivity = 1.7e-3 mm^2/s
    # f: fraction
    # th, ph: orientation
    # d_std: noise/variance param? In `simulator.py`, d_std is the last param.
    # Wait, looking at `simulator.py`:
    #   modelnum=2: [d, f1, th1, ph1, ..., fN, thN, phN, d_std]
    #   "d_std" seems to be part of the distributed model? Or just std dev of diffusivity?
    #   Actually, reading the code: `sig2 = d_std.pow(2)`, `d_alpha = d.pow(2) / sig2`. 
    #   This looks like a Gamma distribution of diffusivities (Zeppelin/Stick distribution).
    #   Use d_std=0 to get a delta function (standard Ball&Stick)?
    #   Code says: "If d_std is ~0, fall back to model 1" logic inside model 2 block.
    #   So we can set d_std = 0.
    
    # Dimensions:
    # 1 (d) + 3*nfib (f, th, ph) + 1 (d_std) = 1 + 3 + 1 = 5 Parameters.
    
    theta = torch.zeros((n_samples, 5), dtype=torch.float32, device=device)
    
    # d = 1.7e-3 s/mm^2. Note: b-values are usually in s/mm^2 (e.g. 1000). 
    # If b=1000, and d=0.0017 (1.7e-3), exp(-b*d) = exp(-1.7). Correct units.
    theta[:, 0] = 1.7e-3
    
    # f1 = Uniform(0.1, 0.9)
    theta[:, 1] = torch.rand(n_samples, device=device) * 0.8 + 0.1
    
    # th1 (theta), ph1 (phi) = Random on sphere
    # Sampling cos(theta) uniformly
    z = torch.rand(n_samples, device=device) * 2 - 1
    theta[:, 2] = torch.acos(z) # theta
    theta[:, 3] = torch.rand(n_samples, device=device) * 2 * np.pi # phi
    
    # d_std = 0 (Standard B&S)
    theta[:, 4] = 0.0
    
    # ==========================================
    # 4. Simulate
    # ==========================================
    print(f"Simulating {n_samples} samples...")
    # simulator expects (N, P), nfib, modelnum
    signals = simulator(theta, nfib=nfib, modelnum=modelnum)
    
    # ==========================================
    # 5. Save Output
    # ==========================================
    out_file = "/data/oracle_sims.npz"
    print(f"Saving to {out_file}...")
    
    np.savez(out_file, 
             signals=signals.cpu().numpy(), 
             theta=theta.cpu().numpy(),
             bvals=bvals,
             bvecs=bvecs)
    
    print("Done!")

if __name__ == "__main__":
    generate_data()
