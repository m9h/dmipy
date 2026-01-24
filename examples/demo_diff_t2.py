
import os
import sys
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dmipy_jax.io.multi_te import MultiTELoader
from dmipy_jax.models.combined_diff_t2 import CombinedStandardModel

class AcquisitionScheme(NamedTuple):
    bvalues: jnp.ndarray
    gradient_directions: jnp.ndarray
    TE: jnp.ndarray

def main():
    # 1. Load Data
    loader = MultiTELoader("/home/mhough/dev/dmipy/data/MultiTE/MTE-dMRI", "sub-03")
    available_tes = loader.load_paper_subset()
    print(f"Loading Paper Subset TEs: {available_tes}")
    
    all_data = []
    all_bvals = []
    all_bvecs = []
    all_tes = []
    
    for te_str in available_tes:
        try:
            data, bvals, bvecs, params = loader.load_data(te_str)
            te_ms = float(te_str)
            
            # Select central voxel (70,70,35)
            if len(all_data) == 0:
                print(f"Volume shape: {data.shape}")
            
            # Use middle of the brain voxel
            voxel_signal = data[70, 70, 35, :] 
            
            all_data.append(voxel_signal)
            all_bvals.append(bvals)
            all_bvecs.append(bvecs)
            
            # TE in ms. Model expects ms if T2 is in ms.
            all_tes.append(jnp.full(bvals.shape, te_ms)) 
            
        except Exception as e:
            print(f"Skipping TE={te_str}: {e}")
            continue

    if not all_data:
        print("No complete data loaded yet. Exiting.")
        return

    # Concatenate
    signal_measured = jnp.concatenate(all_data)
    bvals = jnp.concatenate(all_bvals)
    bvecs = jnp.concatenate(all_bvecs)
    tes = jnp.concatenate(all_tes)
    
    scheme = AcquisitionScheme(bvalues=bvals, gradient_directions=bvecs, TE=tes)
    print(f"Total measurements: {signal_measured.shape[0]}")
    
    # 2. Setup Model
    model = CombinedStandardModel()
    
    # 3. Simulate Signal with Paper Parameters
    # Fixed Parameters:
    # D_csf = 3.0 um^2/ms = 3.0e-9 m^2/s
    # T2_csf = 2000 ms
    
    # Sampled Parameters (Mean of priors):
    # D_ia, D_ec_par \in [1.0, 3.0] um^2/ms -> Mean 2.0e-9
    # T2_ia, T2_ec \in [40, 200] ms -> Mean 120 ms
    # D_ec_perp / D_ec_par \in [0, 1] -> Mean 0.5 -> D_ec_perp ~ 1.0e-9
    # f_ia, f_ex, f_csf -> Dirichlet means (0.33, 0.33, 0.33)
    
    parameters = {
        'f_in': 0.4,
        'f_ex': 0.4,
        'f_csf': 0.2,
        
        'D_in': 2.0e-9,    # 2.0 um^2/ms
        'T2_in': 80.0,     # ms (Paper says [40, 200])
        
        'D_ex_par': 2.0e-9,
        'D_ex_perp': 0.5e-9, # Ratio approx 0.25
        'T2_ex': 60.0,     # ms
        
        'D_csf': 3.0e-9,   # FIXED
        'T2_csf': 2000.0,  # FIXED
        
        'mu': jnp.array([1.0, 0.0, 0.0]) # x-axis
    }
    
    # SI Units Check:
    # bvals in dataset are likely s/mm^2 (e.g. 1000, 2000).
    # If D is in m^2/s (e.g. 2e-9), product b*D is:
    # 1000 s/mm^2 * 2e-9 m^2/s = 1000 * 1e6 s/m^2 * 2e-9 m^2/s = 2000 * 10^-3 = 2.0 -> Reasonable attenuation.
    # WAIT. b=1000 s/mm^2 = 1e9 s/m^2.
    # b*D = 1e9 * 2e-9 = 2.0. Correct.
    # Dataset usually provides bvals in s/mm^2.
    # We must ensure D inputs are converted to compatible units OR bvals converted to SI.
    # Let's assume standard bvals (0, 700, 2000).
    # If we use D in um^2/ms: 1 um^2/ms = 1e-9 m^2/s.
    # b=1000 s/mm^2. D=1 um^2/ms = 1e-3 mm^2/s.
    # b*D = 1000 * 1e-3 = 1.0. Correct.
    # So if providing D in e-9 (SI), we need bvals in SI (s/m^2).
    # OR provide D in mm^2/s (1e-3).
    # The paper says ranges [1.0, 3.0] um^2/ms.
    # 1.0 um^2/ms = 1.0 * 10^-9 m^2/s = 1.0 * 10^-3 mm^2/s.
    # Let's use SI for D (e-9) and convert bvals to SI (multiply by 1e6).
    
    # Inspect bvals
    print(f"B-values stats: min={bvals.min()}, max={bvals.max()}, mean={bvals.mean()}")
    print(f"Unique b-values (rounded): {jnp.unique(jnp.round(bvals, -1))}")

    # 3. Simulate Signal with Paper Parameters
    # ... (parameters same as above) ...
    
    # Correction: Model predicts signal relative to S(TE=0)=1.
    # Measured data is arbitrary units.
    # We should normalize measured data by S(b=0, TE=min).
    # And we should probably interpret 'f' as fractions of signal at TE=0? 
    # Or fractions of PHYSICAL water?
    # If f represents physical water, then at TE=62, the fractions are weighted by e^-62/T2.
    
    # Let's normalize both to S(b=0, TE=62).
    # Shortest TE
    min_te = jnp.min(tes)
    print(f"Min TE: {min_te} ms")
    
    # Calculate simulated S(b=0, TE=min)
    scheme_norm = AcquisitionScheme(
        bvalues = jnp.array([0.0]),
        gradient_directions = jnp.array([[1.0, 0.0, 0.0]]),
        TE = jnp.array([min_te])
    )
    S0_sim_ref = model(parameters, scheme_norm)
    
    print("Converting bvals to SI (s/m^2)...")
    scheme_si = AcquisitionScheme(
        bvalues = bvals * 1e6,
        gradient_directions = bvecs,
        TE = tes
    )
    
    signal_simulated = model(parameters, scheme_si)
    signal_simulated_norm = signal_simulated / S0_sim_ref
    
    # Normalize Measured
    # Find b=0 at min_te
    mask_ref = (bvals < 10) & (tes == min_te)
    if jnp.sum(mask_ref) > 0:
        S0_meas_ref = jnp.mean(signal_measured[mask_ref])
    else:
        # Fallback
        S0_meas_ref = jnp.max(signal_measured)
        
    signal_measured_norm = signal_measured / S0_meas_ref
    
    print(f"Simulated Mean (Norm): {jnp.mean(signal_simulated_norm):.4f}")
    print(f"Measured Mean (Norm): {jnp.mean(signal_measured_norm):.4f}")
    
    mse = jnp.mean((signal_measured_norm - signal_simulated_norm)**2)
    print(f"MSE: {mse:.6f}")

if __name__ == "__main__":
    main()
