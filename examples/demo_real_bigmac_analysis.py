
"""
Demo: Analyzing Real BigMac Dataset with Dmipy-JAX.

This script demonstrates:
1. Loading the BigMac dataset (DWI, acquisition scheme, mask).
2. Fitting a standard NODDI-like model (Stick + Zeppelin + Ball).
3. Visualizing the results (ICVF map).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.io.datasets import load_bigmac_mri
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, sphere_models
from dmipy_jax.fitting.optimization import OptimistixFitter

def main():
    print("Loading BigMac Dataset...")
    try:
        # Load a small slice to keep it fast for demo
        # Center slice roughly
        sl = (slice(100, 150), slice(100, 150), slice(100, 101))
        # Note: Adjust slices based on actual image size if known, else use None for full
        # Safe small load logic:
        # If dataset not present, this will raise FileNotFoundError or start fetching.
        data_dict = load_bigmac_mri(voxel_slice=sl)
        
        dwi = data_dict['dwi']
        scheme = data_dict['scheme']
        mask = data_dict['mask']
        
        print(f"Loaded DWI shape: {dwi.shape}")
        print(f"b-values: {scheme.bvalues.shape}")
        
    except FileNotFoundError as e:
        print(f"Could not load BigMac data: {e}")
        print("Please ensure the dataset is downloaded using `datalad` or `fetch_bigmac`.")
        return

    # Define Model: Simple NODDI (Stick + Zeppelin + Ball)
    # 1. Intra-neurite: Stick
    stick = cylinder_models.C1Stick()
    
    # 2. Extra-neurite: Zeppelin
    zeppelin = cylinder_models.C2Zeppelin()
    
    # 3. CSF: Ball (Isotropic)
    ball = sphere_models.S0Ball()
    
    print("Constructing Model...")
    model = JaxMultiCompartmentModel(models=[stick, zeppelin, ball])
    
    # Define Fitter
    # We use AMICO for initialization if available, or simple branded fitting
    # Here we use the generic OptimistixFitter
    fitter = OptimistixFitter(model)
    
    print("Fitting Model...")
    # Flatten spatial dims for fitting typically, or use vmapped fit
    # OptimistixFitter.fit might expect flat data or handle it.
    # Let's flatten to be safe: (N_vox, N_meas)
    n_vox = np.prod(dwi.shape[:-1])
    n_meas = dwi.shape[-1]
    
    dwi_flat = dwi.reshape(-1, n_meas)
    
    # Run fit
    # This returns a PyTree of parameters
    params = fitter.fit(scheme, dwi_flat)
    
    print("Fit Complete.")
    
    # Extract ICVF (Intra-cellular Volume Fraction)
    # In this 3-compartment model, it's roughly the Stick fraction.
    # Parameters are usually: partial_volume_0, partial_volume_1, partial_volume_2
    # But names depend on model construction order.
    # [Stick, Zeppelin, Ball] -> pv_0, pv_1, pv_2
    
    stick_fraction = params['partial_volume_0']
    
    # Reshape back to image
    icvf_map = stick_fraction.reshape(dwi.shape[:-1])
    
    print(f"Mean ICVF: {jnp.mean(icvf_map)}")
    
    # Visualize if running interactively or save
    # plt.imshow(icvf_map[:, :, 0], cmap='gray', vmin=0, vmax=1)
    # plt.title("BigMac ICVF (Stick Fraction)")
    # plt.savefig("bigmac_icvf_demo.png")
    # print("Saved bigmac_icvf_demo.png")

if __name__ == "__main__":
    main()
