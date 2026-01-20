
import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path
from dmipy_jax.io.connectome2 import fetch_connectome2, load_connectome2_mri
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.distributions.distributions import DD1Gamma
from dmipy_jax.distributions.distribute_models import DistributedModel
from dmipy_jax.core.acquisition import JaxAcquisition
import optax
import equinox as eqx

def main():
    print("=== Connectome 2.0 AxCaliber Analysis ===")
    
    # 1. Load Data
    # Assuming fetch works or data is manual.
    try:
        data_dict = load_connectome2_mri()
        data = data_dict['dwi']
        bvals = data_dict['bvals']
        bvecs = data_dict['bvecs']
        print(f"Data Loaded: {data.shape}")
    except Exception as e:
        print(f"Data load failed ({e}). using MOCK data for demonstration.")
        # Mocking behavior similar to Connectome 2.0
        # D1 = 13ms, D2 = 30ms
        N = 100
        data = jnp.ones((10, 10, 10, N))
        bvals = jnp.concatenate([jnp.linspace(50, 6000, N//2), jnp.linspace(200, 17800, N//2)])
        bvecs = jnp.zeros((N, 3)); bvecs = bvecs.at[:,0].set(1.0)
    
    # 2. Assign Diffusion Times
    # Heuristic based on description:
    # D=13ms (small Delta 13?) -> Delta=0.013? Or is it Diffusion Time tau?
    # Usually "Delta" in Connectome scanner context refers to big Delta.
    # Protocol: 
    #   Set A: Delta = 13 ms? Wait, gradients need time. 
    #   Maybe Delta=13ms is impossible for b=6000 unless delta is very close.
    #   Actully, "Diffusion time 13ms" might mean Delta - delta/3 = 13ms.
    #   Let's assume standard PGSE.
    #   Let's assign Delta = 0.020 for set A, 0.040 for set B?
    #   Wait, the description said "D=13 ms and 30 ms".
    #   I'll assume distinct b-value ranges or interleaved?
    #   If bvals > 10000, it MUST be D=30ms (to achieve high b).
    #   If bvals < 6000, it could be D=13ms.
    
    # Simple split for demo:
    # If we had real data, we'd check json.
    # Here, let's create synthetic scheme with explicit Deltas.
    
    # Assume 2 shells for now:
    big_delta = jnp.where(bvals > 7000e6, 0.030, 0.013) 
    # (Using 7000 as arbitrary cut off if units are s/mm2... wait units)
    # dmipy uses SI s/m2.
    # 6000 s/mm2 = 6000 * 1e6 s/m2 = 6e9.
    
    big_delta = jnp.where(bvals > 7e9, 0.030, 0.013)
    small_delta = jnp.ones_like(bvals) * 0.010 # Assumed
    
    scheme = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs, 
                            big_delta=big_delta, small_delta=small_delta)
    
    print(f"Scheme created. Delta values: {jnp.unique(scheme.big_delta)}")
    
    # 3. Define AxCaliber Model
    cylinder = RestrictedCylinder()
    gamma_dist = DD1Gamma()
    
    # Distribute 'diameter' using Gamma
    axcaliber = DistributedModel(
        cylinder, gamma_dist,
        parameter_map={'diameter': 'radius'}, # Gamma produces 'radius'? No, DD1Gamma range is x.
        # DD1Gamma samples 'x'. We map 'x' to 'diameter'.
        # Wait, usually radius? 
        # RestrictedCylinder takes 'diameter'.
        # If distribution is of diameters, then x -> diameter.
        # If distribution is of radii, then x -> diameter/2?
        # DD1Gamma is generic. Let's assume it models Diameter Distribution.
        # range (0.1um to 20um).
    )
    # We need to map the distributed parameter name.
    # DistributedModel assumes the distribution yields the parameter named in `distribution_parameter_map`?
    # No, DistributedModel integrates `model(..., param=x)` where x is from grid.
    # We need to tell it WHICH parameter is distributed.
    # Actually, DistributedModel __init__ takes `models` and `distribution`.
    # And we assume the distribution applies to *one* parameter? 
    # Current implementation:
    # checks `distribution.parameter_names`? No.
    # It seems I need to verify how `DistributedModel` knows which parameter to vmap over.
    # Reviewing code...
    
    # In `distribute_models.py`:
    # It vmaps over the LAST argument? 
    # No, line 75: `vmap(self.model, in_axes={param_name: 0, ...})`?
    # Actually, look at `DistributedModel` implementation in previous turns.
    # It likely expects a parameter name to be specified or implied.
    
    pass 
    
if __name__ == "__main__":
    main()
