
import jax
import jax.numpy as jnp
from dmipy_jax.signal_models.sphere_models import SphereGPD

def main():
    print("Testing SphereGPD stability...")
    
    model = SphereGPD()
    
    # Parameters from OED Prior + Extremes
    diameters = jnp.array([4.5e-6, 1.0e-9, 50.0e-6, 1e-11]) # meters (1nm, 50um, 0.01nm)
    diffusivities = jnp.array([2e-9, 1e-12, 1e-13, 3e-9]) # m^2/s (1e-12 is likely problem)
    
    # Acquisition
    # b ~ 0 to 3000
    bvals = jnp.linspace(0, 3000, 10) * 1e6 # s/mm^2 -> s/m^2 scaling? 
    # dmipy uses s/mm^2 usually? 
    # WAIT. dmipy works in SI units usually. 
    # b=1000 s/mm^2 = 1e9 s/m^2.
    # AcquisitionProtocol generates b_values.
    # AcquisitionProtocol.max_b_value = 3000.0. 
    # If this is s/mm^2, then in SI it is 3e9.
    # SphereGPD expects SI units for diameter (meters).
    # If bvals are 3000 (s/mm^2 magnitude) but passed as '3000', does GPD handle it?
    # GPD uses `G_mag = sqrt(bvals / ...)`.
    # If b=3000 SI (s/m^2), that is tiny diffusion weighting.
    # If b=3000 s/mm^2, it should be passed as 3e9 s/m^2?
    # Let's check what units GPD expects.
    # `G_mag = ... / (GAMMA * delta)`.
    # If b is small, G is small.
    # `G_mag` is T/m.
    # q = gamma * G * delta / (2pi).
    # q argument = 2pi * q * R = gamma * G * delta * R.
    # If b=3000 (SI), q is tiny. qR is ~ 0.
    # If qR ~ 0, signal is 1.
    
    # If OED passes 3000 as bvalue, and model treats it as SI (s/m^2), it's negligible diffusion.
    # But if prior is meters (4.5e-6), and diff is 2e-9.
    # Everything is SI.
    # So b=3000 s/m^2 is 0.003 s/mm^2. Effectively b=0.
    # There should definitely be NO NaNs for b=0.
    
    # However, if OED passes b=3000, maybe it means 3000e6?
    # Usually users pass b=1000, 2000...
    # If the model expects SI, we must scale b.
    # BUT, let's assume `SphereGPD` handles scaling or expects SI.
    # Let's test both ranges.
    
    print("Test 1: b in 0..3000 (SI / tiny)")
    bvals_si_tiny = jnp.array([0., 1000., 3000.])
    run_test(model, bvals_si_tiny, diameters, diffusivities, "Tiny b")
    
    print("Test 2: b in 0..3e9 (SI / standard dMRI)")
    bvals_si_std = jnp.array([0., 1e9, 3e9])
    run_test(model, bvals_si_std, diameters, diffusivities, "Standard b")
    
def run_test(model, bvals, diams, diffs, name):
    small_delta = 0.010
    big_delta = 0.040
    
    print(f"--- {name} ---")
    for d in diams:
        for D in diffs:
            try:
                sig = model(
                    bvals=bvals, 
                    gradient_directions=None, 
                    diameter=d, 
                    diffusion_constant=D, 
                    big_delta=big_delta, 
                    small_delta=small_delta
                )
                if jnp.isnan(sig).any():
                    print(f"FAIL: NaNs for d={d}, D={D}")
                    # debug specific
                    val = model(bvals=bvals, gradient_directions=None, diameter=d, diffusion_constant=D, big_delta=big_delta, small_delta=small_delta)
                    print(f"Output: {val}")
                else:
                    pass
                    # print(f"OK: d={d}, D={D}, sig={sig}")
            except Exception as e:
                print(f"CRASH: d={d}, D={D}, err={e}")

if __name__ == "__main__":
    main()
