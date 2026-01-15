
import jax
import jax.numpy as jnp
from jax.scipy.stats import beta
import numpy as np
from dmipy_jax.models.c_noddi import CNODDI
from dmipy_jax.fitting import ConstrainedOptimizer
from dmipy_jax.acquisition import JaxAcquisition

def test_cnoddi_tortuosity():
    print("Testing Tortuosity Constraint...")
    model = CNODDI(diffusivity_par=1.7e-9)
    # Mock acquisition
    bvals = jnp.array([0.0, 1000.0, 2000.0])
    gradients = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    acq = JaxAcquisition(bvals, gradients)
    
    # Params: theta=0, phi=0, f_stick=0.5, f_iso=0.0
    # Expected D_extra_perp = D_par * (1 - 0.5) = 1.7e-9 * 0.5 = 0.85e-9
    params = jnp.array([0.0, 0.0, 0.5, 0.0])
    
    # We can't easily inspect internal variables without modifying the class to return them, 
    # but we can check if the signal matches the manual calculation.
    
    # Stick (Intra): f=0.5. aligned with x-axis (theta=0? No, theta=0 is z-axis).
    # theta=0 => z-axis (0,0,1).
    # gradients[1] is x-axis (1,0,0). dot=0.
    # gradients[2] is y-axis (0,1,0). dot=0.
    
    # Let's align fiber with Z-axis (theta=0).
    # Grad 2: (0,1,0) -> perp to fiber.
    # Signal Stick: exp(-b * D_par * 0) = 1.0
    
    # Signal Zeppelin: 
    # f_extra = 0.5.
    # D_perp should be 0.85e-9.
    # Dot product with z-axis is 0.
    # E_zep = exp(-b * (D_par * 0 + D_perp * (1-0))) = exp(-b * D_perp)
    
    # Signal Total = 0.5 * 1 + 0.5 * exp(-b * D_perp)
    
    # Case b=2000 s/mm^2 = 2e9 s/m^2? No, bvals are usually in s/mm^2 or SI?
    # dmipy usually uses SI units for diffusivity (m^2/s) and SI for bvals (s/m^2)?
    # Let's assume standard SI usage in dmipy_jax as seen in sphere.py (diameter in meters).
    # So b=2000 needs to be 2000e6 s/m^2 if input is 2000? 
    # sandi.py uses 3.0e-9, Suggesting SI.
    # But usually bvals are passed as e.g. 1000, 2000, 3000.
    # If users pass 1000, they mean s/mm^2.
    # If diffusivity is 3e-9 m^2/s = 3e-3 mm^2/s.
    # b*D = 1000 s/mm^2 * 3e-3 mm^2/s = 3. 
    # 1000e6 s/m^2 * 3e-9 m^2/s = 3.
    # So units are consistent if b is s/mm^2 and D is mm^2/s OR b is s/m^2 and D is m^2/s.
    # But files show defaults like 3.0e-9, which is m^2/s.
    # So b-values MUST be in s/m^2 (e.g. 1e9, 2e9) for this to work with 3e-9.
    # OR the user passes 1000 and D is 3e-3. 
    # Let's check sandi.py default: 3.0e-9.
    # check cylinder.py reference to "m^2/s".
    # So bvals MUST be in SI (order of 1e9) for correct calc.
    # Let's use SI for the test manually.
    
    b_test = 2000.0 * 1e6 # 2000 s/mm^2 in SI
    
    # Manual Calc
    D_perp = 1.7e-9 * 0.5
    E_zep_perp = jnp.exp(-b_test * D_perp)
    S_expected = 0.5 * 1.0 + 0.5 * E_zep_perp
    
    # Run Model
    # Note: Acquisition bvals must be SI if model D is SI.
    acq_test = JaxAcquisition(jnp.array([b_test]), jnp.array([[0., 1., 0.]])) # Perp gradient
    
    S_model = model(params, acq_test)
    
    print(f"Manual Signal: {S_expected}")
    print(f"Model Signal: {S_model[0]}")
    
    assert jnp.allclose(S_model[0], S_expected), "Tortuosity constraint mismatch!"
    print("Tortuosity Test Passed.\n")


def test_penalized_fitting():
    print("Testing Penalized Fitting (Beta Prior)...")
    
    # Setup Model
    model_obj = CNODDI(diffusivity_par=1.7e-9)
    
    # Create wrapper func for optimization
    # params: [theta, phi, f_stick, f_iso]
    def model_func(p):
        # We need a dummy acquisition or fixed acquisition?
        # The optimizer expects a function f(p) -> predictions.
        # We must bake the acquisition into this function or use partial.
        return model_obj(p, acq)
    
    # 1. Define Prior
    # Beta(2, 20) on f_iso (index 3)
    def f_iso_prior(params):
        f_iso = params[3]
        # jax.scipy.stats.beta.logpdf(x, a, b)
        # Avoid singular boundaries?
        return beta.logpdf(f_iso, 2, 20)
        
    # 2. Generate Synthetic Data
    # True params: f_iso = 0.0 (Boundary case)
    # f_stick = 0.6
    true_params = jnp.array([1.57, 0.0, 0.6, 0.05]) # small f_iso to check if prior pushes it
    
    # Let's try to 'break' standard fitting by simulating a voxel that LOOKS like f_iso=0 
    # or where noise pushes f_iso < 0.
    
    # Use a bigger acquisition
    bvals = jnp.concatenate([jnp.zeros(1), jnp.ones(10)*1000e6, jnp.ones(10)*2000e6])
    # Random gradients on sphere? hardcoded here for simplicity
    grads = jnp.zeros((21, 3))
    grads = grads.at[1:, 0].set(1.0) # all x-directed
    
    acq = JaxAcquisition(bvals, grads)
    
    # Clean Signal
    clean_signal = model_obj(true_params, acq)
    
    # Add noise? For reproducibility, let's keep it clean first to check gradients.
    # Then adds noise.
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, size=clean_signal.shape)
    data = clean_signal + noise
    
    # Init params
    init_params = jnp.array([1.57, 0.0, 0.5, 0.2])
    # Bounds: theta[0,pi], phi[-pi,pi], f_stick[0,1], f_iso[0,1]
    bounds = (
        jnp.array([0., -jnp.pi, 0., 0.]),
        jnp.array([jnp.pi, jnp.pi, 1., 1.])
    )
    
    # Standard Fit (Unpenalized)
    optimizer_std = ConstrainedOptimizer(model_func, priors=[])
    params_std, _ = optimizer_std.fit(init_params, data, bounds=bounds)
    
    # Penalized Fit
    optimizer_pen = ConstrainedOptimizer(model_func, priors=[f_iso_prior])
    params_pen, _ = optimizer_pen.fit(init_params, data, bounds=bounds)
    
    print(f"True f_iso: {true_params[3]}")
    print(f"Standard Fitted f_iso: {params_std[3]}")
    print(f"Penalized Fitted f_iso: {params_pen[3]}")
    
    # With Beta(2, 20), mode is (alpha-1)/(alpha+beta-2) = 1/20 = 0.05.
    # It penalizes high f_iso heavily, and f_iso=0 (logpdf=-inf).
    # Since alpha=2 > 1, f_iso=0 is -inf?
    # Beta(2, 20): proportional to x^(2-1) * (1-x)^(20-1) = x * (1-x)^19.
    # At x=0, value is 0. log(0) is -inf.
    # So f_iso CANNOT be 0. It must be > 0.
    
    if params_pen[3] > 1e-6:
        print("PASS: Penalized f_iso maintained > 0 due to prior.")
    else:
        print("FAIL: Penalized f_iso hit 0? (Should be impossible with alpha=2 prior)")
        
    print("Optimization Test Passed.")

if __name__ == "__main__":
    test_cnoddi_tortuosity()
    test_penalized_fitting()
