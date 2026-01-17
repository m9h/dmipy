import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.composer import compose_models
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition
import time

def test_composition():
    print("Testing Model Composition (Ball + Stick)...")
    
    # 1. Define Acquisition Scheme
    # Synthetic: 2 shells (b=1000, 3000), 30 dirs each + b0
    bvals = jnp.concatenate([jnp.zeros(1), jnp.ones(30)*1000, jnp.ones(30)*3000])
    # Random gradients (normalized)
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(61, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bvecs = jnp.array(vecs)
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # 2. Instantiate Models
    ball = G1Ball()
    stick = C1Stick()
    
    # 3. Compose
    # Model: f_stick * Stick + f_ball * Ball
    # Note: composer.py sums: f1*M1 + f2*M2
    # Params vector: [Stick_Params, Ball_Params, f_stick, f_ball]
    # Stick Props: mu (2), lambda_par (1) -> 3
    # Ball Props: lambda_iso (1) -> 1
    # Total Intrinsic: 4
    # Fractions: 2
    # Total Params: 6
    
    composite_model = compose_models([stick, ball])
    
    # 4. Define Parameters (Ground Truth)
    # Stick: along X-axis (theta=pi/2, phi=0), D_par=1.7e-9 m^2/s = 1.7e-3 mm^2/s
    # Ball: D_iso=3.0e-3 mm^2/s (CSF-like)
    # Fractions: 0.7 Stick, 0.3 Ball
    
    # Orient: theta=pi/2, phi=0
    mu_gt = jnp.array([jnp.pi/2, 0.0])
    lambda_par_gt = 1.7e-3
    lambda_iso_gt = 3.0e-3
    frac_stick = 0.7
    frac_ball = 0.3
    
    # Pack parameters
    # Order depends on model list order passed to compose_models: [stick, ball]
    # Stick Params: [mu_theta, mu_phi, lambda_par] -> 3
    # Ball Params: [lambda_iso] -> 1
    # Fractions: [f_stick, f_ball]
    
    params = jnp.concatenate([
        mu_gt, 
        jnp.array([lambda_par_gt]), 
        jnp.array([lambda_iso_gt]),
        jnp.array([frac_stick, frac_ball])
    ])
    
    print(f"Parameter Vector Shape: {params.shape}")
    print(f"Parameters: {params}")
    
    # 5. Run Forward Model
    print("Running Forward Model...")
    start = time.time()
    signal = composite_model(params, acq)
    signal.block_until_ready()
    duration = time.time() - start
    print(f"Execution Time (compile+run): {duration*1000:.2f} ms")
    
    # 6. Verify Values
    # Check b=0
    # Signal should be f_stick*1 + f_ball*1 = 1.0 (if sum is 1)
    # Here sum is 1.0.
    b0_idx = 0
    print(f"Signal at b=0: {signal[b0_idx]:.4f} (Expected: {frac_stick + frac_ball:.4f})")
    
    # Check High b-value (stick should persist, ball should decay)
    # Direction along X (index where bvec is close to X)
    # bvecs[0] is random, but let's just check mean signal decay logic
    print(f"Mean Signal: {jnp.mean(signal):.4f}")
    
    if jnp.abs(signal[b0_idx] - 1.0) < 1e-5:
        print("SUCCESS: Signal at b=0 is correct.")
    else:
        print("FAILURE: Signal at b=0 deviates.")

if __name__ == "__main__":
    test_composition()
