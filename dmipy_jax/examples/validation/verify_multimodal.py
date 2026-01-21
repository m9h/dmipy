
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.core.multimodal import JointModel
from dmipy_jax.models.ball_stick import BallStick

def verify_multimodal():
    print("Verifying Multi-Modal Fusion (Joint T1/T2-Diffusion)...")
    
    # 1. Setup Acquisition with varying TE and TR
    # Case A: Vary TE (T2 decay), fixed TR, fixed b=0
    n_steps = 20
    bvals_A = jnp.zeros(n_steps)
    bvecs_A = jnp.zeros((n_steps, 3))
    bvecs_A = bvecs_A.at[:, 2].set(1.0) # z-dir
    
    TE_A = jnp.linspace(0.0, 0.2, n_steps) # 0 to 200ms
    TR_A = jnp.full(n_steps, 5.0) # 5s (long TR, T1 sat minimal)
    
    acq_A = acquisition_scheme_from_bvalues(bvals_A, bvecs_A, TE=TE_A, TR=TR_A)
    
    # Case B: Vary TR (T1 recovery), fixed TE, fixed b=0
    TR_B = jnp.linspace(0.1, 5.0, n_steps)
    TE_B = jnp.full(n_steps, 0.0) # Short TE
    bvals_B = jnp.zeros(n_steps)
    acq_B = acquisition_scheme_from_bvalues(bvals_B, bvecs_A, TE=TE_B, TR=TR_B)
    
    # 2. Instantiate Model
    # BallStick + Relaxation
    # We first need to wrap BallStick to make it compaitble? 
    # BallStick in dmipy_jax/models/ball_stick.py is a class, but we need an instance.
    # JointModel takes a model instance.
    
    ball_stick = BallStick()
    # Note: BallStick in current codebase might not inherit from CompartmentModel yet?
    # I updated C1Stick and G1Ball, but BallStick composes them.
    # Let's check if BallStick works as a "model" passed to JointModel.
    # JointModel expects .parameter_names, .parameter_cardinality, ranges, and __call__.
    # I should check BallStick implementation again. 
    # BallStick in ball_stick.py does NOT define parameter_names/cardinality explicitly in class attributes?
    # Wait, I checked ball_stick.py earlier. It didn't seem to inherit eqx.Module or define these?
    # Let me re-read ball_stick.py content shortly if it fails.
    # Assuming standard interface. If not, I might need to fix BallStick too or use C1Stick for test.
    
    # Let's use C1Stick for simplicity fow now as I know I updated it.
    from dmipy_jax.cylinder import C1Stick
    from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
    
    # JointModel is the "Kernel" equivalent here (a single compartment with complex physics)
    from dmipy_jax.core.multimodal import RelaxationModel
    
    # Instantiate models with default or specific values. 
    # C1Stick fields are default None.
    # RelaxationModel fields default None, but we can set defaults here if we want them to act as defaults.
    # However, in this test we pass parameters explicitly in the params dict.
    
    stick = C1Stick() 
    relax = RelaxationModel(t1=1.0, t2=0.05) # These values act as defaults if not in params
    
    joint_kernel = JointModel(diffusion_model=stick, relaxation_model=relax)
    
    # Wrap in JaxMultiCompartmentModel to handle parameter unpacking and acquisition interface
    model = JaxMultiCompartmentModel([joint_kernel])
    
    # 3. Predict
    # JaxMultiCompartmentModel will prefix params if collision, or use names as is.
    # JointModel params: ['mu', 'lambda_par', 't1', 't2']
    # JaxMultiCompartmentModel adds 'partial_volume_0'.
    
    params = {
        'mu': jnp.array([1.57, 0.0]), # theta=pi/2
        'lambda_par': 1.7e-9,
        't1': 1.0,
        't2': 0.05,
        'partial_volume_0': 1.0 # Pure compartment
    }
    
    # Check T2 Decay
    # Model call signature: (parameter_dictionary, acquisition)
    signal_A = model(params, acq_A)
    # Expected: exp(-TE/T2)
    expected_A = jnp.exp(-TE_A / 0.05)
    
    print(f"Mean Error A (T2): {jnp.mean(jnp.abs(signal_A - expected_A))}")
    if jnp.allclose(signal_A, expected_A, atol=1e-5):
        print("PASS: T2 Decay verified.")
    else:
        print("FAIL: T2 Decay mismatch.")
        print("Signal:", signal_A)
        print("Expected:", expected_A)

    # Check T1 Recovery
    params_B = params.copy()
    signal_B = model(params_B, acq_B)
    # Expected: (1 - exp(-TR/T1)) * exp(-TE/T2) -> exp(0)=1
    expected_B = (1.0 - jnp.exp(-TR_B / 1.0)) * 1.0
    
    print(f"Mean Error B (T1): {jnp.mean(jnp.abs(signal_B - expected_B))}")
    if jnp.allclose(signal_B, expected_B, atol=1e-5):
        print("PASS: T1 Recovery verified.")
    else:
        print("FAIL: T1 Recovery mismatch.")
    
    print("Verification Complete.")

if __name__ == "__main__":
    verify_multimodal()
