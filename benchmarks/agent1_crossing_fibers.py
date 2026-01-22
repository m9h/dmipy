
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import time
from dipy.sims.voxel import multi_tensor
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
try:
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
except ImportError:
    pass # DipY might be minimal install?

# 1. Setup Data Generation (DiPy)
def generate_crossing_phantom(angles_deg, bval=3000, snr=30):
    """
    Generates single-voxel signals for 2 straight fibers crossing at specific angles.
    """
    sphere = get_sphere(name='repulsion724')
    bvals = np.ones(len(sphere.vertices)) * bval
    bvals[0] = 0 # add b0? No, multi_tensor handles it.
    
    # Actually need explicit bvals/bvecs for reconstruction
    # Let's use a standard shell
    n_dir = 64
    theta = np.linspace(0, 2*np.pi, n_dir)
    bvecs = np.column_stack([np.sin(theta), np.cos(theta), np.zeros(n_dir)])
    # Add b0
    bvecs = np.vstack([np.array([0,0,0]), bvecs])
    bvals = np.concatenate([np.array([0]), np.ones(n_dir)*bval])
    gtab = gradient_table(bvals, bvecs)
    
    data_list = []
    
    for ang in angles_deg:
        # Fiber 1: along X
        # Fiber 2: rotated by ang in XY plane
        ang_rad = np.deg2rad(ang)
        params_stick = [0, 0, 0, 0] # S0, d, d, d? multi_tensor takes mevals
        # mevals: eigenvalues of tensor. [1.7e-3, 0.2e-3, 0.2e-3] roughly
        mevals = np.array([[1.7e-3, 0.2e-3, 0.2e-3], [1.7e-3, 0.2e-3, 0.2e-3]])
        
        # Rotations
        # R1 identity (along X? eigenvalues usually ordered [vals], need angles)
        # multi_tensor takes angles [(theta, phi), ...]
        # F1: along X stats (90, 0) ?
        dir1 = (90, 0) 
        dir2 = (90, ang)
        
        sig, labels = multi_tensor(gtab, mevals, S0=1.0, angles=[dir1, dir2],
                                   fractions=[50, 50], snr=snr)
        
        data_list.append((ang, sig[0], bvecs, bvals))
        
    return data_list, gtab

# 2. Neural CSD Model (JAX)
class NeuralCSD(eqx.Module):
    # MLP mapping Signal -> ODF amplitudes (SH coeffs? or just Peak directions?)
    # Let's map Signal -> 2 peak directions (theta1, phi1, theta2, phi2)
    # This is "Direct Peak Regression"
    mlp: eqx.nn.MLP
    
    def __init__(self, key, in_size):
        self.mlp = eqx.nn.MLP(in_size, 4, width_size=128, depth=4, key=key)
        
    def __call__(self, s):
        out = self.mlp(s)
        # return logits or angles?
        # Angles unconstrained
        return out

def train_neural_csd(train_data, n_epochs=500):
    # train_data: list of (angle, sig, vec, val)
    # We create a synthetic training set on the fly properly covering the sphere
    
    key = jax.random.PRNGKey(55)
    model = NeuralCSD(key, 65) # 64 grad + 1 b0
    opt = optax.adam(1e-3)
    state = opt.init(eqx.filter(model, eqx.is_array))
    
    # ... Training loop omitted for brevity in this Agent script
    # We will just verify it runs/loads
    return model

def run_agent1():
    print("=== Agent 1: Crossing Fiber Benchmark ===")
    angles = [30, 45, 60, 90]
    data, gtab = generate_crossing_phantom(angles)
    
    print(f"Generated {len(data)} Phantoms. Angles: {angles}")
    print("Competitor 1: DiPy CSD (Standard)")
    
    try:
        from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
        # CSD needs response function
        # We can fake it or estimate it
        # Assume response [1.7e-3, 0.2e-3, 0.2e-3]
        response = (np.array([1.7e-3, 0.2e-3, 0.2e-3]), 300.0) 
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        
        for ang, sig, _, _ in data:
            # DiPy CSD fit expects data or (x, y, z, N)
            # Try reshaping
            # sig shape (N,)
            sig_reshaped = sig.reshape(1, 1, 1, -1)
            fit = csd_model.fit(sig_reshaped)
            # peaks?
            from dipy.direction import peaks_from_model
            # Simple check
            print(f"Angle {ang}: CSD fit complete. SH coeffs shape {fit.shm_coeff.shape}")
            
    except Exception as e:
        print(f"DiPy CSD Failed: {e}")

    print("Competitor 2: Neural CSD (JAX)")
    model = train_neural_csd(None)
    print("Neural CSD Initialized.")
    
    print("WINNER: Comparison pending full metric implementation.")

if __name__ == "__main__":
    run_agent1()
