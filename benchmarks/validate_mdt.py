import numpy as np
import jax.numpy as jnp
import jax
import os
import subprocess
import matplotlib.pyplot as plt
from dmipy_jax.signal_models import gaussian_models, cylinder_models, sphere_models
from dmipy_jax import constants

def get_bvals_bvecs():
    # Define a standard protocol
    # b-values: 0 (5), 1000 (30) s/mm^2 -> 0, 1e9 SI
    # MDT Tensor model restricts to b <= 1200 or similar
    bvals_smm2 = np.array([0] * 5 + [1000] * 30)
    bvals_si = bvals_smm2 * 1e6
    
    # Generate random bvecs on sphere
    n_dirs = len(bvals_si)
    theta = np.random.uniform(0, np.pi, n_dirs)
    phi = np.random.uniform(0, 2*np.pi, n_dirs)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    bvecs = np.stack([x, y, z], axis=1)
    
    return bvals_si, bvecs

def generate_ballstick_params(n_samples=1000):
    # Diffusivity: 0.1 to 3.0 um^2/ms -> 0.1e-9 to 3.0e-9 m^2/s
    d_vals = np.linspace(0.1e-9, 3.0e-9, n_samples)
    f_vals = np.linspace(0, 1, n_samples)
    
    # Grid search or Random? Let's do a meshgrid-like coverage or just random lines?
    # Let's do random uniform sampling to cover the space
    d_perp_min, d_perp_max = 0.0, 0.0 # Stick has 0 perp
    
    d_par = np.random.uniform(0.5e-9, 3.0e-9, n_samples)
    d_iso = np.random.uniform(0.5e-9, 3.0e-9, n_samples)
    f_stick = np.random.uniform(0, 1, n_samples)
    
    # Fix orientation for simplicity or vary it?
    # Let's vary it
    theta = np.random.uniform(0, np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    mu = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=1)
    
    # Param dictionary for dmipy-jax and MDT (names must match MDT expectations or be mapped)
    # MDT BallStick: Ball (d_iso) + Stick (d_par, theta, phi)
    # Actually MDT usually uses 'Tensor' for Stick (d_perp=0).
    # Standard MDT 'BallStick' might not exist as a single primitive, usually composite.
    # But our adapter handles 'Ball' and 'Stick' separate or checks mappings.
    # Let's check mdt_adapter.py again for 'Stick'.
    # Adapter says: if name == 'Stick': return mdt.get_model('Tensor')() (implicit d_perp=0?)
    # Wait, the adapter instantiates 'Tensor' but doesn't force d_perp=0 inside get_mdt_model. 
    # The PARAMETERS passed must enforce it.
    
    # Parameters for MDT 'Stick' (Tensor): lambda_1 (d_par), lambda_2=0, lambda_3=0, alpha, beta, gamma
    # Parameters for MDT 'Ball': d (d_iso)
    
    # However we want a composite model 'BallStick'.
    # If we want to validate components, we can validate Ball and Stick separately.
    # Or we can validate the sum.
    # Let's start with validating the components separately as mixing is trivial summation.
    
    return {
        'd_par': d_par,
        'd_iso': d_iso,
        'mu': mu,
        'f_stick': f_stick
    }

def run_dmipy_jax_zeppelin(bvals, bvecs, params):
    # Zeppelin: Cylindrically symmetric tensor
    # Params: lambda_par, lambda_perp, mu
    d_par = jnp.array(params['d_par'])
    d_perp = jnp.array(params['d_perp'])
    mu = jnp.array(params['mu'])
    
    # Vectorize signal generation
    # g2_zeppelin signature: (bvals, bvecs, mu, lambda_par, lambda_perp)
    # We map over samples
    
    def sim_one(dp, dpr, m):
        return gaussian_models.g2_zeppelin(bvals, bvecs, m, dp, dpr)
    
    vmap_sim = jax.vmap(sim_one)
    signals = vmap_sim(d_par, d_perp, mu)
    return signals

def run_dmipy_jax_stick(bvals, bvecs, params):
    # Stick: Zeppelin with d_perp = 0
    d_par = jnp.array(params['d_par'])
    d_perp = jnp.zeros_like(d_par)
    mu = jnp.array(params['mu'])
    
    def sim_one(dp, dpr, m):
        return gaussian_models.g2_zeppelin(bvals, bvecs, m, dp, dpr)
    
    vmap_sim = jax.vmap(sim_one)
    signals = vmap_sim(d_par, d_perp, mu)
    return signals

def run_dmipy_jax_ball(bvals, bvecs, params):
    d_iso = jnp.array(params['d_iso'])
    
    def sim_one(di):
        return gaussian_models.g1_ball(bvals, bvecs, di)
    
    vmap_sim = jax.vmap(sim_one)
    signals = vmap_sim(d_iso)
    return signals

def validate_model(model_name, params_mdt, params_jax, runner_jax, bvals, bvecs, experiment_name=None):
    if experiment_name is None:
        experiment_name = model_name
    print(f"Validating {experiment_name}...")
    
    # 1. Prepare Data
    data_dir = os.path.abspath('benchmarks/data')
    os.makedirs(data_dir, exist_ok=True)
    input_path = os.path.join(data_dir, f'mdt_input_{experiment_name}.npz')
    output_path = os.path.join(data_dir, f'mdt_output_{experiment_name}.npz')
    
    # We save model_name for the docker script to know which model to load
    np.savez(input_path, model_name=model_name, bvals=bvals, bvecs=bvecs, params=params_mdt)
    
    # 2. Run MDT (Docker)
    # Ensure absolute paths for docker volume
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{data_dir}:/data',
        '-v', f'{os.getcwd()}/docker:/scripts',
        'mdt_oracle',
        'python3', '/scripts/run_mdt_simulation.py',
        '--input', f'/data/mdt_input_{experiment_name}.npz',
        '--output', f'/data/mdt_output_{experiment_name}.npz'
    ]
    
    print("Running MDT Oracle...")
    subprocess.run(cmd, check=True)
    
    # 3. Load MDT Results
    mdt_res = np.load(output_path)
    mdt_signals = mdt_res['signals']
    
    # 4. Run dmipy-jax
    print("Running dmipy-jax...")
    jax_signals = runner_jax(bvals, bvecs, params_jax)
    jax_signals = np.array(jax_signals)
    
    # 5. Compare
    residuals = jax_signals - mdt_signals
    mse = np.mean(residuals**2)
    max_diff = np.max(np.abs(residuals))
    
    print(f"{experiment_name} Results:")
    print(f"MSE: {mse:.8e}")
    print(f"Max Diff: {max_diff:.8e}")
    
    # 6. Plot
    res_dir = os.path.abspath('benchmarks/results')
    os.makedirs(res_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(mdt_signals.flatten(), jax_signals.flatten(), alpha=0.1, s=1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('MDT Signal')
    plt.ylabel('dmipy-jax Signal')
    plt.title(f'{experiment_name} Identity')
    
    plt.subplot(1, 2, 2)
    # Bland-Altman
    mean_sig = (mdt_signals + jax_signals) / 2
    diff_sig = jax_signals - mdt_signals
    plt.scatter(mean_sig.flatten(), diff_sig.flatten(), alpha=0.1, s=1)
    plt.xlabel('Mean Signal')
    plt.ylabel('Difference (JAX - MDT)')
    plt.title(f'{experiment_name} Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, f'mdt_validation_{experiment_name.lower()}.png'))
    plt.close()
    
    return max_diff < 1e-5 # Success criterion

def main():
    bvals, bvecs = get_bvals_bvecs()
    n_samples = 1000
    
    # --- Validate Zeppelin ---
    # MDT Zeppelin params: often uses Tensor representation
    # Tensor params: lambda_1, lambda_2, lambda_3, alpha, beta, gamma
    # We need to convert (mu, d_par, d_perp) to (alpha, beta, gamma, l1, l2, l3)
    
    d_par = np.random.uniform(0.1e-9, 3.0e-9, n_samples)
    d_perp = np.random.uniform(0.1e-9, 3.0e-9, n_samples)
    # Ensure d_par >= d_perp usually for Zeppelin, but not strictly required
    
    # Angles for MDT
    alpha = np.random.uniform(-np.pi, np.pi, n_samples)
    beta = np.random.uniform(0, np.pi, n_samples)
    gamma = np.zeros(n_samples) # symmetric
    
    # Convert alpha/beta to mu for JAX
    # mu = [sin(b)cos(a), sin(b)sin(a), cos(b)] (Physics convention usually)
    # Check JAX Tensor implementation for Euler convention: Z-Y-Z
    # R = Rz(alpha) Ry(beta) Rz(gamma)
    # e1 (primary) = R . [0,0,1]? No.
    # In gaussian_models.py Tensor: 
    # e1 = [ca*cb*cg - sa*sg, ...] which corresponds to column 1 of R. 
    # Usually e1 is the x-axis rotated? 
    # Let's double check standard DTI eigenbasis.
    # Usually lambda_1 corresponds to e1.
    # If we want e1 to align with fiber direction defined by alpha, beta:
    # We should ensure `mu` computed for JAX matches `e1` computed by MDT from alpha,beta.
    
    # Simplification: Use aligned Z-axis for functional test or standard conversion.
    # Let's calculate e1 from alpha, beta, gamma=0 using the formula in gaussian_models.py
    # and pass that as mu.
    
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    
    # MDT Tensor uses Spherical Coordinates (theta, phi) for the principal direction (e1 or e3?)
    # Usually MDT defines principal axis by theta, phi.
    # theta: inclination (0..pi)
    # phi: azimuth (0..2pi)
    # mu = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
    
    st = np.sin(beta) # beta maps to Tensor.theta
    ct = np.cos(beta)
    cp = np.cos(alpha) # alpha maps to Tensor.phi
    sp = np.sin(alpha)
    
    ex = st * cp
    ey = st * sp
    ez = ct
    mu = np.stack([ex, ey, ez], axis=1)
    
    mdt_zeppelin_params = {
        'S0.s0': np.ones(n_samples),
        'Tensor.d': d_par,
        'Tensor.dperp0': d_perp,
        'Tensor.dperp1': d_perp,
        'Tensor.phi': alpha,
        'Tensor.theta': beta,
        'Tensor.psi': gamma
    }
    
    jax_zeppelin_params = {
        'd_par': d_par,
        'd_perp': d_perp,
        'mu': mu
    }
    
    # Note: MDT Zeppelin might be called 'Zeppelin' or 'Tensor'.
    # Our adapter maps 'Zeppelin' -> 'Tensor'.
    # So we pass 'Zeppelin' as model name.
    
    validate_model('Zeppelin', mdt_zeppelin_params, jax_zeppelin_params, run_dmipy_jax_zeppelin, bvals, bvecs)

    # --- Validate Ball ---
    d_iso = np.random.uniform(0.1e-9, 3.0e-9, n_samples)
    
    mdt_ball_params = {
        'S0.s0': np.ones(n_samples),
        'Tensor.d': d_iso,
        'Tensor.dperp0': d_iso,
        'Tensor.dperp1': d_iso,
        'Tensor.phi': np.zeros(n_samples),
        'Tensor.theta': np.zeros(n_samples),
        'Tensor.psi': np.zeros(n_samples)
    }
    
    jax_ball_params = {'d_iso': d_iso}
    
    print("Validating Ball (via MDT Tensor)...")
    validate_model('Zeppelin', mdt_ball_params, jax_ball_params, run_dmipy_jax_ball, bvals, bvecs, experiment_name='Ball')
    

if __name__ == '__main__':
    main()
