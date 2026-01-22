
import numpy as np
import scipy.special as ssp
import time
import argparse

def get_connectome_scheme():
    """
    Returns b-values and gradient directions for a Connectome 2.0 style acquisition.
    4 Shells: b=1000, 3000, 5000, 10000 s/mm^2
    50 directions per shell -> 200 measurements.
    """
    shells = [1000, 3000, 5000, 10000]
    n_dirs_per_shell = 50
    
    bvals = []
    bvecs = []
    
    # Random directions on sphere (fibonacci lattice or similar is better, but random is fine for oracle)
    # Using a deterministic seed for reproducibility of the *scheme*
    rng = np.random.default_rng(2024)
    
    for b in shells:
        # Generate random points on unit sphere
        indices = np.arange(0, n_dirs_per_shell, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_dirs_per_shell)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        dirs = np.stack([x, y, z], axis=1)
        
        bvals.append(np.full(n_dirs_per_shell, b))
        bvecs.append(dirs)
        
    return np.concatenate(bvals), np.concatenate(bvecs)

def precompute_roots(n_roots, n_functions):
    """
    Precompute roots of J'_n(x) = 0.
    alpha[k, n] is the k-th root of J'_n(x).
    """
    alpha = np.zeros((n_roots, n_functions))
    # m=0 roots: J'_0(x) = -J_1(x) = 0 -> roots of J1
    # Note: scipy.special.jnp_zeros(n, k) returns k-th root of J'_n.
    
    # n=0
    # jnp_zeros(0, k) gives roots of J'_0 i.e. J_1
    alpha[:, 0] = ssp.jnp_zeros(0, n_roots)
    
    # n > 0
    for n in range(1, n_functions):
        alpha[:, n] = ssp.jnp_zeros(n, n_roots)
        
    return alpha

def callaghan_cylinder_signal(bvals, bvecs, mu, lambda_par, diameter, D_perp, tau, alpha):
    """
    Calculate signal for finite cylinder using Callaghan approximation.
    Vectorized over measurements. Assumes single parameter set (mu, lambda, diam).
    """
    # Stick (Parallel)
    dot = np.dot(bvecs, mu)
    S_par = np.exp(-bvals * lambda_par * dot**2)
    
    # Perpendicular
    # q = 1/2pi * sqrt(b/tau) (assuming narrow pulse for q-definition consistency)
    q_mag = np.sqrt(bvals / tau) / (2 * np.pi) 
    q_mag = q_mag * 1e3 # mm^-1 -> m^-1
    
    sin_sq = 1 - dot**2
    sin_sq = np.clip(sin_sq, 0, 1)
    q_perp = q_mag * np.sqrt(sin_sq)
    
    R = diameter / 2
    x = 2 * np.pi * q_perp * R
    
    # Summation
    n_roots, n_funcs = alpha.shape
    
    S_perp = np.zeros_like(x)
    
    # m = 0
    # alpha_0: (K,)
    alpha_0 = alpha[:, 0] 
    
    # For broadcasting: x is (N,), alpha is (K,)
    # We want sum over K.
    
    # Term: 4 * exp(-alpha^2 D tau / R^2) * x^2 / (x^2 - alpha^2)^2 * (J_1(x))^2
    J1_x = ssp.j1(x) # (N,)
    
    # Mask x ~ 0 separately? 
    # If x is small, J1(x) ~ x/2, J1(x)^2 ~ x^2 / 4
    # Term -> 0 if alpha != 0.
    
    # Expand dims
    x_col = x[:, None] # (N, 1)
    alpha_0_row = alpha_0[None, :] # (1, K)
    
    # Avoid singularity
    denom = (x_col**2 - alpha_0_row**2)**2
    denom = np.maximum(denom, 1e-12)
    
    exp_factor = np.exp(-alpha_0_row**2 * D_perp * tau / R**2) # (1, K)
    
    term_0 = 4 * exp_factor * x_col**2 / denom # (N, K)
    sum_0 = np.sum(term_0, axis=1) # (N,)
    
    S_perp += sum_0 * (J1_x**2)
    
    # m > 0
    for m in range(1, n_funcs):
        alpha_m = alpha[:, m] # (K,)
        alpha_m_row = alpha_m[None, :]
        
        # J'_m(x) = 0.5 * (J(m-1) - J(m+1))
        J_prime = 0.5 * (ssp.jv(m-1, x) - ssp.jv(m+1, x)) # (N,)
        
        exp_factor_m = np.exp(-alpha_m_row**2 * D_perp * tau / R**2)
        
        denom_m = (x_col**2 - alpha_m_row**2)**2
        denom_m = np.maximum(denom_m, 1e-12)
        
        # Coeff: alpha^2 / (alpha^2 - m^2)
        coeff = alpha_m_row**2 / (alpha_m_row**2 - m**2)
        
        term_m = 8 * exp_factor_m * coeff * (x_col * J_prime[:, None])**2 / denom_m
        S_perp += np.sum(term_m, axis=1)
        
    # Handle q_perp -> 0 limit (Signal -> 1)
    S_perp = np.where(q_perp < 1e-6, 1.0, S_perp)
    
    return S_par * S_perp

def generate_data(n_samples=100000):
    print(f"Generating {n_samples} samples using Connectome 2.0 protocol...")
    
    bvals, bvecs = get_connectome_scheme()
    n_meas = len(bvals)
    print(f"Protocol: {n_meas} measurements. Max b-value: {np.max(bvals)}")
    
    # Parameters
    # tau (diffusion time): Delta - delta/3. Assume Delta=20ms, delta=8ms -> tau = 17.33 ms
    # These are typical high-gradient settings.
    delta = 0.008 # s
    Delta = 0.020 # s
    tau = Delta - delta/3.0
    
    # Model Constants
    D_0 = 2.0e-3 # mm^2/s (Ball diffusivity)
    D_intra_par = 2.0e-3 # mm^2/s (Intra-axonal parallel)
    D_intra_perp = 1.7e-3 # mm^2/s (Intra-axonal perp? Usually intrinsic same) - Let's use 1.7e-3? Or typically same as parallel if intrinsic.
    # Let's match typical validation: 1.7e-3 for everything
    D_intra_par = 1.7e-3
    D_intra_perp = 1.7e-3 
    D_ball = 1.7e-3
    
    # D_intra_perp is in mm^2/s. Convert to m^2/s for the exponent (since R is in m, tau in s).
    D_intra_perp_SI = D_intra_perp * 1e-6
    
    # Precompute roots
    print("Precomputing Bessel roots...")
    n_roots = 20
    n_funcs = 20 # 50 is better for convergence but slower. 20 is usually okay.
    alpha_roots = precompute_roots(n_roots, n_funcs)
    
    # Parameter Distributions
    rng = np.random.default_rng(42)
    
    # 1. Volume Fractions
    f_intra = rng.uniform(0.3, 0.8, n_samples)
    f_ball = 1.0 - f_intra
    
    # 2. Orientations (Uniform on sphere)
    u = rng.uniform(0, 1, n_samples)
    v = rng.uniform(0, 1, n_samples)
    phi = 2 * np.pi * u
    theta = np.arccos(2 * v - 1)
    
    mu_x = np.cos(phi) * np.sin(theta)
    mu_y = np.sin(phi) * np.sin(theta)
    mu_z = np.cos(theta)
    mus = np.stack([mu_x, mu_y, mu_z], axis=1) # (N, 3)
    
    # 3. Diameters (The key parameter!)
    # Log-normal or Uniform? Uniform 1-10 um is good for benchmarking.
    diameters = rng.uniform(1.0e-6, 10.0e-6, n_samples) # meters
    
    signals = np.zeros((n_samples, n_meas))
    
    print("Starting simulation loop...")
    start_time = time.time()
    
    # Batch processing to print progress
    batch_size = 1000
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_n = end - i
        
        # Batch params
        batch_mu = mus[i:end]
        batch_diam = diameters[i:end]
        batch_f = f_intra[i:end]
        
        # We need to simulate per-fiber because orientation changes per sample.
        # Vectorized implementation of callaghan_cylinder_signal assumes (N_meas,) output for ONE fiber.
        # Can we vectorise over samples too?
        # bvecs: (M, 3). mu: (B, 3). dot: (B, M).
        
        # Let's reimplement callaghan_cylinder_signal core for (B, M) broadcasting
        # To avoid massive memory, let's just loop over batch or vectorize carefully.
        # Scipy functions broadcast well.
        
        # Dot products: (B, 3) @ (3, M) -> (B, M) ? No, bvecs is (M, 3)
        # batch_mu: (B, 3). bvecs.T: (3, M). -> (B, M)
        dot = batch_mu @ bvecs.T
        
        # Stick Signal (B, M)
        S_par = np.exp(-bvals * D_intra_par * dot**2)
        
        # Ball Signal (Isotropic) (M,) -> (B, M)
        S_ball = np.exp(-bvals * D_ball)
        S_ball = np.tile(S_ball, (batch_n, 1))
        
        # Perpendicular Signal (Complex part)
        # q_mag: (M,)
        q_mag = np.sqrt(bvals / tau) / (2 * np.pi) * 1e3
        
        # sin_theta: (B, M)
        sin_sq = 1 - dot**2
        sin_sq = np.clip(sin_sq, 0, 1)
        q_perp = q_mag * np.sqrt(sin_sq) # (B, M)
        
        # R: (B, 1)
        R = batch_diam[:, None] / 2
        
        # x: (B, M)
        x = 2 * np.pi * q_perp * R
        
        S_perp = np.zeros_like(x)
        
        # --- Summation ---
        # alpha_0: (K,)
        # Reshape for (B, M, K)
        # This will be huge: 1000 * 200 * 20 * 8 bytes ~ 32 MB. Fine.
        
        x_expanded = x[..., None] # (B, M, 1)
        
        # m=0
        alpha_0 = alpha_roots[:, 0]
        alpha_0_row = alpha_0[None, None, :]
        
        denom = (x_expanded**2 - alpha_0_row**2)**2
        denom = np.maximum(denom, 1e-12)
        
        exp_factor = np.exp(-alpha_0_row**2 * D_intra_perp_SI * tau / R[..., None]**2) # (B, 1, K)
        
        term = 4 * exp_factor * x_expanded**2 / denom
        sum_0 = np.sum(term, axis=2) # (B, M)
        
        S_perp += sum_0 * (ssp.j1(x)**2)
        
        for m in range(1, n_funcs):
            alpha_m = alpha_roots[:, m]
            alpha_m_row = alpha_m[None, None, :]
            
            J_prime = 0.5 * (ssp.jv(m-1, x) - ssp.jv(m+1, x)) # (B, M)
            
            exp_factor = np.exp(-alpha_m_row**2 * D_intra_perp_SI * tau / R[..., None]**2)
            denom = (x_expanded**2 - alpha_m_row**2)**2
            denom = np.maximum(denom, 1e-12)
            coeff = alpha_m_row**2 / (alpha_m_row**2 - m**2)
            
            term = 8 * exp_factor * coeff * (x_expanded * J_prime[..., None])**2 / denom
            S_perp += np.sum(term, axis=2)
            
        S_perp = np.where(q_perp < 1e-6, 1.0, S_perp)
        
        S_tot = batch_f[:, None] * (S_par * S_perp) + (1 - batch_f[:, None]) * S_ball
        
        # Add Rician Noise
        # SNR = 30
        # sigma = 1 / SNR (if S0=1)
        sigma = 1.0 / 30.0
        n1 = rng.normal(0, sigma, S_tot.shape)
        n2 = rng.normal(0, sigma, S_tot.shape)
        S_noisy = np.sqrt((S_tot + n1)**2 + n2**2)
        
        signals[i:end] = S_noisy
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {end}/{n_samples}...")

    elapsed = time.time() - start_time
    print(f"Simulation complete in {elapsed:.2f}s ({n_samples/elapsed:.1f} samples/s)")
    
    # Save
    out_path = '/data/connectome_oracle_100k.npz'
    np.savez_compressed(out_path, 
                        signals=signals, 
                        bvals=bvals, 
                        bvecs=bvecs,
                        mu=mus,
                        diameter=diameters,
                        f_intra=f_intra)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    generate_data()
