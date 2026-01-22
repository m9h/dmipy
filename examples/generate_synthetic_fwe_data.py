
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G1Ball

import os

def generate_random_bvecs(n_dirs, key):
    """Generate random unit vectors on the sphere."""
    k1, k2 = jax.random.split(key)
    z = jax.random.uniform(k1, (n_dirs,), minval=-1.0, maxval=1.0)
    theta = jax.random.uniform(k2, (n_dirs,), minval=0.0, maxval=2.0*jnp.pi)
    
    x = jnp.sqrt(1 - z**2) * jnp.cos(theta)
    y = jnp.sqrt(1 - z**2) * jnp.sin(theta)
    
    vecs = jnp.stack([x, y, z], axis=1)
    return vecs

def simulate_voxel_batch(key, n_samples, acquisition):
    """
    Simulate a batch of voxels with random parameters.
    """
    # 1. Sample Parameters
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    # Free Water Fraction: U[0, 1]
    # Bias slightly towards lower fractions? No, uniform coverage is best for training.
    f_iso = jax.random.uniform(k1, (n_samples,))
    f_tissue = 1.0 - f_iso
    
    # Tissue Orientation: Random Scheme
    # Sample random unit vectors
    orientations = generate_random_bvecs(n_samples, k2)
    # Convert to theta, phi for C1Stick (which likely expects mu vector or angles?)
    # C1Stick.call(..., mu=...) expects unit vector or angles? 
    # Checking my previous read of ball_stick.py:
    # "mu = jnp.array([theta, phi])" -> implicitly it took angles.
    # Let's check C1Stick source via assumption first or read it.
    # Wait, ball_stick.py line 45: mu = jnp.array([theta, phi])
    # But C1Stick likely supports vector input if designed well.
    # Let's look at ball_stick.py again... 
    # line 64: mu=mu. 
    # If C1Stick accepts angles, then I need angles.
    # Let's compute angles from vectors just to be safe.
    # But wait, ball_stick used [theta, phi].
    # Let's stick to vectors for now, check C1Stick if I can.
    # Actually, to be safe, I'll pass vectors if I see C1Stick accepts them.
    # I'll rely on generating angles to be safe as ball_stick.py suggests it uses angles.
    
    # Angles from Uniform Sphere
    # cos(theta) ~ U[-1, 1] is for standard spherical coords (theta is inclination).
    # dRmipy usually uses (theta, phi) where theta [0, pi], phi [0, 2pi].
    # Let's use `distrib_3d` if available or manual.
    
    # Manual:
    # u = U[0, 1]
    # v = U[0, 1]
    # theta = arccos(2u - 1)
    # phi = 2pi * v
    u = jax.random.uniform(k3, (n_samples,))
    v = jax.random.uniform(k4, (n_samples,))
    theta = jnp.arccos(2 * u - 1)
    phi = 2 * jnp.pi * v
    
    mu_angles = jnp.stack([theta, phi], axis=1) # (N, 2)
    
    # Tissue Diffusivity: N(1.7, 0.3) clipped
    D_tissue = 1.7e-9 + 0.3e-9 * jax.random.normal(k5, (n_samples,))
    D_tissue = jnp.clip(D_tissue, 0.5e-9, 3.0e-9)
    
    # Free Water Diffusivity: Fixed
    D_water = 3.0e-9
    
    # 2. Simulate
    stick = C1Stick()
    ball = G1Ball()
    
    # We need to vmap the models or pass batched inputs if supported.
    # Most dmipy-jax models support batched inputs if written purely in jax.numpy.
    
    # Helper to run simulation
    def sim_one(f_t, f_i, ang, d_t):
        # Stick
        # Note: ball_stick.py constructed mu per call.
        # We assume C1Stick.__call__ signature matches ball_stick.py usages:
        # call(bvals, gradient_directions, mu, lambda_par, ...)
        
        # Checking ball_stick.py:
        # S_stick = self.stick(
        #     bvals=acquisition.bvalues,
        #     gradient_directions=acquisition.gradient_directions,
        #     mu=mu,
        #     lambda_par=self.diffusivity
        # )
        
        S_stick = stick(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=ang,
            lambda_par=d_t
        )
        S_ball = ball(
            bvals=acquisition.bvalues,
            lambda_iso=D_water
        )
        return f_t * S_stick + f_i * S_ball

    # Batch simulation
    # vmap over samples
    sim_batch = jax.vmap(sim_one)(f_tissue, f_iso, mu_angles, D_tissue)
    
    return sim_batch, f_iso

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_fwe.npz", help="Output file")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--bval", type=float, default=1000.0, help="b-value for single shell")
    parser.add_argument("--n_dirs", type=int, default=32, help="Number of directions")
    args = parser.parse_args()
    
    # 1. Setup Acquisition
    # Bvals: 0 (1 b0) + N_dirs * bval
    bvals = jnp.concatenate([jnp.array([0.0]), jnp.full(args.n_dirs, args.bval)])
    bvals = bvals * 1e6 # Convert to s/m^2 if input is s/mm^2? 
    # Standard is s/mm^2 = 1000. SI is 1000 * 1e6.
    # Wait, usually input is 1000 s/mm^2. Python libraries vary.
    # Dmipy (original) used SI. Dmipy-jax uses SI (m^2/s for Diffusivity implies s/m^2 for bval).
    # 1 s/mm^2 = 1e6 s/m^2.
    # D_tissue is 1e-9 m^2/s.
    # b * D ~ 1000e6 * 1e-9 = 1000e-3 = 1.0. Correct range.
    
    # Bvecs
    rng = jax.random.PRNGKey(42)
    rng, k_bvecs = jax.random.split(rng)
    
    # Random directions for acquisition
    bvecs_shell = generate_random_bvecs(args.n_dirs, k_bvecs)
    bvecs_b0 = jnp.array([[0.0, 0.0, 0.0]]) # Or just [1,0,0], doesn't matter for b=0
    bvecs = jnp.concatenate([bvecs_b0, bvecs_shell], axis=0)
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # 2. Simulate Data
    print(f"Simulating {args.n_samples} samples with b={args.bval} s/mm^2...")
    rng, k_sim = jax.random.split(rng)
    
    # Generate clean signal
    batched_sim = jax.jit(lambda k: simulate_voxel_batch(k, args.n_samples, acq))
    signals, f_iso_gt = batched_sim(k_sim)
    
    # 3. Add Noise
    # Rician Noise
    # SNR = 30 regarding b0
    # sigma = 1.0 / SNR (assuming S0=1)
    # Signals are naturally normalized to S0=1 in these models?
    # Actually ball/stick usually return attenuation E(q). E(0)=1.
    sigma = 1.0 / 30.0
    
    rng, k_noise1, k_noise2 = jax.random.split(rng, 3)
    n1 = jax.random.normal(k_noise1, signals.shape) * sigma
    n2 = jax.random.normal(k_noise2, signals.shape) * sigma
    
    noisy_signals = jnp.sqrt((signals + n1)**2 + n2**2)
    
    # 4. Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Save as numpy arrays
    np.savez(args.out, 
             signals=np.array(noisy_signals),
             f_iso=np.array(f_iso_gt),
             bvals=np.array(bvals),
             bvecs=np.array(bvecs))
    
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
