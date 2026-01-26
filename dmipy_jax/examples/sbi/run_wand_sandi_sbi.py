import os
import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from dipy.io.image import save_nifti
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.io.wand import WANDLoader
from dmipy_jax.signal_models.sandi import get_sandi_model
from dmipy_jax.core.invariants import compute_invariants
from dmipy_jax.inference.trainer import create_trainer, train_loop
from dmipy_jax.acquisition import JaxAcquisition

def get_shell_indices(bvals, b_tol=50):
    """
    Identifies indices for each b-value shell.
    """
    # Round bvals to nearest 100 for grouping
    b_rounded = jnp.round(bvals / 100) * 100
    unique_b = jnp.unique(b_rounded, size=10, fill_value=-1) # Assumes max 10 shells
    unique_b = unique_b[unique_b >= 0]
    
    indices = []
    shell_bvals = []
    
    for b in unique_b:
        # Ignore b0
        if b < b_tol:
            continue
        idx = jnp.abs(bvals - b) < b_tol
        indices.append(idx)
        shell_bvals.append(b)
        
    return shell_bvals, indices

@jax.jit
def compute_multishell_invariants(signal, bvecs, shell_masks, max_order=6):
    """
    Computes rotational invariants for each shell and concatenates them.
    
    Args:
        signal: (N_features,) or (Batch, N_features)
        bvecs: (N_features, 3)
        shell_masks: List of boolean masks for each shell.
        
    Returns:
        invariants: (Batch, N_shells * (L/2 + 1))
    """
    # Handle single sample vs batch
    is_batch = signal.ndim > 1
    if not is_batch:
        signal = signal[None, :]
        
    batch_size = signal.shape[0]
    all_invs = []
    
    for mask in shell_masks:
        # Extract shell data
        # Mask is (N_features,)
        # We need to slice bvecs and signal
        # Since JIT requires static shapes, standard boolean masking might be tricky if sizes vary.
        # But here shell_masks are passed as *arguments*.
        # For JIT, we might need to assume fixed sizes or use where with padding?
        # Actually, compute_invariants uses Least Squares which handles arbitrary N_dirs.
        # But constructing the basis B depends on bvecs.
        
        # To make this fully JIT-able with static shell sizes:
        # We assume the shell structure is constant (which it is for a given acquisition).
        # We can implement this by passing pre-sliced arrays if we move this logic outside?
        # Or we can use `jnp.take` with fixed indices if we calculate them once.
        pass
    
    # Re-impl for JIT safety: expected inputs are pre-sliced shell data?
    # No, let's keep it simple. We will process per-shell outside of JIT or assume fixed indices.
    # The 'shell_masks' argument is tricky for JIT if variable length.
    
    # BETTER APPROACH:
    # 1. We pre-compute the Basis matrices for each shell once.
    # 2. We pass the Basis matrices to a function that just does dot products.
    return jnp.zeros((batch_size, 1)) # Placeholder logic overridden below

def make_simulator(acq_scheme, basis_matrices, shell_indices):
    """
    Creates the simulator function for the SBITrainer.
    """
    sandi_model = get_sandi_model()
    
    # Constants
    D_long_mean = 3.0e-9
    R_min, R_max = 2e-6, 15e-6
    
    @jax.jit
    def simulator(key, theta):
        """
        Args:
            key: PRNGKey
            theta: (Batch, 5) -> [f_stick, f_sphere, f_ball, R_sphere, D_long_offset]
                   We model fractions on simplex? Or just raw params re-normalized?
                   Let's use raw params in range [0, 1] and normalize.
        """
        # Theta unconstrained:
        # We transform them to valid ranges.
        
        # Split key is handled by vmap if mapped? 
        # Trainer passes (Batch, Theta) so we vmap internally? 
        # No, Trainer simulator expects (key, theta) -> signal.
        # Wait, Trainer calls `simulator(k2, theta)`.
        # If theta is batched, we need vmap.
        
        def single_sim(p):
            # 1. Transform Parameters
            # Input p: 5 params from Normal(0, 1) or Uniform(0, 1)?
            # Let's assume Prior returns Uniform(0, 1).
            
            # Fractions
            # We take first 3 as logits for softmax? Or Dirichlet?
            # Simpler: f_stick, f_sphere, f_ball raw [0, 1]. Normalize.
            
            f_raw = p[0:3]
            f_norm = f_raw / (jnp.sum(f_raw) + 1e-6)
            # But we also have f_zeppelin = 1 - sum. 
            # SANDI has 4 comps.
            # Let's parameterize 4 fractions?
            # Or 3 fractions + remainder.
            # Let's normalize p[0:3] + (1 - sum)? 
            # Let's use softmax on 4 logits?
            # But theta dim is limited.
            
            # Strategy: 3 inputs for fractions.
            # f_stick = p[0]
            # f_sphere = p[1]
            # f_ball = p[2]
            # normalized = softmax([p0, p1, p2, 0.5])?
            
            # Let's assume input p is already valid from prior sampler?
            # Yes, let's shape the PriorSampler to return valid ranges.
            f_stick = p[0]
            f_sphere = p[1]
            f_ball = p[2]
            f_zeppelin = 1.0 - (f_stick + f_sphere + f_ball)
            f_zeppelin = jnp.clip(f_zeppelin, 0.0, 1.0) # Safety
            
            # Re-normalize if sum > 1 (heuristic)
            total = f_stick + f_sphere + f_ball + f_zeppelin
            scale = 1.0 / total
            f_stick *= scale
            f_sphere *= scale
            f_ball *= scale
            f_zeppelin *= scale
            
            R_sphere = p[3] # in meters, e.g. 5e-6
            # D_long = p[4] # e.g. 3e-9
            D_long = 3.0e-9 # Fix D_long for stability as per paper? 
            # Or let it vary slightly.
            
            # Orientation: Randomly sampled?
            # Since we use Rotational Invariants, the orientation doesn't matter for the *features*!
            # But the *simulator* needs an orientation to generate the signal.
            # We can fix mu = z-axis, since Invariants should be identical?
            # YES. That's the power of invariants.
            # But technically, discrete sampling noise might affect it.
            # Let's use fixed Z-axis for speed.
            theta_or, phi_or = 0.0, 0.0
            
            params_sandi = jnp.array([
                theta_or, phi_or, 
                f_stick, f_sphere, f_ball, 
                R_sphere, 
                0.0 # lambda_perp for Zeppelin (stick-like)
            ])
            
            # Simulate
            S = sandi_model(params_sandi, acq_scheme)
            
            # Add Noise (Rician)
            # SNR ~ 30-50 for WAND?
            # Let's assume SNR=50. Sigma = 1/50 = 0.02
            # Noise is added in Trainer? Yes.
            # But we need Rician bias? Trainer adds Gaussian.
            # For invariants, we might want magnitude signal?
            # Let's return noise-free S here, Trainer adds noise.
            # Wait, Trainer simulator usually adds noise if it's part of the 'model'.
            # Trainer code: "If noise_std > 0: signal + noise".
            # Gaussian noise.
            # For dMRI, we often want Rician. magnitude(S + complex_noise).
            # We'll stick to Gaussian approx for simplicity or modify Trainer.
            # Let's rely on Trainer's Gaussian noise.
            
            # Compute Invariants
            # Loop over shells
            invs = []
            for i, idxs in enumerate(shell_indices):
                # S_shell: (N_dirs,)
                S_shell = S[idxs]
                B = basis_matrices[i]
                
                # Fit SH: C = S . B_pinv.T
                # Invariants logic inline
                # B is (N_dirs, N_coeffs)
                # We precomputed B_pinv? 
                # Let's use fit_sh_coefficients from core if accessible, or impl here.
                # B_pinv = jnp.linalg.pinv(B)
                coeffs = jnp.dot(S_shell, jnp.linalg.pinv(B).T)
                
                # Power Spectrum
                p_l = []
                start = 0
                max_l = 6
                for l in range(0, max_l + 1, 2):
                    n_m = 2 * l + 1
                    c_l = coeffs[start : start + n_m]
                    power = jnp.sum(c_l**2) / (2 * l + 1) # Mean energy? Or Sum?
                    # Paper Jallais 2022 says: "mean energy" or "norm"?
                    # Let's use Sum |c|^2.
                    p_l.append(jnp.sum(c_l**2))
                    start += n_m
                
                invs.append(jnp.array(p_l))
                
            return jnp.concatenate(invs)

        return jax.vmap(single_sim)(theta)
        
    return simulator

def prior_sampler(key, batch_size):
    """
    Samples parameters from prior.
    Theta: [f_stick, f_sphere, f_ball, R_sphere]
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Fractions: Dirichlet-like
    # Simple: Sample 3 uniform, normalize
    f = jax.random.uniform(k1, (batch_size, 4))
    f = f / jnp.sum(f, axis=1, keepdims=True)
    # Mapping: 0->stick, 1->sphere, 2->ball
    # We output first 3. Implicitly 4th is remainder.
    
    # Radius: Uniform [2e-6, 12e-6]
    r = jax.random.uniform(k2, (batch_size, 1), minval=2e-6, maxval=12e-6)
    
    return jnp.concatenate([f[:, 0:3], r], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--session", default="ses-02")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    
    print(f"--- Running WAND SANDI SBI (Subject: {args.subject} Session: {args.session}) ---")
    if args.test_mode:
        print("!!! TEST MODE ENABLED !!!")
        args.steps = 500
        
    # 1. Load Data
    loader = WANDLoader(subject=args.subject)
    loader.fetch_data() # Ensures structure
    
    # Load middle slice for memory efficiency if not defined?
    # Loader loads everything. 
    # For test mode, we might want to crop later.
    print("Loading data...")
    wand_dict = loader.load_axcaliber_data(session=args.session)
    
    # Extract
    data = wand_dict['data'] # (X, Y, Z, N)
    bvals = wand_dict['bvals']
    bvecs = wand_dict['bvecs']
    delta = wand_dict['small_delta'] # Array
    Delta = wand_dict['big_delta']   # Array
    affine = wand_dict['affine']
    
    print(f"Data Shape: {data.shape}")
    print(f"B-values: {jnp.unique(jnp.round(bvals/100)*100)}")
    
    # 2. Setup Acquisition & Invariants Basis
    # Identify shells
    # Shells usually: 0, 1000, 2000, ...
    shell_val, shell_idxs = get_shell_indices(bvals, b_tol=100)
    print(f"Found {len(shell_val)} shells: {shell_val}")
    
    # Create simulator acquisition object
    # We pass the full arrays.
    # Note: SANDI model needs 'delta' and 'Delta'. 
    # JaxAcquisition wrapper.
    # Check shape of delta/Delta. If they vary per volume, we need to pass full arrays.
    acq = JaxAcquisition(
        bvalues=bvals,
        gradient_directions=bvecs,
        delta=delta,
        Delta=Delta
    )
    
    # Precompute Basis Matrices for Invariants
    # Max Order L=6
    from dmipy_jax.core.invariants import sph_harm_basis
    basis_matrices = []
    
    # Filter shells to keep only non-b0 for invariants (usually)
    # Or keep all? b0 has no orientation info.
    # We typically skip b0 for SH fitting.
    valid_shells_idx = []
    
    for i, idxs in enumerate(shell_idxs):
        b = shell_val[i]
        if b < 100: # Skip b0
            continue
            
        print(f"Processing Shell b={b}...")
        # Get bvecs for this shell
        # idxs is boolean mask
        # We need numpy/jax indices
        idx_array = jnp.where(idxs)[0]
        shell_bvecs = bvecs[idx_array]
        
        B = sph_harm_basis(shell_bvecs, max_order=6)
        basis_matrices.append(B)
        valid_shells_idx.append(idx_array)
        
    print(f"Computed Basis for {len(basis_matrices)} shells.")
    
    # 3. Initialize Trainer
    print("Initializing SBI Trainer...")
    simulator_fn = make_simulator(acq, basis_matrices, valid_shells_idx)
    
    # Dimensions
    # Theta: 4 (f_stick, f_sphere, f_ball, R)
    # Signal: (L/2 + 1) * N_shells = 4 * N_shells
    theta_dim = 4
    signal_dim = 4 * len(basis_matrices)
    
    key = jax.random.key(42)
    trainer = create_trainer(
        key, theta_dim, signal_dim, 
        simulator_fn, prior_sampler,
        hidden_dim=64, num_layers=4
    )
    
    # 4. Train
    print("Training Flow...")
    trainer = train_loop(
        trainer, key, 
        num_steps=args.steps, 
        batch_size=256, 
        noise_std=0.05, # Add some noise to observation
        print_every=100
    )
    
    # 5. Inference Step
    print("Running Inference on Data...")
    
    # Flatten Data to List of Voxels
    # data: (X, Y, Z, N)
    # Masking: simple non-zero check on mean b0?
    b0_mask = bvals < 100
    S0 = jnp.mean(data[..., b0_mask], axis=-1)
    mask = S0 > (0.1 * jnp.max(S0)) # Brain mask
    
    if args.test_mode:
        # Crop to small ROI
        nx, ny, nz = data.shape[:3]
        cx, cy, cz = nx//2, ny//2, nz//2
        mask = jnp.zeros_like(mask)
        mask = mask.at[cx-5:cx+5, cy-5:cy+5, cz].set(True) # 10x10 single slice
        print("Test Mode: Using 10x10 ROI.")
        
    voxels = jnp.where(mask) # Tuple of arrays
    n_voxels = len(voxels[0])
    print(f"Inference on {n_voxels} voxels.")
    
    # Prepare Data for Inference (Compute Invariants)
    # We need to process real data same as simulator
    # Loop over voxels? Slow.
    # Vectorized:
    # 1. Extract masked data: (N_vox, N_measurements)
    S_masked = data[mask]
    
    # 2. Compute Invariants
    # Separate shells
    invs_list = []
    
    # Using batches for memory?
    # If N_vox is large (1M), we need batching.
    # WAND is high res?
    # Let's chunk.
    chunk_size = 10000
    
    full_invariants = []
    
    print("Computing Invariants for Subject Data...")
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        S_chunk = S_masked[start:end]
        
        # Compute invariants per shell
        chunk_invs = []
        for i, idx_array in enumerate(valid_shells_idx): # The pre-computed indices
            # S_chunk_shell: (Batch, N_dirs_in_shell)
            # Use jnp.take or slice
            S_shell = S_chunk[:, idx_array]
            B = basis_matrices[i]
            
            # Fit SH
            # C = S . pinv(B).T
            coeffs = jnp.dot(S_shell, jnp.linalg.pinv(B).T)
            
            # Power
            p_l = []
            start_l = 0
            for l_deg in range(0, 7, 2): # 0, 2, 4, 6
                n_m = 2 * l_deg + 1
                c_l = coeffs[..., start_l : start_l + n_m]
                power = jnp.sum(c_l**2, axis=-1)
                p_l.append(power)
                start_l += n_m
            
            chunk_invs.append(jnp.stack(p_l, axis=-1))
            
        # Concat shells: (Batch, N_shells, 4) -> (Batch, N_shells*4)
        flat = jnp.concatenate(chunk_invs, axis=-1)
        full_invariants.append(flat)
        
    full_invariants = jnp.concatenate(full_invariants, axis=0) # (N_vox, Feat)
    
    # Normalizing Flow Inference
    # Sample Posterior: input invariants -> get samples of theta.
    # We want Mean Posterior.
    
    print("Sampling Posteriors...")
    
    # FlowJAX specific: flow.sample(key, condition=context)
    # Batched sampling
    
    @eqx.filter_jit
    def sample_batch(key, context):
        # Sample 50 points per voxel
        # FlowJAX sample expects context (Batch, ContextDim)
        # Returns (Batch, ThetaDim) ? 
        # Usually sample returns (Batch, ThetaDim). 
        # But we want N_samples per voxel.
        # We can vmap over context? 
        # flow.sample(key, sample_shape=(N_samples,), condition=context)
        # FlowJAX supports this.
        return trainer.flow.sample(key, sample_shape=(50,), condition=context)
        
    # Result arrays
    # Theta: [f_stick, f_sphere, f_ball, R]
    mean_maps = np.zeros((n_voxels, 4))
    std_maps = np.zeros((n_voxels, 4))
    
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        context = full_invariants[start:end]
        
        # We need vmap over batch dimension? Or does sample handle it?
        # FlowJAX condition handling:
        # If context has batch dim, sample usually broadcasts or requires vmap.
        # Let's try vmap.
        
        batch_key = jax.random.split(key, end - start)
        # vmap sample_batch: context (Dim,) -> (50, Dim)
        
        # Redefine sample single
        def sample_single(k, c):
             return trainer.flow.sample(k, sample_shape=(50,), condition=c)
             
        # samples: (Batch, 50, 4)
        samples = jax.vmap(sample_single)(batch_key, context)
        
        # Compute stats
        mean_maps[start:end] = jnp.mean(samples, axis=1)
        std_maps[start:end] = jnp.std(samples, axis=1)
        
        if start % (chunk_size * 5) == 0:
            print(f"Processed {start}/{n_voxels} voxels...")
            
    # 6. Reconstruct Volumes
    print("Reconstructing Volumes...")
    # Maps of shape (X, Y, Z)
    vol_shape = data.shape[:3]
    
    # f_sphere map (Index 1)
    f_sphere_vol = np.zeros(vol_shape)
    f_sphere_vol[voxels] = mean_maps[:, 1]
    
    # R_sphere map (Index 3)
    r_sphere_vol = np.zeros(vol_shape)
    r_sphere_vol[voxels] = mean_maps[:, 3]
    
    # Save
    out_dir = Path("sbi_results")
    out_dir.mkdir(exist_ok=True)
    
    save_nifti(str(out_dir / f"{args.subject}_sandi_fsphere.nii.gz"), f_sphere_vol, affine)
    save_nifti(str(out_dir / f"{args.subject}_sandi_radius.nii.gz"), r_sphere_vol, affine)
    
    print(f"Saved results to {out_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
