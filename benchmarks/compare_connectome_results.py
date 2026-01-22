
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def compare_results():
    print("Loading data...")
    try:
        oracle = np.load('data/connectome_oracle_pytorch.npz')
        jax_res = np.load('data/dmipy_jax_connectome.npz')
    except FileNotFoundError:
        print("Data not found.")
        return

    # Use clean signals for physics validation if available
    if 'signals_clean' in oracle:
        print("Using CLEAN Oracle signals for validation.")
        sig_oracle = oracle['signals_clean']
    else:
        sig_oracle = oracle['signals']

    sig_jax = jax_res['signals']   # (N, M)
    
    # Slice Oracle to match JAX
    n_jax = sig_jax.shape[0]
    if sig_oracle.shape[0] > n_jax:
        print(f"Slicing Oracle from {sig_oracle.shape[0]} to {n_jax} samples.")
        sig_oracle = sig_oracle[:n_jax]
    
    # Check shapes
    if sig_oracle.shape != sig_jax.shape:
        print(f"Shape mismatch! Oracle: {sig_oracle.shape}, Jax: {sig_jax.shape}")
        return

    print(f"Comparing {sig_oracle.shape[0]} samples with {sig_oracle.shape[1]} measurements.")
    
    # 1. Pointwise Error
    # Oracle has Rician Noise.
    # JAX simulation: I did NOT add noise in the benchmark script!
    # Wait, benchmark_connectome.py returns `model(p, acq)`. This is noiseless signal.
    # Oracle script RETURNS `S_noisy`.
    
    # Comparison strategy:
    # A) Compare noiseless Oracle vs noiseless JAX? (Requires modifying Oracle script to save noiseless).
    # B) Compare Distributions (KS test).
    
    # Ideally, physics verification should be Noiseless vs Noiseless.
    # But I wanted a "simulation benchmark", so JAX should ideally produce noisy data too if benchmarking throughput of "simulation".
    # But usually we benchmark signal generation speed.
    
    # If I compare Noisy Oracle vs Noiseless JAX, MSE will be huge (noise variance).
    # But the Mean of Oracle should match JAX.
    
    # Let's check if Oracle saved noiseless.
    # `generate_connectome_oracle.py`: saves `signals` which is `S_noisy`.
    
    # Solution: Update `benchmark_connectome.py` to add Rician noise (fair comparison of throughput).
    # AND/OR: Just compare the distributions.
    
    # Let's add Rician noise to JAX script to match the workload?
    # Or just analyze the "Mean Signal" vs "Oracle Mean"?
    
    # Let's assume validation is "Does JAX match the underlying physics?"
    # Since Oracle is noisy, I can't check exact values.
    # But I can check if JAX signal lies within the noise distribution of Oracle.
    # Or calculate the Residual = Oracle - JAX.
    # Mean(Residual) should be 0 (if Rician bias is handled? Rician has bias).
    # Mean(Residual) should be ~ Bias.
    
    # Better: Update Oracle to save Noiseless signal too?
    # Too late, simulation running.
    
    # Let's rely on Distribution Matching.
    # KS Test on the flattened signals.
    
    print("\n--- Statistical Comparison ---")
    
    # Flatten
    s1 = sig_oracle.flatten()
    s2 = sig_jax.flatten() # Noiseless
    
    # If s2 is noiseless and s1 is noisy (SNR=30), they won't match distributions exactly (s1 is broader).
    # BUT, if I add Rician noise to s2 locally, they should match.
    
    rng = np.random.default_rng(42)
    # SNR=30 usually means sigma = S(b=0)/30. S(b=0)=1 (or close).
    sigma = 1.0/30.0
    n1 = rng.normal(0, sigma, s2.shape)
    n2 = rng.normal(0, sigma, s2.shape)
    s2_noisy = np.sqrt((s2 + n1)**2 + n2**2)
    
    # Now compare s1 (Oracle Noisy) vs s2_noisy (Jax Noisy).
    
    ks_stat, p_val = ks_2samp(s1, s2_noisy)
    print(f"KS Statistic: {ks_stat:.5f}")
    print(f"P-Value: {p_val:.5e}")
    
    if p_val > 0.05:
        print("PASS: Distributions are statistically matching.")
    else:
        print("FAIL: Distributions differ significantly.")
        
    # Check Means
    mean_oracle = np.mean(s1)
    mean_jax_noisy = np.mean(s2_noisy)
    print(f"Mean Signal: Oracle={mean_oracle:.4f}, Jax(Noisy)={mean_jax_noisy:.4f}")
    
    # Check histograms
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(s1[::100], bins=100, alpha=0.5, label='Oracle (Scipy)', density=True)
        plt.hist(s2_noisy[::100], bins=100, alpha=0.5, label='Dmipy-Jax (GPU)', density=True)
        plt.legend()
        plt.title('Signal Histogram Comparison (Connectome 2.0)')
        plt.savefig('data/connectome_comparison.png')
        print("Saved histogram to data/connectome_comparison.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    compare_results()
