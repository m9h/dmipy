import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def compare_results():
    print("Loading Data...")
    oracle = np.load("data/oracle_sims.npz") # Note: complex_oracle_1M? 
    # The oracle script saved to /data/complex_oracle_1M.npz
    # But locally it is mounted to data/
    # Let's check filename. 'generate_complex_oracle.py' saved to 'complex_oracle_1M.npz'.
    # The previous simple one was 'oracle_sims.npz'.
    
    try:
        oracle = np.load("data/complex_oracle_1M.npz")
    except FileNotFoundError:
        print("Complex oracle file not found. Checking simple...")
        oracle = np.load("data/oracle_sims.npz")

    jax_data = np.load("data/dmipy_jax_1M.npz")
    
    # Extract Signals (Noisy)
    # Oracle saves: signals (noisy), signals_clean
    # Jax saves: signals (noisy), signals_clean
    
    s_oracle = oracle['signals']
    s_jax = jax_data['signals']
    
    t_oracle = float(oracle.get('time_seconds', 0.0)) # Might not have saved it in first run?
    # Complex script saved 'time_seconds'.
    
    t_jax = float(jax_data['time_seconds'])
    
    print(f"Dataset Shapes: Oracle {s_oracle.shape}, Jax {s_jax.shape}")
    
    # Flatten signals for distribution comparison
    s_oracle_flat = s_oracle.flatten()
    s_jax_flat = s_jax.flatten()
    
    print("\n--- Performance Comparison ---")
    print(f"Oracle Time: {t_oracle:.4f} s")
    print(f"Dmipy-Jax Time: {t_jax:.4f} s")
    speedup = t_jax / t_oracle if t_oracle > 0 else 0
    print(f"Relative Speed: Oracle is {speedup:.2f}x faster")
    
    print("\n--- Statistical Comparison (KS Test) ---")
    # Kolmogorov-Smirnov test on signal distributions
    # We expect them to come from the same distribution (Rician matches)
    # N is huge (1M * 200 = 200M points). 
    # KS test will be very sensitive.
    # We'll sample subset for KS to avoid memory/time issues and over-sensitivity.
    
    subset_size = 100_000
    idx = np.random.choice(len(s_oracle_flat), subset_size, replace=False)
    
    ks_stat, p_value = ks_2samp(s_oracle_flat[idx], s_jax_flat[idx])
    
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-Value: {p_value:.4e}")
    
    if p_value > 0.05:
        print(">> Distributions are statistically indistinguishable (p > 0.05)")
    else:
        print(">> Distributions are statistically different (p <= 0.05)")
        # With 1M samples, even tiny differences in random number generation (numpy vs torch)
        # or float precision will trigger p < 0.05.
        # KS Stat is more useful size effect.
    
    print("\n--- Signal Statistics ---")
    print(f"Oracle Mean: {np.mean(s_oracle_flat):.4f}, Std: {np.std(s_oracle_flat):.4f}")
    print(f"Jax    Mean: {np.mean(s_jax_flat):.4f}, Std: {np.std(s_jax_flat):.4f}")
    
    # Check max deviation
    # We can't compare sample-to-sample because random seeds/generators differ.
    
    # Clean Signal Comparison?
    # Ideally we'd compare clean signals for the SAME parameters.
    # But parameters were generated randomly in each script (same seed 42? yes).
    # If seeds match exactly (numpy), theta might match.
    # But torch vs jax random generation differs for 'rand'.
    # So we cannot assume paired samples.
    # Distribution match is the goal.

if __name__ == "__main__":
    compare_results()
