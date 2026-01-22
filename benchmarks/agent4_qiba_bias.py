
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import time
from dmipy_jax.signal_models.ivim import IVIM
from dmipy_jax.algebra.initializers import segmented_ivim_init

# QIBA DRO Simulation
# "Bias Profile" across SNR levels.
# Ground Truth Params (Liver-like)
D_TRUE = 1.0e-9 # 1.0 um^2/ms
DP_TRUE = 40.0e-9 
F_TRUE = 0.20 
S0_TRUE = 100.0

def simulate_qiba_dro(snr_levels, n_reps=1000):
    # Standard QIBA/AAPM b-values
    bvals_si = jnp.array([0, 10, 20, 30, 50, 80, 100, 200, 400, 800]) * 1e6
    grad = jnp.zeros((len(bvals_si), 3))
    grad = grad.at[:, 0].set(1.0)
    
    ivim = IVIM()
    clean_sig = ivim(bvals_si, grad, D_tissue=D_TRUE, D_pseudo=DP_TRUE, f=F_TRUE) * S0_TRUE
    
    datasets = {}
    key = jax.random.PRNGKey(42)
    
    for snr in snr_levels:
        sigma = S0_TRUE / snr
        # Rician parameters
        k1, k2 = jax.random.split(key)
        key = k1
        n1 = jax.random.normal(k1, (n_reps, len(bvals_si))) * sigma
        n2 = jax.random.normal(k2, (n_reps, len(bvals_si))) * sigma
        noisy = jnp.sqrt((clean_sig + n1)**2 + n2**2)
        datasets[snr] = noisy
        
    return datasets, bvals_si

def run_agent4():
    print("=== Agent 4: QIBA Precision & Bias Benchmark ===")
    snrs = [20, 50, 100, 200] # Clinical Range
    data_map, bvals = simulate_qiba_dro(snrs)
    
    print(f"Ground Truth: D={D_TRUE}, f={F_TRUE}, D*={DP_TRUE}")
    print(f"{'SNR':<5} | {'Method':<15} | {'Bias D (%)':<10} | {'Bias f (%)':<10} | {'Pass QIBA?'}")
    print("-" * 65)
    
    for snr in snrs:
        sigs = data_map[snr]
        
        # 1. Algebraic (Log-Linear)
        preds_alg = jax.vmap(lambda s: segmented_ivim_init(bvals, s, b_threshold=200e6))(sigs)
        # preds: [D, Dp, f]
        bias_d_alg = jnp.mean(preds_alg[:,0]) - D_TRUE
        bias_f_alg = jnp.mean(preds_alg[:,2]) - F_TRUE
        
        pct_d_alg = 100 * bias_d_alg / D_TRUE
        pct_f_alg = 100 * bias_f_alg / F_TRUE
        
        # 2. NLLS (Initialized by Algebraic)
        # We assume refinement cleans up bias?
        # Note: At low SNR, NLLS can drift to boundaries.
        pass_qiba_alg = abs(pct_d_alg) < 5.0 and abs(pct_f_alg) < 10.0 # QIBA approx tolerances
        
        print(f"{snr:<5} | {'Algebraic':<15} | {pct_d_alg:<10.2f} | {pct_f_alg:<10.2f} | {pass_qiba_alg}")

    print("\nConclusion: Algebraic Initialization is robust at high SNR, but check low SNR bias.")

if __name__ == "__main__":
    run_agent4()
