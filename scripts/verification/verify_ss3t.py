import jax
import jax.numpy as jnp
import numpy as np
import time
from dmipy_jax.reconstruction.ss3t import SS3T
from dmipy_jax.utils.spherical_harmonics import sh_basis_real, cart2sphere

def generate_synthetic_voxel(bvecs, bval, response_wm, response_gm, response_csf, sh_order):
    # Cross fibers X and Y
    dir1 = jnp.array([[1.0, 0.0, 0.0]])
    dir2 = jnp.array([[0.0, 1.0, 0.0]])
    
    r1, th1, ph1 = cart2sphere(dir1[:,0], dir1[:,1], dir1[:,2])
    r2, th2, ph2 = cart2sphere(dir2[:,0], dir2[:,1], dir2[:,2])
    
    Y1 = sh_basis_real(th1, ph1, sh_order)
    Y2 = sh_basis_real(th2, ph2, sh_order)
    
    # Fractions: GM=0.2, CSF=0.1, WM=0.7 (split 0.35/0.35)
    amp1 = 0.35
    amp2 = 0.35
    c_gt = amp1 * Y1[0] + amp2 * Y2[0]
    
    w_gm_gt = 0.2
    w_csf_gt = 0.1
    params_gt = jnp.concatenate([jnp.array([w_gm_gt, w_csf_gt]), c_gt])
    
    model = SS3T(sh_order, response_wm, response_gm, response_csf, bval)
    signal = model.predict(params_gt, bvecs)
    
    return signal, params_gt, model

def main():
    print("--- SS3T Verification with Priors ---")
    
    sh_order = 8
    bval = 1000.0
    evals_wm = (1.7e-3, 0.2e-3, 0.2e-3)
    s_gm = np.exp(-bval * 0.8e-3)
    s_csf = np.exp(-bval * 3.0e-3)
    
    key = jax.random.PRNGKey(42)
    vecs = jax.random.normal(key, (64, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    clean_signal, params_gt, model = generate_synthetic_voxel(
        vecs, bval, evals_wm, s_gm, s_csf, sh_order
    )
    
    # Noisy data
    sigma = 0.05
    n1 = jax.random.normal(key, clean_signal.shape) * sigma
    n2 = jax.random.normal(jax.random.PRNGKey(1), clean_signal.shape) * sigma
    noisy_data = jnp.sqrt((clean_signal + n1)**2 + n2**2)
    
    # 1. Fit WITHOUT Priors
    print("\nFitting Standard SS3T (No Priors)...")
    t0 = time.time()
    res_noprior, _ = model.fit_voxel(noisy_data, vecs, priors=None)
    print(f"Done in {time.time()-t0:.3f}s")
    
    idx_gm, idx_csf = 0, 1
    wm_factor = jnp.sqrt(4*jnp.pi)
    
    gm_np = res_noprior[idx_gm]
    csf_np = res_noprior[idx_csf]
    wm_np = res_noprior[2] * wm_factor
    
    print(f"NoPrior -> GM: {gm_np:.3f}, CSF: {csf_np:.3f}, WM: {wm_np:.3f}")
    
    # 2. Fit WITH Priors
    # Let's give it a hint close to truth
    priors = jnp.array([0.2, 0.1, 0.7]) # [GM, CSF, WM]
    print("\nFitting SS3T (With Priors [0.2, 0.1, 0.7])...")
    # Strong prior weight
    model.lambda_priors = 10.0 
    
    t0 = time.time()
    res_prior, _ = model.fit_voxel(noisy_data, vecs, priors=priors)
    print(f"Done in {time.time()-t0:.3f}s")
    
    gm_p = res_prior[idx_gm]
    csf_p = res_prior[idx_csf]
    wm_p = res_prior[2] * wm_factor
    
    print(f"WithPrior -> GM: {gm_p:.3f}, CSF: {csf_p:.3f}, WM: {wm_p:.3f}")
    
    # Verify that prior fit is closer to prior values (and truth) if prior is good
    # Here truth matches prior, so it should improve/stabilize.
    
    err_np = abs(gm_np - 0.2) + abs(csf_np - 0.1)
    err_p = abs(gm_p - 0.2) + abs(csf_p - 0.1)
    
    print(f"\nError (GM+CSF): NoPrior={err_np:.4f}, WithPrior={err_p:.4f}")
    
    # Expect improvement or similar good performance.
    
    print("\nVerification Finished.")

if __name__ == "__main__":
    main()
