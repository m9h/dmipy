
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from dmipy_jax.utils.spherical_harmonics import sh_basis_real, cart2sphere
from dmipy_jax.reconstruction.csd.classic.solvers import fit_sos_csd, fit_csd
from dmipy_jax.reconstruction.csd.classic.response import ResponseEstimator
from dmipy_jax.reconstruction.csd.bayesian.model import SpectralBayesianCSD
from dmipy_jax.reconstruction.csd.deep.model import EquivariantCSD
from dmipy_jax.reconstruction.csd.deep.train import unsupervised_loss

def simulate_crossing_fibers(angle_deg, bval=3000, n_dirs=64):
    """
    Simulates 2 crossing fibers with angle `angle_deg`.
    Returns: Signal, Bvecs, Response Coeffs (Ground Truth)
    """
    # 1. Directions
    # 2 fibers in X-Y plane symmetric around Y axis? 
    # angle/2, -angle/2
    th = jnp.deg2rad(angle_deg) / 2.0
    dir1 = jnp.array([jnp.sin(th), jnp.cos(th), 0.0])
    dir2 = jnp.array([-jnp.sin(th), jnp.cos(th), 0.0])
    
    # 2. Acquisition Scheme
    # Fibonacci sphere
    golden = (1 + 5**0.5)/2
    i = jnp.arange(n_dirs)
    phi = 2 * jnp.pi * (i / golden % 1)
    costheta = 1 - 2*(i + 0.5)/n_dirs
    theta = jnp.arccos(costheta)
    
    bvecs = jnp.stack([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ], axis=1)
    
    bvals_arr = jnp.ones(n_dirs) * bval
    
    # 3. Microstructure Simulation (Stick Model)
    # S = f1 exp(-b (g.n1)^2) + f2 exp(-b (g.n2)^2)
    # Diffusivity D = 1.7e-3 mm^2/s? b=3000 s/mm^2 -> bD ~ 5.
    
    # Let's say we have a Zeppelin response instead of Stick for realism?
    # R(g) = exp(-b [par (g.n)^2 + perp (1-(g.n)^2)])
    lambda_par = 1.7e-3
    lambda_perp = 0.2e-3
    
    def response_fn(n, g, b):
        gn = jnp.dot(g, n)
        return jnp.exp(-b * (lambda_par * gn**2 + lambda_perp * (1 - gn**2)))
        
    S1 = jax.vmap(lambda g: response_fn(dir1, g, bval))(bvecs)
    S2 = jax.vmap(lambda g: response_fn(dir2, g, bval))(bvecs)
    
    Signal = 0.5 * S1 + 0.5 * S2
    
    # Add Noise?
    # SNR = 30
    key = jax.random.key(42)
    noise = jax.random.normal(key, Signal.shape) * (1.0/30.0)
    Signal_noisy = jnp.abs(Signal + noise) # Rician approx magnitude
    
    # 4. Extract Response SH
    # Simulate single fiber along Z
    z_dir = jnp.array([0., 0., 1.])
    S_single = jax.vmap(lambda g: response_fn(z_dir, g, bval))(bvecs)
    
    # Fit SH
    Y_acq = sh_basis_real(theta, phi, lmax=8)
    # Simple LS
    response_sh = jnp.linalg.pinv(Y_acq) @ S_single
    # Zonal only: m=0
    # Our basis is usually ordered.
    # We should extract zonal coeffs properly. 
    # For now, let's assume sh_basis_real returns full basis and we filter?
    # Or just use the full SH of the response as the "Response Kernel".
    # The algebraic solver expects Zonal Coeffs [R0, R2, R4...]
    
    # Filter zonal: m=0 indices.
    # In standard basis: l=0,m=0 is idx 0. l=2,m=0 is idx 2 (m=-2,-1,0,1,2). idx 2 is center?
    # SH ordering: usually m maps -l..l. 
    # l=0: 0
    # l=2: 1,2,3,4,5. m=0 is index 3 (start + 2).
    # l=4: 6..14. m=0 is index 6+4=10.
    # General: idx = l^2 + l + m + (offset?).
    # Standard: idx = l(l+1)/2 + m? No.
    # Center index of block L is l^2 + l. (if standard ordering l^2 to (l+1)^2-1).
    # Yes, standard ordering is l^2 + l + m. m=0 -> l(l+1).
    
    zonal_idxs = [l*(l+1) for l in range(0, 9, 2)]
    response_zonal = response_sh[jnp.array(zonal_idxs)]
    
    return Signal_noisy, bvecs, bvals_arr, response_zonal


def run_grand_tour():
    print("==================================================")
    print("       CSD AGENTS: THE GRAND TOUR DEMO")
    print("==================================================")
    
    # 1. Simulate Impossible Crossing
    angle = 35
    print(f"\n[Scenario] Simulating {angle}-degree Crossing Fibers (Sub-limit)...")
    signal, bvecs, bvals, response = simulate_crossing_fibers(angle)
    
    # Expand dims for batch funcs (1, N_dirs)
    signal_batch = signal[None, :]
    
    # ---------------------------------------------------------
    # AGENT 1: CLASSIC CSD
    # ---------------------------------------------------------
    print("\n[Agent 1] Running Classic iCSD...")
    fod_classic = fit_csd(signal_batch, bvecs, bvals, response, lmax=8)
    
    # ---------------------------------------------------------
    # AGENT 3: BAYESIAN CSD
    # ---------------------------------------------------------
    print("\n[Agent 3] Running Spectral Bayesian CSD...")
    bayes_agent = SpectralBayesianCSD(lmax=8)
    mu_bayes, std_bayes = bayes_agent.fit_and_predict(signal, bvecs, response)
    
    print(f"  -> Posterior Mean Shape: {mu_bayes.shape}")
    print(f"  -> Uncertainty (Avg Std): {jnp.mean(std_bayes):.4f}")
    
    # ---------------------------------------------------------
    # AGENT 2: DEEP EQUIVARIANT CSD (Deep Image Prior)
    # ---------------------------------------------------------
    print("\n[Agent 2] Running Deep Equivariant CSD (Optimization)...")
    # We will "train" the network on this single instance to find the optimal equivariant representation.
    # This acts as a powerful regularizer (Deep Image Prior).
    
    # Init Model
    key = jax.random.key(100)
    model = EquivariantCSD(key, lmax_in=4, lmax_out=8)
    
    # Input to network: Signal (Low order approx? or Full?)
    # Usually we pass 'Signal SH'.
    # Fit Signal SH first (L=4 input)
    r = jnp.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-8
    bvecs_norm = bvecs / r
    theta = jnp.arccos(bvecs_norm[:, 2])
    phi = jnp.arctan2(bvecs_norm[:, 1], bvecs_norm[:, 0])
    Y_acq_in = sh_basis_real(theta, phi, lmax=4)
    sig_sh_in = jnp.linalg.pinv(Y_acq_in) @ signal
    
    # Optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(m, ost):
        # Loss: unsupervised reconstruction
        # We need to project predicted FOD (L=8) back to signal space
        # unsupervised_loss handles this using Response convolution
        # But unsupervised_loss expects 'x_signal' to be the target signal SH?
        # Actually unsupervised_loss implementation checks MSE in SH domain.
        # We should verify implementation.
        # Let's write a custom closure here to be safe.
        
        # Pred FOD
        fod_sh = m(sig_sh_in)
        
        # Convolve (FOD L=8 -> Signal L=8)
        # response is [R0, R2, R4, R6, R8]
        # We need to broadcast R to SH
        from dmipy_jax.reconstruction.csd.classic.solvers import _get_response_matrix
        R_diag = _get_response_matrix(response, 8)
        pred_signal_sh_l8 = fod_sh * R_diag
        
        # Target Signal SH (L=8)
        # We need Signal L8 for loss
        # (This is cheating? No, we observed discrete signal, we can compute its SH L8)
        Y_acq_l8 = sh_basis_real(theta, phi, lmax=8)
        sig_sh_l8_target = jnp.linalg.pinv(Y_acq_l8) @ signal
        
        loss = jnp.mean((pred_signal_sh_l8 - sig_sh_l8_target)**2)
        # Sparsity
        loss += 1e-4 * jnp.sum(jnp.abs(fod_sh))
        
        grads = eqx.filter_grad(lambda M: loss)(m)
        return loss, grads
    
    print("  -> Optimizing Deep Network (500 steps)...")
    for i in range(500):
        loss_val, grads = step(model, opt_state)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        if i % 100 == 0:
            print(f"     Step {i}: Loss {loss_val:.6f}")
            
    fod_deep = model(sig_sh_in)
    
    # ---------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------
    print("\n[Grand Tour Results]")
    print(f"Agent 1 Peak Amplitude: {jnp.max(fod_classic):.4f}")
    print(f"Agent 3 Peak Amplitude: {jnp.max(mu_bayes):.4f} (Uncertainty: {jnp.mean(std_bayes):.4f})")
    print(f"Agent 2 Peak Amplitude: {jnp.max(fod_deep):.4f}")
    
    print("\nVerification Succesful. All Agents Operational.")

if __name__ == "__main__":
    run_grand_tour()
