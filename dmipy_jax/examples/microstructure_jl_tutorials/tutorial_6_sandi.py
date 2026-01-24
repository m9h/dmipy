
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.sandi import get_sandi_model

def main():
    print("Tutorial 6: SANDI Model")
    
    # 1. Setup SANDI Model
    # soma_density (f_sphere), neurite_density (f_stick + f_zeppelin), extra is Ball.
    # We use get_sandi_model helper.
    
    sandi_func = get_sandi_model()
    
    # 2. Acquisition
    # SANDI requires high b-values to sensitivity to soma.
    # b-values up to 10,000 s/mm^2 or more used in original paper/Human High Grad.
    # Let's use b=1000, 3000, 5000, 10000.
    
    bvals_shell = jnp.array([1000, 2000, 3000, 5000, 10000]) * 1e6
    n_dirs = 10
    
    key = jax.random.PRNGKey(42)
    vecs = jax.random.normal(key, (n_dirs, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    bvals = jnp.kron(bvals_shell, jnp.ones(n_dirs))
    bvecs = jnp.tile(vecs, (5, 1))
    
    # SANDI requires delta/Delta.
    # Connectome scanner: delta=10ms, Delta=40ms approx ?
    # Standard: delta=0.0129, Delta=0.0218 (HCP?) 
    # Let's use generic values.
    acq = JaxAcquisition(
        bvalues=bvals, 
        gradient_directions=bvecs,
        delta=0.015,
        Delta=0.040
    )
    
    # 3. Simulate Signal
    # Params: [theta, phi, f_stick, f_sphere, f_ball, diameter, lambda_perp]
    # Note: f_zeppelin = 1 - sum(fractions)
    
    theta, phi = 0.0, 0.0 # Z-axis
    f_stick = 0.4  # Intra-neurite
    f_sphere = 0.3 # Soma
    f_ball = 0.1   # CSF
    # f_zeppelin = 1 - 0.8 = 0.2
    
    diameter = 10e-6 # 10um Sphere
    lambda_perp = 0.0 # Stick/Zeppelin radius ~ 0 or small
    
    params = jnp.array([theta, phi, f_stick, f_sphere, f_ball, diameter, lambda_perp])
    
    S_sim = sandi_func(params, acq)
    
    # 4. Visualize Somas Influence
    # Plot Signal vs b-value (Powder Average)
    
    unique_b = jnp.unique(bvals_shell)
    S_mean = []
    for b in unique_b:
        mask = jnp.isclose(acq.bvalues, b)
        S_mean.append(jnp.mean(S_sim[mask]))
    S_mean = jnp.array(S_mean)
    
    # Compare with No Soma (f_sphere = 0)
    # Renormalize fractions? Or just set f_sphere=0, add to f_stick?
    f_stick_2 = f_stick + f_sphere
    params_no_soma = jnp.array([theta, phi, f_stick_2, 0.0, f_ball, diameter, lambda_perp])
    S_sim_no_soma = sandi_func(params_no_soma, acq)
    
    S_mean_no_soma = []
    for b in unique_b:
        mask = jnp.isclose(acq.bvalues, b)
        S_mean_no_soma.append(jnp.mean(S_sim_no_soma[mask]))
    S_mean_no_soma = jnp.array(S_mean_no_soma)

    plt.figure(figsize=(8, 6))
    plt.semilogy(unique_b * 1e-6, S_mean, 'bo-', label=f'With Soma (f={f_sphere})')
    plt.semilogy(unique_b * 1e-6, S_mean_no_soma, 'r^--', label='No Soma (f=0)')
    plt.xlabel('b-value (ms/um^2)') # 10^6 s/m^2 = 1 ms/um^2
    plt.ylabel('Signal (Log scale)')
    plt.title('Tutorial 6: SANDI Signal Sensitivity to Soma')
    plt.legend()
    plt.grid(True)
    plt.savefig('tutorial_6_output.png')
    print("Saved SANDI plot.")

if __name__ == "__main__":
    main()
