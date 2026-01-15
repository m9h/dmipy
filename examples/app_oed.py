
import streamlit as st
import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import numpy as np
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.design.oed import optimize_protocol
from scipy.special import jnp_zeros

# =============================================================================
# 1. Restricted Cylinder Model (Gaussian Phase Distribution)
# =============================================================================

# Precompute roots for Bessel function derivative J_1'(x) = 0
# These are needed for the GPD approximation in cylinders.
# We use scipy to get them, then convert to JAX array.
try:
    _CYLINDER_ROOTS_SCIPY = jnp_zeros(1, 20) # Use 20 roots for speed/demo
    CYLINDER_ROOTS = jnp.array(_CYLINDER_ROOTS_SCIPY)
except Exception as e:
    # Fallback if scipy not available or fails (approximate roots)
    CYLINDER_ROOTS = jnp.array([1.8412, 5.3314, 8.5363, 11.706, 14.864])

@jit
def restricted_cylinder_gpd_signal(
    bvals: jnp.ndarray,
    gradient_directions: jnp.ndarray,
    delta: float,
    Delta: float,
    diameter: float,
    mu: jnp.ndarray,
    lambda_par: float = 1.7e-9,
    diffusion_perp: float = 1.7e-9,
):
    """
    JAX implementation of Cylinder with GPD approximation (Van Gelderen 1994).
    
    Args:
        bvals: (N,) b-values in s/m^2.
        gradient_directions: (N, 3) unit vectors.
        delta: Pulse duration (s).
        Delta: Pulse separation (s).
        diameter: Cylinder diameter (m).
        mu: Cylinder orientation [theta, phi] (radians).
        lambda_par: Parallel diffusivity (m^2/s).
        diffusion_perp: Intrinsic perpendicular diffusivity (m^2/s).
    """
    # Constants
    gamma = 267.513e6 # rad/s/T
    
    # 1. Orientation handling
    theta, phi = mu[0], mu[1]
    sintheta = jnp.sin(theta)
    mu_cart = jnp.array([
        sintheta * jnp.cos(phi),
        sintheta * jnp.sin(phi),
        jnp.cos(theta)
    ])
    
    # Dot product n . mu
    dot_prod = jnp.dot(gradient_directions, mu_cart) # (N,)
    
    # Parallel Component (Stick)
    # E_par = exp(-b * D_par * (n.u)^2)
    E_par = jnp.exp(-bvals * lambda_par * dot_prod**2)
    
    # Perpendicular Component (GPD)
    # We need Gradient Strength G.
    # b = (gamma * G * delta)^2 * (Delta - delta/3)
    # G = sqrt( b / (gamma^2 * delta^2 * (Delta - delta/3)) )
    
    tau = Delta - delta / 3.0
    # Avoid div by zero if b=0
    safe_tau = jnp.where(tau <= 1e-6, 1.0, tau)
    
    # Calculate G from b-value
    # G^2 = b / (gamma^2 * delta^2 * tau)
    # We use G^2 directly in the formula usually.
    
    # Factor for G^2
    # G_squared = bvals / (gamma**2 * delta**2 * tau)
    
    # Actually, let's look at the formula:
    # ln(E_perp) = -2 * (gamma * G)^2 * SUM(...)
    # (gamma * G)^2 = gamma^2 * [b / (gamma^2 * delta^2 * tau)] = b / (delta^2 * tau)
    
    gamma_G_sq = bvals / (delta**2 * safe_tau)
    
    # Component of G perpendicular to cylinder: G_perp = G * sin(alpha)
    # sin^2(alpha) = 1 - (n.u)^2
    sin_sq_angle = 1.0 - dot_prod**2
    
    # Effective driving factor for perpendicular attenuation
    factor = -2 * gamma_G_sq * sin_sq_angle # Shape (N,)
    
    # Summation term
    R = diameter / 2.0
    D = diffusion_perp
    
    # roots alpha_m = x_m / R
    alpha_m = CYLINDER_ROOTS / R # Shape (M,)
    alpha_m_sq = alpha_m**2
    alpha_m_sq_D = alpha_m_sq * D # Shape (M,)
    
    # Terms dependent on m
    # shape (M,)
    term_denom = (D**2) * (alpha_m**6) * ( (R**2 * alpha_m_sq) - 1 )
    
    # Time dependent terms
    # exp(-alpha^2 D delta) ...
    # We need to broadcast time terms against M roots? 
    # Current delta/Delta are scalar floats in this simple app context.
    
    am2D_delta = alpha_m_sq_D * delta
    am2D_Delta = alpha_m_sq_D * Delta
    
    # Summands numerator
    # 2*alpha2D*delta - 2 + 2*exp(-alpha2D*delta) + 2*exp(-alpha2D*Delta) 
    # - exp(-alpha2D*(Delta - delta)) - exp(-alpha2D*(Delta + delta))
    
    numerator = (
        2 * alpha_m_sq_D * delta 
        - 2 
        + 2 * jnp.exp(-am2D_delta)
        + 2 * jnp.exp(-am2D_Delta)
        - jnp.exp(-alpha_m_sq_D * (Delta - delta))
        - jnp.exp(-alpha_m_sq_D * (Delta + delta))
    )
    
    sum_terms = numerator / term_denom # (M,)
    total_sum = jnp.sum(sum_terms) # Scalar (given scalar diameter/delta)
    
    # E_perp = exp( factor * sum )
    # factor is (N,), total_sum is scalar
    E_perp = jnp.exp(factor * total_sum)
    
    return E_par * E_perp

# Wrapper function for the optimizer
def model_wrapper(bvalues, gradient_directions, delta, Delta, **kwargs):
    # kwargs contains 'diameter', etc.
    # We fix mu to x-axis [pi/2, 0] for simple sensitivity optimization if not provided?
    # Or should we optimize for unknown orientation?
    # Usually OED assumes we know the ground truth params and want to max sensitivity around them.
    # Let's assume fibers along Z-axis [0, 0] for simplicity in demo visual.
    # theta=0, phi=0.
    
    mu_fixed = jnp.array([0.0, 0.0]) 
    
    diameter = kwargs.get('diameter', 2.0e-6)
    
    return restricted_cylinder_gpd_signal(
        bvalues, gradient_directions, 
        delta=delta, Delta=Delta,
        diameter=diameter,
        mu=mu_fixed
    )


# =============================================================================
# 2. Streamlit App
# =============================================================================

def main():
    st.set_page_config(page_title="OED Protocol Designer", layout="wide")
    
    st.title("🧬 MRI Protocol Designer (OED)")
    st.markdown("""
    Optimize your diffusion MRI acquisition protocol to be maximally sensitive 
    to a specific **Axon Diameter**.
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("1. Target Tissue Parameters")
    
    target_diameter_um = st.sidebar.slider(
        "Target Axon Diameter (µm)", 
        min_value=1.0, max_value=10.0, value=3.0, step=0.5
    )
    target_diameter = target_diameter_um * 1e-6

    st.sidebar.header("2. Scanner Constraints")
    max_gradient = st.sidebar.slider(
        "Max Gradient Strength (mT/m)", 
        40, 300, 80, step=10
    )
    
    # Constraints
    G_max = max_gradient * 1e-3 # T/m
    delta = 0.020 # 20ms
    Delta = 0.040 # 40ms
    gamma = 267.513e6
    
    # Calculate b_max from G_max
    # b = (gamma * G * delta)^2 * (Delta - delta/3)
    b_max_calc = (gamma * G_max * delta)**2 * (Delta - delta/3.0)
    b_max_calc_smm = b_max_calc / 1e6 # s/mm^2
    
    st.sidebar.write(f"**Calculated Max b-value:** {b_max_calc_smm:.0f} s/mm²")
    
    st.sidebar.header("3. Optimization")
    run_opt = st.sidebar.button("✨ Optimize Protocol")

    # --- Main Area ---
    
    # Helper to create random protocol
    def create_random_protocol(n_measurements=30, b_max=3000e6):
        # Random directions
        key = jax.random.PRNGKey(42)
        dirs = jax.random.normal(key, (n_measurements, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        
        # Random b-values (uniform)
        # bvals = jax.random.uniform(key, (n_measurements,), minval=100e6, maxval=b_max)
        # Using linspace shells for 'Before' looks cleaner
        bvals = jnp.linspace(200e6, b_max, n_measurements)
        
        return JaxAcquisition(
            bvalues=bvals,
            gradient_directions=dirs,
            delta=delta,
            Delta=Delta
        )

    col1, col2 = st.columns(2)

    if run_opt:
        with st.spinner("Optimizing generic protocol..."):
            # 1. Setup
            acq_initial = create_random_protocol(n_measurements=60, b_max=b_max_calc)
            
            # 2. Run Optimization
            # Target: diameter
            target_params = {'diameter': target_diameter}
            
            # Optimize!
            acq_opt, history = optimize_protocol(
                acq_initial,
                model_wrapper,
                target_params,
                n_steps=50,
                learning_rate=0.05,
                b_min=0.0,
                b_max=b_max_calc,
                return_history=True,
                b_scale=1e9
            )
            
            # 3. Analyze Results
            # Compute Final Sensitivity (CRB proxy)
            # FIM = J.T @ J
            # CRB ~ 1/FIM (scalar parameter)
            
            # Loss is -logdet(FIM). For 1 param, log(FIM).
            # Lower loss -> Higher FIM -> Better Precision.
            
            loss_initial = history['loss'][0]
            loss_final = history['loss'][-1]
            
            # Precision (1/variance) is proportional to FIM
            # log(Prec) ~ -Loss
            # Improvement ratio = exp(-Loss_final) / exp(-Loss_initial) = exp(Loss_init - Loss_final)
            precision_gain = np.exp(loss_initial - loss_final)
            
            st.success(f"Optimization Complete! Predicted Precision Improvement: **{precision_gain:.1f}x**")
            
            # --- Visuals ---
            
            # Plot 1: B-value Distribution (Histograms)
            fig, ax = plt.subplots(figsize=(6, 4))
            
            b_init_smm = acq_initial.bvalues / 1e6
            b_opt_smm = acq_opt.bvalues / 1e6
            
            # ax.hist(b_init_smm, bins=15, alpha=0.5, label='Before (Uniform)', color='gray')
            ax.hist(b_opt_smm, bins=15, alpha=0.7, label='After (Optimized)', color='#FF4B4B')
            
            ax.set_xlabel("b-value (s/mm²)")
            ax.set_ylabel("Count")
            ax.set_title("Optimized B-value Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            col1.pyplot(fig)
            
            # Plot 2: Optimization History
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(history['loss'], linewidth=2, color='#0068C9')
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Loss (-LogDet FIM)")
            ax2.set_title("Convergence Trace")
            ax2.grid(True, alpha=0.3)
            
            col2.pyplot(fig2)
            
            st.markdown("### Interpretation")
            st.info(
                f"The optimizer shifted b-values to maximize sensitivity for a **{target_diameter_um} µm** cylinder. "
                "Notice how b-values cluster around specific 'optimal' shells rather than being uniform."
            )

    else:
        st.info("Adjust parameters on the left and click **Optimize Protocol** to see the magic! 🎩")

if __name__ == "__main__":
    main()
