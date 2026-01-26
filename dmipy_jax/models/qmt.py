
import jax
import jax.numpy as jnp
import equinox as eqx

def super_lorentzian_lineshape(delta: jax.Array, T2r: float) -> jax.Array:
    """
    Super-Lorentzian Lineshape for the Macromolecular Pool.
    
    G(delta) = \int_0^{pi/2} \sin(\theta) \sqrt{2/\pi} \frac{T2r}{|3\cos^2\theta - 1|} 
               \exp( -2 (2\pi \delta \frac{T2r}{3\cos^2\theta - 1} )^2 ) d\theta
               
    Approximated by numerical integration or interpolating function.
    Here we use a high-precision discretization of the integral.
    """
    # Grid for theta integration (0 to pi/2)
    # Avoid singularities at theta approx 54.7 deg (magic angle) where 3cos^2-1 = 0.
    # We split integration or use dense grid.
    
    n_pts = 100
    theta = jnp.linspace(0.0, jnp.pi/2, n_pts)
    sin_theta = jnp.sin(theta)
    cos2_theta = jnp.cos(theta)**2
    term = jnp.abs(3 * cos2_theta - 1)
    
    # Regularize singularity
    term = jnp.where(term < 1e-6, 1e-6, term)
    
    T2_eff = T2r / term
    
    # Gaussian term for each orientation
    # exp( - (2 * pi * delta * T2_eff)^2 / 2 ) ?
    # Standard eq: G(w) = T2 / sqrt(2pi) * exp( - (w T2)^2 / 2 ) ?
    # Let's verify scaling.
    # Usually: g(w) = ... exp( - (2 pi delta T2)^2 ) ... check prefactors.
    
    # Using Morrison & Henkelman (1995) form:
    # Int sin(theta) * sqrt(2/pi) * (1/T2eff) * exp( - (2 pi delta / T2eff)^2 / ?? )
    # Let's stick to a standard implementation form:
    
    # Argument X = 2 * pi * delta * T2r
    X = 2 * jnp.pi * delta * T2_eff # Corrected T2_eff inside
    
    # We'll use the definition: f_L(w) = T2/sqrt(2pi) * exp(-(w T2)^2/2) ??
    # Actually, let's use the explicit integrated form if possible, or summed.
    
    
    val = jnp.sqrt(2/jnp.pi) * (T2_eff) * jnp.exp( -2.0 * (2 * jnp.pi * delta * T2_eff)**2 )
    
    # Integration
    # jnp.trapz is deprecated in recent versions. Use jnp.trapezoid.
    integral = jnp.trapezoid(val * sin_theta, theta)
    return integral

class qMT_SPGR(eqx.Module):
    """
    Quantitative Magnetization Transfer for SPGR sequences.
    Based on the simplistic "Continuous Wave equivalent" approximation
    or Sled & Pike pulsed correction.
    
    For high-throughput fitting, we implement the Ramani/Yarnykh Approximation:
    - Treat saturation pulse as average rate W.
    - W = pi * gamma^2 * B1^2 * G(Delta) * (tau / TR)
    """
    
    @staticmethod
    def forward(
        f: float,        # Pool size ratio F = M0r / (M0f + M0r) ? No, f is usually MPF = M0r/M0_total.
        k_mf: float,     # Exchange rate m->f (inverse of T_m->f ?? checks conventions)
                         # Usually k is given as k_fr (f->r) or k_mf (r->f). 
                         # We use k_mf (macromolecular to free).
        R1_f: float,     # 1/T1_free (s^-1)
        R1_m: float,     # 1/T1_bound (s^-1)
        T2_f: float,     # T2_free (s)
        T2_m: float,     # T2_bound (s)
        
        # Protocol
        tr: float,       # Repetition Time (s)
        exc_fa: float,   # Excitation Flip Angle (degrees)
        mt_fa: float,    # MT Pulse Flip Angle (degrees)
        mt_offset: float,# MT Pulse Offset (Hz)
        mt_dur: float = 0.010 # MT Pulse Duration (s) (e.g. 10ms typically)
    ) -> jax.Array:
        
        # SI Units
        exc_rad = jnp.radians(exc_fa)
        mt_rad = jnp.radians(mt_fa)
        
        # 1. Determine Saturation Rate W
        # B1_avg_squared (for CW approx matching pulse power)
        # alpha_mt = gamma * B1 * tau
        # B1 = alpha_mt / (gamma * tau) -- this assumes square pulse. 
        # Real pulses like Gaussian/Fermi have shape factors.
        # W = pi * w1^2 * G(Delta)
        # But we must account for Duty Cycle: tau / TR.
        
        gamma = 42.576e6 * 2 * jnp.pi # rad/s/T
        # Approx B1 amplitude from Flip Angle
        # w1_rms = mt_rad / mt_dur  (rad/s)
        # Power P = w1_rms^2
        
        w1_sq = (mt_rad / mt_dur)**2 
        
        # Lineshape
        g_val = super_lorentzian_lineshape(mt_offset, T2_m)
        
        # Saturation Rate (Mean over TR)
        # W = pi * w1^2 * G * (tau/TR)
        W = jnp.pi * w1_sq * g_val * (mt_dur / tr)
        
        # 2. Steady State Magnetization (Helms / Yarnykh)
        # Solve coupled equations.
        # R1_effective for free pool
        # Mzs = M0 * ( (1-E1)*cos(a) ... )
        
        # Using simple matrix inversion for steady state (exact for CW)
        # A = [[-R1f - kfr - Wf,   kmf         ],
        #      [ kfr,             -R1m - kmf - W]]
        # But this is CW. SPGR excitation is instantaneous.
        
        # Let's use the explicit SPGR steady state formula from Yarnykh 2002 EQ [8]
        # M_zf / M_0f approx ...
        
        # Explicit Code for 2-pool steady state with Excitation + Relaxation
        # Let MPF = f_struct.
        # M0r = f * Total
        # M0f = (1-f) * Total
        # But usually we fit normalized signal.
        
        # Parameters
        R1r = R1_m
        R1f = R1_f
        # Exchange constraints: k_fr * M0f = k_mf * M0r
        # k_fr * (1-f) = k_mf * f
        # k_fr = k_mf * f / (1-f)
        k_fr = k_mf * f / (1.0 - f + 1e-9)
        
        # Relaxation Matrix R
        # dM/dt = R * M + C
        # R = [[ -R1f - kfr,    kmf        ],
        #      [  kfr,         -R1r - kmf - W ]]
        # Note: W is applied to restricted pool usually. Direct effect on free pool is usually small if offset >> width.
        # We assume W_f = 0.
        
        R = jnp.array([
            [-(R1f + k_fr),       k_mf       ],
            [  k_fr,         -(R1r + k_mf + W)]
        ])
        
        # Steady State Vector M_eq (= M0 * [1-f, f] ?)
        # In absence of RF, M tends to thermal equilibrium [M0f, M0r].
        # The equation is dM/dt = R(M - M_eq) -- NO, that's not right for Exchange.
        # Bloch-McConnell: dM/dt = Lambda * M + M0*R1
        # Lambda includes Exchange and Transverse Decays (if any) and -R1.
        
        # Correct form:
        # d/dt [Mz_f] = R1f(M0f - Mz_f) - kfr Mz_f + kmf Mz_r
        # d/dt [Mz_r] = R1r(M0r - Mz_r) - kmf Mz_r + kfr Mz_f - W Mz_r
        
        # Matrix form: dM/dt = A M + B
        # A = [[-R1f-kfr, kmf], [kfr, -R1r-kmf-W]]
        # B = [[R1f M0f], [R1r M0r]]
        
        M0f = (1.0 - f)
        M0r = f
        B = jnp.array([R1f * M0f, R1r * M0r])
        
        # Evolution over TR (Saturation + Relaxation + Exchange)
        # E = exp(A * TR)
        # But Excitation is separate.
        # SPGR Sequence:
        # 1. Pulse (instant): Mz_f -> Mz_f * cos(alpha). Mz_r -> Mz_r (assumed unaffected by excitation if T2r is very short / alpha is on resonance).
        #    Wait, does On-Resonance alpha affect Mz_r? Usually R is affected by SaturationPulse, not ExcitationPulse (unless broad).
        #    Yarnykh assumes Mz_r is unaffected by excitation alpha.
        # 2. Evolution (TR): M(t) = exp(A*t) M(0) + (exp(At)-I)A^-1 (-B) ...
        #    M_end = E * M_start + (I - E) * M_ss_cw
        
        # Steady State Condition: M_start = M_end (cyclic)
        # M_start = Excitation * M_end
        # Let M_minus be magnetisation before pulse.
        # M_plus = Q * M_minus
        # Q = [[cos(alpha), 0], [0, 1]]
        # M_minus = E * M_plus + M_relax
        # M_minus = E Q M_minus + M_relax
        # (I - E Q) M_minus = M_relax
        # M_minus = inv(I - E Q) * M_relax
        
        # Calculate Matrix Exponential
        # For 2x2, we can use analytical or jax.scipy.linalg.expm
        E = jax.scipy.linalg.expm(R * tr)
        
        # Integrated recovery (M_relax)
        # M_relax = \int_0^T exp(A(T-t)) B dt = A^-1 (exp(AT) - I) B
        #        = A^-1 (E - I) B
        A_inv = jnp.linalg.inv(R)
        M_relax = A_inv @ (E - jnp.eye(2)) @ B
        
        # Excitation Matrix
        Q = jnp.array([
            [jnp.cos(exc_rad), 0.0],
            [0.0,              1.0] # Restricted pool not tipped? Or saturated effectively? Yarnykh assumes 1.0.
        ])
        
        # Solve for Steady State
        Mat = jnp.eye(2) - E @ Q
        M_ss_minus = jnp.linalg.solve(Mat, M_relax)
        
        # Signal is M_ss_minus_f * sin(alpha)
        # (SPGR signal is proportional to transverse magnetization after pulse)
        # M_plus = Q M_minus -> M_plus_f = M_minus_f * cos(a)
        # M_xy = M_minus_f * sin(a)
        
        S = M0f * jnp.sin(exc_rad) * M_ss_minus[0] # Scaling? M_ss_minus[0] is Mz_f.
        # Wait, M_ss_minus is already Mz.
        # And M_ss_minus IS the magnetization available to be tipped.
        # So S \propto M_ss_minus[0] * sin(exc_rad).
        # Note: M0f factor is included in B, so M_ss_minus is absolute magnetization.
        # We don't mult by M0f again unless extracting relative.
        
        return M_ss_minus[0] * jnp.sin(exc_rad)

