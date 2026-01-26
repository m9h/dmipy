import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Tuple, Optional

class EPGStates(NamedTuple):
    """
    State of the Extended Phase Graph.
    F_plus: Transverse magnetization (F+ states)
    F_minus: Transverse magnetization (F- states) - typically conjugate of F+
    Z: Longitudinal magnetization (Z states)
    """
    F_plus: jax.Array  # Shape (N_states,)
    F_minus: jax.Array # Shape (N_states,)
    Z: jax.Array       # Shape (N_states,)

class JAXEPG(eqx.Module):
    """
    Differentiable Extended Phase Graph (EPG) Simulator.
    
    References:
    - Weigel, M. (2015). Extended phase graphs: Dephasing, RF pulses, and echoes - pure and simple. JMRI.
    """
    
    @staticmethod
    def init_states(N_states: int) -> EPGStates:
        """Initialize EPG states at equilibrium (M0 along Z)."""
        F_plus = jnp.zeros(N_states, dtype=jnp.complex64)
        F_minus = jnp.zeros(N_states, dtype=jnp.complex64)
        Z = jnp.zeros(N_states, dtype=jnp.complex64)
        
        # Initial magnetization M0=1 is in Z_0 state
        Z = Z.at[0].set(1.0 + 0j)
        return EPGStates(F_plus, F_minus, Z)

    @staticmethod
    def relaxation(states: EPGStates, T1: float, T2: float, dt: float) -> EPGStates:
        """Apply T1 and T2 relaxation over time interval dt."""
        E1 = jnp.exp(-dt / T1)
        E2 = jnp.exp(-dt / T2)
        
        # F states decay with E2
        F_plus = states.F_plus * E2
        F_minus = states.F_minus * E2
        
        # Z states decay with E1, except Z0 which recovers to Equilibrium (1)
        Z = states.Z * E1
        
        # Recovery of Z0 state: Z0(t) = Z0(0)*E1 + (1-E1)
        # We add (1-E1) to the 0th component
        Z = Z.at[0].add(1.0 - E1)
        
        return EPGStates(F_plus, F_minus, Z)

    @staticmethod
    def rf_pulse(states: EPGStates, alpha: float, phi: float) -> EPGStates:
        """
        Apply RF pulse with flip angle alpha (radians) and phase phi (radians).
        Using the transition matrix defined in Weigel 2015.
        """
        # Rotation Matrix Elements
        # R = [ cos^2(a/2)   e^{2ip}sin^2(a/2)   -i e^{ip}sin(a) ]
        #     [ e^{-2ip}sin^2(a/2)  cos^2(a/2)   i e^{-ip}sin(a) ]
        #     [ -i/2 e^{-ip}sin(a)  i/2 e^{ip}sin(a)   cos(a)    ]
        
        # Note: Weigel uses a specific definition. Common implementation:
        ## R00 = cos(alpha/2)**2
        ## R01 = exp(2*1j*phi) * sin(alpha/2)**2
        ## R02 = -1j * exp(1j*phi) * sin(alpha)
        ## ... 
        
        # Optimized Implementation using rotation equations directly for vectorization
        
        c = jnp.cos(alpha / 2)
        s = jnp.sin(alpha / 2)
        c2 = c**2
        s2 = s**2
        
        exp_phi = jnp.exp(1j * phi)
        exp_neg_phi = jnp.exp(-1j * phi)
        
        Fp = states.F_plus
        Fm = states.F_minus
        Z  = states.Z
        
        # Update Equations
        # F_k+ = c2 * F_k+  +  e^{2ip} s2 * F_k-  +  -i e^{ip} sin(alpha) * Z_k
        Fp_new = c2 * Fp + (exp_phi**2 * s2) * Fm + (-1j * exp_phi * jnp.sin(alpha)) * Z
        
        # F_k- = e^{-2ip} s2 * F_k+  +  c2 * F_k-  +  i e^{-ip} sin(alpha) * Z_k
        Fm_new = (exp_neg_phi**2 * s2) * Fp + c2 * Fm + (1j * exp_neg_phi * jnp.sin(alpha)) * Z
        
        # Z_k = -i/2 e^{-ip} sin(alpha) * F_k+  +  i/2 e^{ip} sin(alpha) * F_k-  +  cos(alpha) * Z_k
        # Note: Factor of 1/2 comes from Z definition? Check Weigel. 
        # Actually standard Z state is just Mz. EPG papers vary on factor of 2.
        # Weigel Eq [12]:
        # F+_k = cos^2(a/2) F+_k + e^{2i\phi} sin^2(a/2) F-_k - i e^{i\phi} sin(a) Z_k
        # F-_k = e^{-2i\phi} sin^2(a/2) F+_k + cos^2(a/2) F-_k + i e^{-i\phi} sin(a) Z_k
        # Z_k = -i/2 e^{-i\phi} sin(a) F+_k + i/2 e^{i\phi} sin(a) F-_k + cos(a) Z_k
        
        Z_new = (-0.5j * exp_neg_phi * jnp.sin(alpha)) * Fp + \
                ( 0.5j * exp_phi * jnp.sin(alpha)) * Fm + \
                (jnp.cos(alpha)) * Z
                
        return EPGStates(Fp_new, Fm_new, Z_new)

    @staticmethod
    def shift(states: EPGStates) -> EPGStates:
        """
        Apply Gradient Shift (Dephasing).
        F_k+ -> F_{k+1}+
        F_k- -> F_{k-1}-
        Z_k  -> Z_k
        """
        # F_plus shifts to higher orders: [F0, F1, F2] -> [0, F0, F1]
        # But wait, F0+ is special. F0+ comes from F(-1)+ ? 
        # Standard shift:
        # F_new[k] = F_old[k-1]
        # F0 comes from F(-1). Since F(-k) = conj(F(k)), F(-1) is conj(F(1)-).
        # Actually in EPG basis F- states represent negative orders.
        # F-_k is order -k.
        # Shift increases order by 1.
        # +k -> +(k+1)
        # -k -> -(k-1)
        
        Fp = states.F_plus
        Fm = states.F_minus
        
        # Shift Fp right: [0, Fp[0], Fp[1]...]
        Fp_new = jnp.concatenate([jnp.array([Fm[0].conj()]), Fp[:-1]]) 
        # Note: F(0) state usually handled carefully. 
        # F0+ -> F1+
        # F0- -> F(-1)- which is F1*
        
        # Let's use the explicit indices logic which is safer for implementation.
        # F+_new[k] = F+[k-1] for k>=1. F+_new[0] = F-[0].conj() ??
        # F-_new[k] = F-[k+1] for k>=0.
        
        # Correct shift logic for dephasing +1:
        # F_k+ comes from k-1. For k=0, it comes from -1. F(-1) = conj(F(1)-) ?? 
        # No, typically F- array stores F_{k}^{-} corresponding to order -k.
        # So we have separate positive and negative banks.
        # Order 0 is shared? Typically Z0 is separate. F0 is just Mxy (transverse).
        # Let's assume F_plus[k] stores order +k, F_minus[k] stores order -k.
        # Shift +1:
        # Order p becomes p+1.
        # F_new(+k) = F_old(k-1). For k=0 (order 0), source is order -1. Order -1 is F_minus[1] ??
        # F_new(-k) = F_old(-(k+1)). i.e. F_minus_new[k] = F_minus_old[k+1].
        
        # F_plus shift
        # F_plus[0] (order 0) <- F_minus[1] (order -1). Wait, order -1 is F_minus[1].
        # We need to account that F_plus[0] and F_minus[0] are typically the same state (Mxy).
        # But in Weigel notation they are distinct pre-rotation states?
        # Usually we enforce F_minus[0] = conj(F_plus[0]).
        
        # Standard implementation (e.g. MRTRIX/Pysim):
        # FpF = [Fm[0].conj(), Fp[0], Fp[1], ...] (truncated)
        # FmF = [Fm[1], Fm[2], ..., 0]
        
        Fp_new = jnp.concatenate([jnp.array([states.F_plus[0].conj()]), states.F_plus[:-1]]) # Basic approximation if F-[0]=F+[0]*
        # Actually correct is: Source of +0 is -1. 
        # Explicitly:
        # F_plus_new[1:] = F_plus[:-1]
        # F_plus_new[0] = F_minus[1].conj() ?? No, coherence F(-1) is conj(F(1)).
        # Actually, let's look at Weigel Fig 3.
        # Gradient Dk=1:
        # F0 -> F1
        # F-1 -> F0
        # F-2 -> F-1
        
        # Mapping to our arrays:
        # F_plus[k] is +k. F_minus[k] is -k.
        # F_plus_new[k] (for k>=1) = F_plus[k-1]
        # F_plus_new[0] (order 0) = F_minus[1]  <-- This is the crossover
        # F_minus_new[k] (for k>=1) = F_minus[k+1]
        
        Fp_new = jnp.concatenate([jnp.array([states.F_minus[1]]), states.F_plus[:-1]]) # wait, size match?
        # If N=3: [Fp0, Fp1, Fp2] -> [Fm1, Fp0, Fp1]
        # [Fm0, Fm1, Fm2] -> [ ??, Fm2, 0 ]
        # Fm0 (order 0) should be same as Fp0? 
        # The new state at order 0 comes from -1. 
        # So F_plus_new[0] is correct.
        # F_minus_new[0] same as F_plus_new[0]? 
        # F_minus_new[0] is order 0. Source is order 1? No, shift is +1.
        # Source of 0 was -1.
        # So F_minus_new[0] should be... well F states map to specific pathways.
        # Let's stick to generating F_plus and F_minus arrays properly.
        
        Fp_shifted = jnp.concatenate([jnp.expand_dims(states.F_minus[1],0), states.F_plus[:-1]])
        Fm_shifted = jnp.concatenate([states.F_minus[1:], jnp.array([0.+0.j])])
        
        # Wait, F_minus[0] is order 0. F_minus[1] is order -1.
        # F_new(Order -k) = F_old(Order -k - 1). 
        # New order -1 (Fm_new[1]) comes from Old order -2 (Fm[2]). Correct.
        # New order 0 (Fm_new[0]) comes from Old order -1 (Fm[1]).
        # So Fm_shifted[0] should be Fm[1]. 
        # My concatenation `Fm_shifted = concatenate([Fm[1:]...])` puts Fm[1] at index 0. Correct.
        
        # Does Fp_shifted[0] match Fm_shifted[0]?
        # Fp_shifted[0] = Fm[1]. 
        # Fm_shifted[0] = Fm[1].
        # Yes. Consistency maintained for order 0.
        
        # What about conjugate symmetry?
        # F(-k) = conj(F(k)) is NOT generally true during the sequence, only if started that way and symmetric pulses?
        # Actually EPG typically tracks both if phases are complex.
        
        return EPGStates(Fp_shifted, Fm_shifted, states.Z)

    @staticmethod
    def twist(states: EPGStates, phi: float) -> EPGStates:
        """
        Apply phase twist (Z-rotation) for off-resonance precession.
        Equivalent to a rotation of angle phi around Z axis.
        
        F_k+ -> F_k+ * e^{i phi}
        F_k- -> F_k- * e^{-i phi}
        Z_k  -> Z_k
        """
        E = jnp.exp(1j * phi)
        E_conj = jnp.exp(-1j * phi)
        
        return EPGStates(
            states.F_plus * E,
            states.F_minus * E_conj,
            states.Z
        )
    
    @staticmethod
    def simulate_spgr(T1: float, T2: float, TR: float, alpha: float, 
                      N_pulses: int = 200, N_states: int = 50, 
                      rf_spoiling: bool = True) -> jax.Array:
        """
        Simulate Spoiled Gradient Echo (SPGR) to Steady State using EPG.
        
        Args:
            rf_spoiling: If True, applies quadratic phase cycling (117 deg increment)
                         to destroy transverse coherence, mimicking ideal SPGR.
        
        Returns:
            Signal magnitude (Mxy) at steady state. 
            Note: Returns magnitude as Signal is phase-demodulated.
        """
        # Initialize
        init_state = JAXEPG.init_states(N_states)
        
        # Initial Carry: (State, current_phase, phase_increment)
        # Phase increment usually constant 117? No, Phi[n] = Phi[n-1] + n*117
        # Or Phi[n] = 0.5 * 117 * n^2 + ...
        # Recursive: phi_new = phi + inc; inc_new = inc + 117
        
        init_carry = (init_state, 0.0, 0.0) 
        
        SPOIL_INC = jnp.deg2rad(117.0)
        
        def scan_fn(carry, _):
            current_state, phi, inc = carry
            
            # 1. Update Phase
            # For 1st pulse (n=0): Phase 0.
            # Then relax.
            
            # Apply RF
            state_rf = JAXEPG.rf_pulse(current_state, alpha, phi)
            
            # Extract Signal (F0+)
            # Demodulate: S_demod = S * exp(-i phi)
            sig_demod = state_rf.F_plus[0] * jnp.exp(-1j * phi)
            
            # 2. Relaxation + Spoiling
            state_rel = JAXEPG.relaxation(state_rf, T1, T2, TR)
            state_shifted = JAXEPG.shift(state_rel)
            
            # 3. Update Phase Logic
            if rf_spoiling:
                inc_new = inc + SPOIL_INC
                phi_new = phi + inc_new
            else:
                phi_new = phi
                inc_new = inc
                
            return (state_shifted, phi_new, inc_new), sig_demod

        # Run for N_pulses
        (final_carry, _, _), signals = jax.lax.scan(scan_fn, init_carry, jnp.arange(N_pulses))
        
        # Return the magnitude of the last signal to be robust
        # Usually for SPGR we just want the amplitude |S|
        return jnp.abs(signals[-1])

    @staticmethod
    def simulate_bssfp(T1: float, T2: float, TR: float, alpha: float, 
                       off_resonance: float = 0.0, phase_cycling: float = 0.0,
                       TE: Optional[float] = None,
                       N_pulses: int = 200, N_states: int = 10) -> jax.Array:
        """
        Simulate Balanced SSFP (bSSFP) using EPG.
        
        Args:
            off_resonance: Precession angle per TR (in radians) due to B0.
            phase_cycling: RF phase increment per TR (e.g. pi for 0-180 alternation).
            TE: Echo Time. If None, signal is measured immediately after RF (TE=0).
                If provided, relaxation and precession occur for time TE before measurement,
                and then for (TR-TE) after.
        """
        # Initialize
        init_state = JAXEPG.init_states(N_states)
        
        # If TE is provided, define split intervals
        if TE is not None:
            dt1 = TE
            dt2 = TR - TE
            # Fraction of off-resonance per interval
            # off_res is angle per TR. Angle(t) = off_res * (t/TR)
            phi1 = off_resonance * (dt1 / TR)
            phi2 = off_resonance * (dt2 / TR)
        else:
            phi_full = off_resonance
        
        def scan_fn(carry, idx):
            current_state = carry
            
            # RF Phase
            rf_phi = phase_cycling * idx 
            
            # 1. RF Pulse
            state_rf = JAXEPG.rf_pulse(current_state, alpha, rf_phi)
            
            if TE is not None:
                # Evolve to TE
                state_te = JAXEPG.relaxation(state_rf, T1, T2, dt1)
                state_te = JAXEPG.twist(state_te, phi1)
                
                # Measure Signal
                sig = state_te.F_plus[0] * jnp.exp(-1j * rf_phi)
                
                # Evolve rest of TR
                state_rel = JAXEPG.relaxation(state_te, T1, T2, dt2)
                state_final = JAXEPG.twist(state_rel, phi2)
            else:
                # Measure Signal (TE=0)
                sig = state_rf.F_plus[0] * jnp.exp(-1j * rf_phi)
                
                # Evolve TR
                state_rel = JAXEPG.relaxation(state_rf, T1, T2, TR)
                state_final = JAXEPG.twist(state_rel, phi_full)
            
            return state_final, sig

        final_state, signals = jax.lax.scan(scan_fn, init_state, jnp.arange(N_pulses).astype(float))
        
        return signals[-1]
