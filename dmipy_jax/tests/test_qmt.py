
import pytest
import jax
import jax.numpy as jnp
from dmipy_jax.models.qmt import QMTSPGR, QMTParameters
from dmipy_jax.models.epg import JAXEPG

class TestQMTSPGR:
    
    @pytest.fixture
    def params(self):
        return QMTParameters(
            f_bound=0.15,
            k_fb=2.0,       # 2 Hz exchange
            T1_f=1.0,       # 1s
            T2_f=0.050,     # 50ms
            T1_b=1.0,       # 1s (Similar to free)
            T2_b=0.000010   # 10us (Super short)
        )
    
    def test_zero_saturation_limit(self, params):
        """
        Test that when MT pulse is effectively off (0 degrees), 
        result matches analytical SPGR scaled by (1-f).
        
        CRITICAL: We must set k_fb=0 for this test, otherwise the Exchange
        acts as an additional T2 decay channel (transfer to invisible bound pool)
        which makes the signal decay faster than the standard 1-pool SPGR model.
        """
        # Create uncoupled params
        p_uncoupled = QMTParameters(
            f_bound=params.f_bound,
            k_fb=0.0, # NO EXCHANGE for validity check against 1-pool
            T1_f=params.T1_f, T2_f=params.T2_f,
            T1_b=params.T1_b, T2_b=params.T2_b
        )
        
        TR = 0.030 # 30ms
        exc_flip = 15.0 # degrees
        mt_flip = 0.0 # OFF
        mt_offset = 100000.0 # Far off resonance
        mt_dur = 0.010
        
        # Run QMT Model
        model = QMTSPGR()
        sig_qmt = model(
            p_uncoupled, TR, exc_flip, mt_flip, mt_offset, mt_pulse_duration=mt_dur, N_pulses=200
        )
        
        # Run Standard SPGR (Free Pool only)
        # Note: Standard SPGR simulates evolution of M0=1.
        # Free pool has M0 = (1-f).
        # T1, T2 are T1_f, T2_f.
        sig_std = JAXEPG.simulate_spgr(
            params.T1_f, params.T2_f, TR, jnp.deg2rad(exc_flip), N_pulses=200
        )
        
        expected = (1.0 - params.f_bound) * sig_std
        
        # Tolerance: Exchange might cause slight deviation due to numerical steps vs exact relaxation?
        # Also T2 decay during the MT pulse duration (10ms) is modeled in QMT loop
        # as R2+k_fb.
        # In Standard SPGR `simulate_spgr`, 'TR' includes everything.
        # My QMT loop does: Pulse(tau) -> Spoiler -> Readout -> Relax(TR-tau).
        # Wait, usually Excitation happens first?
        # SPGR: Pulse -> Readout -> Relax -> ... 
        # My QMT loop: MT(tau) -> Exc(0) -> Readout -> Relax.
        # This is MT-prepared SPGR.
        # Standard SPGR is Exc -> Readout -> Relax.
        # So "TR" in standard SPGR is time between Excitations.
        # In QMT loop, time between Excitations is tau + (TR-tau) = TR. Correct.
        # But QMT loop has T2 decay during tau (10ms) BEFORE excitation??
        # Usually MT pulse is played BEFORE excitation? YES.
        # But wait, in steady state does it matter where we start?
        # We start at equilibrium.
        # Step 1: MT Pulse (duration tau). Z decays/exchanges. F decays.
        # Step 2: Excitation.
        # So there is relaxation/exchange for `tau` BEFORE the first excitation?
        # Standard SPGR assumes relaxation for TR between excitations.
        # If W=0, QMT loop is: Relax(tau) -> Exc -> Readout -> Relax(TR-tau).
        # Total relaxation time = TR.
        # So Z recovery is same.
        # What about F? Spoiler kills F before excitation.
        # So F is fresh at Excitation.
        # So it should match perfectly.
        
        diff = jnp.abs(sig_qmt - expected)
        print(f"\nZero Saturation: QMT={sig_qmt:.6f}, Exp={expected:.6f}, Diff={diff:.6e}")
        assert diff < 1e-4

    def test_saturation_effect(self, params):
        """
        Verify that applying MT pulse reduces signal.
        """
        TR = 0.030
        exc_flip = 15.0
        
        model = QMTSPGR()
        
        # Ref (W=0)
        sig_ref = model(params, TR, exc_flip, mt_flip=0.0, mt_offset=1e5)
        
        # Saturation (W > 0)
        # Flip 500 deg, Offset 1kHz
        sig_sat = model(params, TR, exc_flip, mt_flip=500.0, mt_offset=1000.0)
        
        print(f"\nSaturation Effect: Ref={sig_ref:.6f}, Sat={sig_sat:.6f}")
        
        # MTR = (S0 - Ssat)/S0
        mtr = (sig_ref - sig_sat) / sig_ref
        print(f"MTR: {mtr:.2%}")
        
        assert sig_sat < sig_ref
        assert mtr > 0.1 # Expect significant effect

    def test_differentiability(self, params):
        """
        Verify JAX gradients w.r.t parameters.
        """
        model = QMTSPGR()
        
        def loss(p_vals):
            # p_vals = [f, k, T1f, T2f, T1b, T2b]
            p = QMTParameters(*p_vals)
            return model(p, 0.030, 15.0, 500.0, 1000.0)
            
        p_init = jnp.array([params.f_bound, params.k_fb, params.T1_f, params.T2_f, params.T1_b, params.T2_b])
        
        grads = jax.grad(loss)(p_init)
        print(f"\nGradients: {grads}")
        
        assert jnp.all(jnp.isfinite(grads))
        assert grads[0] != 0.0 # Gradient w.r.t f_bound should be non-zero
        assert grads[1] != 0.0 # w.r.t k_fb

if __name__ == "__main__":
    t = TestQMTSPGR()
    p = t.params()
    t.test_zero_saturation_limit(p)
    t.test_saturation_effect(p)
    t.test_differentiability(p)
    print("All tests passed manually.")
