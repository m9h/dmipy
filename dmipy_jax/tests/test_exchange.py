import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.components.exchange import KargerExchange
from dmipy_jax.signal_models.sphere_models import SphereStejskalTanner
from dmipy_jax.signal_models.gaussian_models import G1Ball
from dmipy_jax.core.acquisition import JaxAcquisition

class TestKargerExchange:
    def test_initialization(self):
        """Test proper initialization and parameter management."""
        c1 = G1Ball()
        c2 = SphereStejskalTanner()
        karger = KargerExchange([c1, c2])
        
        # Check standard parameters
        assert "model0_diffusivity" in karger.parameter_names
        assert "model1_diameter" in karger.parameter_names
        
        # Check exchange logic
        # N=2 -> 1 fraction (partial_volume_0), 1 exchange time (exchange_time_01)
        assert "partial_volume_0" in karger.parameter_names
        assert "exchange_time_01" in karger.parameter_names
        assert karger.n_models == 2

    def test_prediction_basic(self):
        """Test prediction runs without error and returns reasonable signal."""
        c1 = G1Ball()
        c2 = SphereStejskalTanner()
        karger = KargerExchange([c1, c2])
        
        # Simple acquisition
        bvals = jnp.array([0.0, 1000.0, 3000.0])
        bvecs = jnp.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
        acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs, delta=0.01, Delta=0.02)
        
        # Construct params
        # model0_diffusivity: 2e-9
        # model1_diameter: 5e-6
        # partial_volume_0: 0.5
        # exchange_time_01: 1.0 (slow eq)
        params = jnp.array([2.0e-9, 5.0e-6, 0.5, 1.0])
        
        signal = karger.predict(params, acq)
        
        assert signal.shape == (3,)
        assert jnp.all(jnp.isfinite(signal))
        assert signal[0] > 0.99 # s(0) should be ~1
        
    def test_jit_compilation(self):
        """Ensure the predict function is JIT-compatible."""
        c1 = G1Ball()
        c2 = SphereStejskalTanner()
        karger = KargerExchange([c1, c2])
        
        bvals = jnp.array([1000.0])
        bvecs = jnp.array([[1.0, 0, 0]])
        acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs, delta=0.01, Delta=0.02)
        params = jnp.array([2.0e-9, 5.0e-6, 0.5, 0.2])
        
        jit_pred = jax.jit(karger.predict)
        res = jit_pred(params, acq)
        assert jnp.isfinite(res)

    def test_no_exchange_limit(self):
        """
        With extremely long exchange time (tau -> inf), implementation should limit 
        to sum of independent compartments: f1*S1 + f2*S2.
        """
        c1 = G1Ball()
        c2 = SphereStejskalTanner()
        karger = KargerExchange([c1, c2])
        
        bvals = jnp.array([2000.0])
        bvecs = jnp.array([[1.0, 0, 0]])
        acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs, delta=0.01, Delta=0.02)
        
        diff = 2e-9
        diam = 6e-6
        f1 = 0.6
        tau = 1e9 # Very long time -> No exchange
        
        params = jnp.array([diff, diam, f1, tau])
        
        signal_karger = karger.predict(params, acq)
        
        # Independent calculation
        s1 = c1(bvals=bvals, gradient_directions=bvecs, diffusivity=diff)
        s2 = c2(bvals=bvals, gradient_directions=bvecs, diameter=diam)
        signal_indep = f1 * s1 + (1 - f1) * s2
        
        # Should be very close
        assert jnp.allclose(signal_karger, signal_indep, atol=1e-5)

