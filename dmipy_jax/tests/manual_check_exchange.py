import jax
import jax.numpy as jnp
from dmipy_jax.components.exchange import KargerExchange
from dmipy_jax.signal_models.sphere_models import SphereStejskalTanner
from dmipy_jax.signal_models.gaussian_models import G1Ball
from dmipy_jax.core.acquisition import JaxAcquisition
import sys
import traceback

def check():
    with open("debug_log.txt", "w") as f:
        try:
            f.write("Initializing models...\n")
            c1 = G1Ball()
            c2 = SphereStejskalTanner()
            karger = KargerExchange([c1, c2])
            
            f.write("Checking parameters...\n")
            if "model0_diffusivity" not in karger.parameter_names:
                f.write("FAIL: Parameter names missing model0_diffusivity\n")
                return 1
                
            f.write("Setting up acquisition...\n")
            bvals = jnp.array([0.0, 1000.0, 3000.0])
            bvecs = jnp.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
            acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs, delta=0.01, Delta=0.02)
            
            params = jnp.array([2.0e-9, 5.0e-6, 0.5, 1.0])
            
            f.write("Predicting...\n")
            signal = karger.predict(params, acq)
            f.write(f"Signal shape: {signal.shape}\n")
            f.write(f"Signal values: {signal}\n")
            
            if not jnp.isfinite(signal).all():
                f.write("FAIL: Signal has NaNs/Infs\n")
                return 1
                
            f.write("SUCCESS\n")
            return 0
        except Exception as e:
            f.write(f"EXCEPTION: {e}\n")
            traceback.print_exc(file=f)
            return 1

if __name__ == "__main__":
    sys.exit(check())
