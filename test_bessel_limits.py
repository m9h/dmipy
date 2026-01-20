
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

jax.config.update("jax_debug_nans", True)

def run():
    print("Testing bessel_jn limits (transition)")
    
    # Test values
    zs = jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    
    print("Individual checks (v=1):")
    for z in zs:
        print(f"Checking {z}...")
        try:
            r = jsp.bessel_jn(v=1, z=jnp.array([z]))
            print(f"  OK: val={r[1]}")
        except Exception as e:
            print(f"  FAIL: {e}")

if __name__ == "__main__":
    run()
