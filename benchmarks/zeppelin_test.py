import jax.numpy as jnp
from dmipy_jax.signal_models import g2_zeppelin
from dmipy_jax.signal_models.zeppelin import Zeppelin

# Test Setups
bvals = jnp.array([1000.0, 2000.0, 3000.0])

# Gradients parallel to X
bvecs_par = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
# Gradients perpendicular to X (along Y)
bvecs_perp = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

mu = jnp.array([1.0, 0.0, 0.0]) # Fiber along X

lambda_par = 2.0e-3 # High diffusivity
lambda_perp = 0.5e-3 # Low diffusivity

# Test 1: Parallel
s_par = g2_zeppelin(bvals, bvecs_par, mu, lambda_par, lambda_perp)
print("Test 1: Parallel (Should reflect lambda_par=2.0e-3)")
print(f"B=1000 Signal: {s_par[0]:.4f} (Exp[-2] ~ 0.1353)")
print(f"B=3000 Signal: {s_par[-1]:.4f}")

# Test 2: Perpendicular
s_perp = g2_zeppelin(bvals, bvecs_perp, mu, lambda_par, lambda_perp)
print("\nTest 2: Perpendicular (Should reflect lambda_perp=0.5e-3)")
print(f"B=1000 Signal: {s_perp[0]:.4f} (Exp[-0.5] ~ 0.6065)")
print(f"B=3000 Signal: {s_perp[-1]:.4f}")

# Test 3: Isotropic Limit
s_iso = g2_zeppelin(bvals, bvecs_par, mu, 1.0e-3, 1.0e-3)
print("\nTest 3: Isotropic Limit (lambda_par = lambda_perp = 1.0e-3)")
print(f"B=1000 Signal: {s_iso[0]:.4f} (Exp[-1] ~ 0.3679)")

print("\n---------------------------------------------------")
print("Test 4: Equinox Module Wrapper (Zeppelin class)")
# Instantiate the module
zeppelin_mod = Zeppelin(mu=mu, lambda_par=lambda_par, lambda_perp=lambda_perp)
print(f"Module created: {zeppelin_mod}")
# Call it
s_mod = zeppelin_mod(bvals=bvals, gradient_directions=bvecs_par)
print(f"B=1000 Signal: {s_mod[0]:.4f} (Should match Test 1 Parallel: {s_par[0]:.4f})")
assert jnp.allclose(s_mod, s_par), "Module output mismatch!"
print("Module verification constant parameters PASSED")

# Call with overrides
s_mod_override = zeppelin_mod(bvals=bvals, gradient_directions=bvecs_perp, lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu)
print(f"Override call (Perpendicular) Signal: {s_mod_override[0]:.4f}")
assert jnp.allclose(s_mod_override, s_perp), "Module override mismatch!"
print("Module verification override parameters PASSED")

