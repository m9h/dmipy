import jax.numpy as jnp
from dmipy_jax.signal_models import g2_tensor

# Test Setups
bvals = jnp.array([1000.0, 3000.0])

# Gradients along X, Y, Z
bvecs_x = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
bvecs_y = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
bvecs_z = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

# Tensor Geometry
# e1 along X
e1 = jnp.array([1.0, 0.0, 0.0])
# e2 along Y
e2 = jnp.array([0.0, 1.0, 0.0])
# e3 will be Z (implicitly)

# Eigenvalues
L1 = 2.0e-3
L2 = 1.0e-3
L3 = 0.5e-3

# Test 1: Signal along Eigenvectors
# Should match exp(-b * Li)
print("Test 1: Signal along Principals")
s_x = g2_tensor(bvals, bvecs_x, L1, L2, L3, e1, e2)
print(f"X-Signal (L1={L1}): {s_x[0]:.4f} (Exp[-2] ~ 0.1353)")

s_y = g2_tensor(bvals, bvecs_y, L1, L2, L3, e1, e2)
print(f"Y-Signal (L2={L2}): {s_y[0]:.4f} (Exp[-1] ~ 0.3679)")

s_z = g2_tensor(bvals, bvecs_z, L1, L2, L3, e1, e2)
print(f"Z-Signal (L3={L3}): {s_z[0]:.4f} (Exp[-0.5] ~ 0.6065)")

# Test 2: Zeppelin Reduction
# if L2 == L3, it should behave like Zeppelin(mu=e1, par=L1, perp=L2)
# Checking Y vs Z symmetry
print("\nTest 2: Zeppelin Reduction (L2=L3=0.5e-3)")
s_y_zep = g2_tensor(bvals, bvecs_y, L1, 0.5e-3, 0.5e-3, e1, e2)
s_z_zep = g2_tensor(bvals, bvecs_z, L1, 0.5e-3, 0.5e-3, e1, e2)
print(f"Y Signal: {s_y_zep[0]:.4f}")
print(f"Z Signal: {s_z_zep[0]:.4f}")
if jnp.allclose(s_y_zep, s_z_zep):
    print("PASS: Axisymmetric (Zeppelin-like).")
else:
    print("FAIL: Symmetry broken.")
