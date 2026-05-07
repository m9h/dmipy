import jax
import jax.numpy as jnp
import jax.random as jr
from dmipy_jax.morphogenesis.cann import MorphoCANN

def test_morpho_cann_identity():
    """Verify that stress and energy are zero at identity deformation."""
    key = jr.PRNGKey(0)
    model = MorphoCANN(n_basis=4, key=key)
    
    F = jnp.eye(3)
    a0 = jnp.array([1.0, 0.0, 0.0])
    
    psi = model.psi(F, a0)
    P = model.first_piola_stress(F, a0)
    
    print(f"Energy at identity: {psi}")
    print(f"Stress at identity:\n{P}")
    
    assert jnp.allclose(psi, 0.0, atol=1e-5)
    assert jnp.allclose(P, 0.0, atol=1e-5)

def test_morpho_cann_stiffness():
    """Verify that stretching increases energy and stress."""
    key = jr.PRNGKey(0)
    model = MorphoCANN(n_basis=4, key=key)
    
    # Simple tension along X
    F = jnp.diag(jnp.array([1.1, 1.0, 1.0]))
    a0 = jnp.array([1.0, 0.0, 0.0])
    
    psi = model.psi(F, a0)
    P = model.first_piola_stress(F, a0)
    sigma = model.cauchy_stress(F, a0)
    
    print(f"Energy at 10% stretch: {psi}")
    print(f"Cauchy Stress at 10% stretch:\n{sigma}")
    
    assert psi > 0.0
    assert sigma[0, 0] > 0.0
    # Cross terms should be small/zero for uniaxial tension in isotropic/aligned case
    assert jnp.allclose(sigma[0, 1], 0.0, atol=1e-5)

if __name__ == "__main__":
    test_morpho_cann_identity()
    test_morpho_cann_stiffness()
    print("All tests passed!")
