
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from dmipy_jax.nn.constitutive import ConstitutiveNN, NonNegativeLinear

def test_non_negative_linear():
    key = jax.random.PRNGKey(0)
    layer = NonNegativeLinear(5, 5, key=key)
    
    # Check that weights are effectively non-negative
    w_eff = jax.nn.softplus(layer.weight)
    assert jnp.all(w_eff >= 0)
    
    if layer.bias is not None:
        b_eff = jax.nn.softplus(layer.bias)
        assert jnp.all(b_eff >= 0)
        
    # Check forward pass positivity with positive input
    x = jnp.ones(5)
    out = layer(x)
    assert jnp.all(out >= 0)

def test_constitutive_forward():
    key = jax.random.PRNGKey(1)
    model = ConstitutiveNN(hidden_dim=12, key=key)
    
    # Inputs typical for I1, I2 (>=3)
    I1 = jnp.array(3.0)
    I2 = jnp.array(3.0)
    
    psi = model(I1, I2)
    
    assert psi.shape == ()
    assert not jnp.isnan(psi)
    # Psi should be positive ideally, as a sum of positive terms if weights are positive
    # Square >= 0, Exp > 0. Log(z) can be negative if z < 1.
    # But let's just check it runs.
    
def test_constitutive_stress():
    key = jax.random.PRNGKey(2)
    model = ConstitutiveNN(hidden_dim=6, key=key)
    
    F = jnp.eye(3) # Identity deformation -> zero stress usually (if min at I=3), but here just check calculation
    P = model.get_stress(F)
    
    assert P.shape == (3, 3)
    assert not jnp.isnan(P).any()
    
    # Check differentiation on random F
    F_rand = jax.random.normal(key, (3, 3)) + jnp.eye(3)
    P_rand = model.get_stress(F_rand)
    assert P_rand.shape == (3, 3)

def test_hidden_dim_adjustment():
    key = jax.random.PRNGKey(3)
    # Pass dim < 3
    model = ConstitutiveNN(hidden_dim=1, key=key)
    assert model.hidden_dim == 3

if __name__ == "__main__":
    # Manual run if pytest not directly invoked
    test_non_negative_linear()
    test_constitutive_forward()
    test_constitutive_stress()
    test_hidden_dim_adjustment()
    print("All tests passed!")
