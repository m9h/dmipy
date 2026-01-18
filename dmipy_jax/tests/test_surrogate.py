
import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.core.surrogate import LegendrePolynomial, HermitePolynomial, PolynomialChaosExpansion

def test_legendre_polynomial_evaluation():
    poly = LegendrePolynomial()
    # Test values at x=0.5
    x = jnp.array([0.5])
    # P0(0.5) = 1
    # P1(0.5) = 0.5
    # P2(0.5) = 0.5 * (3 * 0.5**2 - 1) = 0.5 * (0.75 - 1) = -0.125
    
    values = poly.evaluate(x, order=2)
    assert jnp.allclose(values[0, 0], 1.0)
    assert jnp.allclose(values[0, 1], 0.5)
    assert jnp.allclose(values[0, 2], -0.125)

def test_hermite_polynomial_evaluation():
    poly = HermitePolynomial()
    # Probabilists' Hermite polynomials He_n(x)
    # He0(x) = 1
    # He1(x) = x
    # He2(x) = x^2 - 1
    # He3(x) = x^3 - 3x
    
    x = jnp.array([2.0])
    values = poly.evaluate(x, order=3)
    
    assert jnp.allclose(values[0, 0], 1.0)
    assert jnp.allclose(values[0, 1], 2.0)
    assert jnp.allclose(values[0, 2], 2.0**2 - 1) # 3.0
    assert jnp.allclose(values[0, 3], 2.0**3 - 3*2.0) # 8 - 6 = 2.0

def test_legendre_orthogonality():
    poly = LegendrePolynomial()
    order = 5
    # Use quadrature to integrate P_i P_j
    points, weights = poly.quadrature(order * 2) # High enough order for exact integration
    
    # Evaluate polynomials at quadrature points
    # evaluate expects (batch, ...)
    evals = poly.evaluate(points, order=order)
    
    # Compute inner products matrix: M_ij = sum_k w_k P_i(x_k) P_j(x_k)
    # evals shape: (num_points, order+1)
    # weights shape: (num_points,)
    
    weighted_evals = evals * weights[:, None]
    gram_matrix = jnp.dot(evals.T, weighted_evals)
    
    # Diagonal should be normalization constants N_n = 2 / (2n+1)
    expected_diag = 2.0 / (2 * jnp.arange(order + 1) + 1)
    
    diag = jnp.diag(gram_matrix)
    off_diag = gram_matrix - jnp.diag(diag)
    
    assert jnp.allclose(diag, expected_diag, atol=1e-5)
    assert jnp.allclose(off_diag, 0.0, atol=1e-5)

def test_pce_fit_legendre_1d():
    # Fit y = x^2 on [-1, 1]
    # x^2 = 2/3 P_2(x) + 1/3 P_0(x)
    
    # Generate data
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(100, 1), minval=-1.0, maxval=1.0)
    y = x[:, 0]**2
    
    surrogate = PolynomialChaosExpansion.fit(
        parameters=x,
        values=y,
        distributions=['Uniform'],
        total_order=2
    )
    
    # Coefficients:
    # Order 0 (index 0): 1/3
    # Order 1 (index 1): 0
    # Order 2 (index 2): 2/3
    
    # Note: basis indices order might vary, need to check corresponding indices
    indices = surrogate.basis_indices
    coeffs = surrogate.coefficients
    
    # Find coefficient for degree 0
    idx0 = jnp.where((indices == 0).all(axis=1))[0]
    assert len(idx0) == 1
    assert jnp.allclose(coeffs[idx0], 1.0/3.0, atol=1e-2)
    
    # Find coefficient for degree 2
    idx2 = jnp.where((indices == 2).all(axis=1))[0]
    assert len(idx2) == 1
    assert jnp.allclose(coeffs[idx2], 2.0/3.0, atol=1e-2)
    
    # Verify prediction
    pred = surrogate(x)
    mse = jnp.mean((pred - y)**2)
    assert mse < 1e-4

def test_pce_fit_multivariate():
    # Fit y = x1^2 + x2
    # dimensions: x1 (Uniform), x2 (Normal)
    # x1 on [-1, 1], x2 on R
    # y = (2/3 P2(x1) + 1/3 P0(x1)) + He1(x2)
    
    key = jax.random.PRNGKey(42)
    x1 = jax.random.uniform(key, shape=(200, 1), minval=-1.0, maxval=1.0)
    x2 = jax.random.normal(key, shape=(200, 1))
    X = jnp.hstack([x1, x2])
    
    y = x1[:, 0]**2 + x2[:, 0]
    
    surrogate = PolynomialChaosExpansion.fit(
        parameters=X,
        values=y,
        distributions=['Uniform', 'Normal'],
        total_order=2
    )
    
    preds = surrogate(X)
    mse = jnp.mean((preds - y)**2)
    assert mse < 1e-4
    
    # Check coefficient for x2 (He1(x2) * P0(x1)) => degrees [0, 1]
    # Check degree [0, 1]
    indices = surrogate.basis_indices
    coeffs = surrogate.coefficients
    
    # Term corresponding to x2: degrees [0, 1] (since x1 is dim 0, x2 is dim 1)
    mask = (indices[:, 0] == 0) & (indices[:, 1] == 1)
    idx_x2 = jnp.where(mask)[0]
    
    assert len(idx_x2) == 1
    # Coefficient should be 1.0
    assert jnp.allclose(coeffs[idx_x2], 1.0, atol=1e-2)

