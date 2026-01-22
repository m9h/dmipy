
import abc
from typing import Sequence, Tuple, Union, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int

class Polynomial(eqx.Module):
    """Abstract base class for polynomial bases."""
    
    @abc.abstractmethod
    def evaluate(self, x: Float[Array, " batch"], order: int) -> Float[Array, " batch order+1"]:
        """
        Evaluates polynomials up to `order` at points x.
        Returns array of shape (batch, order+1) where [:, k] is P_k(x).
        """
        pass

    @abc.abstractmethod
    def quadrature(self, order: int) -> Tuple[Float[Array, " order"], Float[Array, " order"]]:
        """
        Returns quadrature points and weights for exact integration of polynomial of degree `2*order - 1`.
        """
        pass

class LegendrePolynomial(Polynomial):
    """
    Legendre polynomials for Uniform distributions on [-1, 1].
    
    Recurrence Relation:
    P_0(x) = 1
    P_1(x) = x
    (n+1) P_{n+1}(x) = (2n + 1) x P_n(x) - n P_{n-1}(x)
    """

    def evaluate(self, x: Float[Array, " batch"], order: int) -> Float[Array, " batch order+1"]:
        batch_size = x.shape[0]
        # P_0(x) = 1
        p0 = jnp.ones_like(x)
        if order == 0:
            return p0[:, None]
            
        # P_1(x) = x
        p1 = x
        
        values = [p0, p1]
        
        # Recurrence
        for n in range(1, order):
            # (n+1) P_{n+1} = (2n+1)x P_n - n P_{n-1}
            # P_{n+1} = ((2n+1)x P_n - n P_{n-1}) / (n+1)
            pn = values[-1]
            pn_minus_1 = values[-2]
            
            p_next = ((2 * n + 1) * x * pn - n * pn_minus_1) / (n + 1)
            values.append(p_next)
            
        return jnp.stack(values[:order+1], axis=1)

    def quadrature(self, order: int) -> Tuple[Float[Array, " order"], Float[Array, " order"]]:
        # Using numpy's legendre module for roots/weights as they are static constants usually
        # But here we want jax compatibility. 
        # For small orders we can use jax.numpy.roots on characteristic polynomial, 
        # or just hardcode/wrap scipy.special.roots_legendre.
        # Since this is usually done once or is static, we can use simple eigenvalue method (Golub-Welsch).
        
        n = order
        i = jnp.arange(n - 1)
        beta = 0.5 / jnp.sqrt(1.0 - (2.0 * (i + 1)) ** -2.0)
        idx = jnp.arange(n)
        diag = jnp.zeros(n)
        # Construct tridiagonal matrix
        J = jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
        
        # Eigenvalues and eigenvectors
        points, eigenvectors = jnp.linalg.eigh(J)
        
        # Weights
        weights = 2 * eigenvectors[0, :]**2
        
        return points, weights


class HermitePolynomial(Polynomial):
    """
    Probabilists' Hermite polynomials He_n(x) for Standard Normal distribution N(0, 1).
    Orthogonal with respect to weight w(x) = e^{-x^2/2}.
    
    Recurrence Relation:
    He_0(x) = 1
    He_1(x) = x
    He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
    """

    def evaluate(self, x: Float[Array, " batch"], order: int) -> Float[Array, " batch order+1"]:
        batch_size = x.shape[0]
        p0 = jnp.ones_like(x)
        if order == 0:
            return p0[:, None]
        
        p1 = x
        values = [p0, p1]
        
        for n in range(1, order):
            # He_{n+1} = x He_n - n He_{n-1}
            pn = values[-1]
            pn_minus_1 = values[-2]
            p_next = x * pn - n * pn_minus_1
            values.append(p_next)
            
        return jnp.stack(values[:order+1], axis=1)

    def quadrature(self, order: int) -> Tuple[Float[Array, " order"], Float[Array, " order"]]:
        # Golub-Welsch algorithm for Hermite (Probabilists')
        # Diagnonal is 0.
        # Off-diagonal beta_n = sqrt(n/2) ? No, that is for Physicists'.
        # For Probabilists' He_n: 
        # x P_n = P_{n+1} + n P_{n-1}
        # alpha_n = 0.
        # beta_n = sqrt(n). (coefficient of P_{n-1} in recurrence x P_n = ... + beta_n^2 P_{n-1} ???)
        # Recurrence: He_{n+1} = x He_n - n He_{n-1}  => x He_n = He_{n+1} + n He_{n-1}
        # Comparison with x p_n = a_n p_{n+1} + b_n p_n + c_n p_{n-1}.
        # Here p_n are monic. 
        # Jacobi matrix J has diagonal 0.
        # Off-diagonal J_{i, i+1} = sqrt(i+1) ... check.
        # Actually simplest to defer to numpy if inside JIT isn't required for constructing the grid.
        # But let's implementing Golub-Welsch. 
        # Recurrence: x He_n = He_{n+1} + n He_{n-1}.
        # With normalized polys \tilde{He}_n = He_n / sqrt(n!), we have
        # x \tilde{He}_n = \sqrt{n+1} \tilde{He}_{n+1} + \sqrt{n} \tilde{He}_{n-1}
        # The matrix J_{nm} = <x phi_n, phi_m>. 
        # J_{n, n-1} = sqrt(n). J_{n, n+1} = sqrt(n+1).
        
        n = order
        # i goes 1 to n-1
        beta = jnp.sqrt(jnp.arange(1, n)) 
        J = jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
        
        points, eigenvectors = jnp.linalg.eigh(J)
        
        # Weights: w_j = (first component of eigenvector)^2 * integral(w(x) dx)
        # Integral e^{-x^2/2} dx = sqrt(2pi)
        weights = eigenvectors[0, :]**2 * jnp.sqrt(2 * jnp.pi)
        
        return points, weights

class PolynomialChaosExpansion(eqx.Module):
    """
    Generalized Polynomial Chaos Expansion implementation.
    Evaluates a surrogate model based on gPC expansion.
    """
    coefficients: Float[Array, " num_terms"]
    basis_indices: Int[Array, " num_terms num_dims"]
    polynomials: Sequence[Polynomial]
    
    def __init__(
        self,
        coefficients: Float[Array, " num_terms"],
        basis_indices: Int[Array, " num_terms num_dims"],
        distributions: Sequence[str]
    ):
        """
        distributions: List of distribution names 'Uniform' or 'Normal' corresponding to each dimension.
        """
        self.coefficients = coefficients
        self.basis_indices = basis_indices
        
        self.polynomials = []
        for dist in distributions:
            if dist == 'Uniform':
                self.polynomials.append(LegendrePolynomial())
            elif dist == 'Normal':
                self.polynomials.append(HermitePolynomial())
            else:
                raise ValueError(f"Unknown distribution: {dist}. Supported: 'Uniform', 'Normal'")
    
    def __call__(self, parameters: Float[Array, " batch num_dims"]) -> Float[Array, " batch"]:
        """
        Evaluate the gPC expansion at given parameters.
        parameters: (batch, num_dims) array of inputs.
        """
        return self.evaluate(parameters)

    def evaluate(self, parameters: Float[Array, " batch num_dims"]) -> Float[Array, " batch"]:
        # parameters shape: (batch, num_dims)
        batch_size, num_dims = parameters.shape
        num_terms = self.coefficients.shape[0]
        
        if num_dims != len(self.polynomials):
            raise ValueError("Mismatch between input dimensions and defined polynomial bases.")
        
        # 1. Evaluate univariate polynomials for each dimension
        # We need the max order for each dimension
        max_orders = jnp.max(self.basis_indices, axis=0)
        
        # evaluations list will store (batch, max_order+1) for each dim
        univariate_evals = []
        for i, poly in enumerate(self.polynomials):
            # Evaluate polynomial i at parameters[:, i]
            # shape: (batch, max_orders[i] + 1)
            evals = poly.evaluate(parameters[:, i], max_orders[i].item())
            univariate_evals.append(evals)
            
        # 2. Compute multivariate basis terms
        # Term k is product of univariate evals: Phi_k(x) = prod_i phi_{k,i}(x_i)
        
        # We can construct the design matrix Psi of shape (batch, num_terms)
        # Psi[b, k] = product_{i=1}^d univariate_evals[i][b, basis_indices[k, i]]
        
        # Vectorized gather approach:
        # Collect all univariate evals into a specific structure?
        # Actually simple loop over terms or dimensions might be okay if not too many dimensions. 
        # But let's try to be vectorized.
        
        # Gather approach:
        # We want to form a tensor of shape (batch, num_terms)
        # Initialize with ones.
        term_values = jnp.ones((batch_size, num_terms))
        
        for dim_idx in range(num_dims):
            # Indices for this dimension for all terms: shape (num_terms,)
            degrees = self.basis_indices[:, dim_idx] 
            
            # Values for this dimension: shape (batch, max_order_dim + 1)
            vals = univariate_evals[dim_idx]
            
            # Gather: for each term, get the column from vals corresponding to degree
            # selected_vals shape: (batch, num_terms)
            selected_vals = vals[:, degrees]
            
            term_values = term_values * selected_vals
            
        # 3. Sum coefficients * basis terms
        # result shape: (batch,)
        return jnp.dot(term_values, self.coefficients)

    @classmethod
    def fit(
        cls,
        parameters: Float[Array, " num_samples num_dims"],
        values: Float[Array, " num_samples"],
        distributions: Sequence[str],
        total_order: int
    ) -> "PolynomialChaosExpansion":
        """
        Fits a gPC expansion using Least Squares Regression.
        
        Args:
            parameters: Input samples (N, D).
            values: Model evaluations at sampled points (N,).
            distributions: List of distribution names for each dimension.
            total_order: Maximum total polynomial order.
            
        Returns:
            Fitted PolynomialChaosExpansion instance.
        """
        num_samples, num_dims = parameters.shape
        
        # 1. Create basis indices
        basis_indices = cls.create_basis_indices(num_dims, total_order)
        num_terms = basis_indices.shape[0]
        
        if num_samples < num_terms:
            # Maybe warning? Or just let lstsq handle it (underdetermined).
            pass

        # 2. Initialize temporary PCE to evaluate basis terms
        # We need coefficients to be set to something to initialize the class, 
        # but here we just want to reuse the evaluation logic to build the design matrix.
        # However, the `evaluate` method computes dot product. We want the basis matrix.
        # Let's refactor `evaluate` to expose `compute_basis_matrix` or just implementation here.
        
        # Create polynomial objects
        polynomials = []
        for dist in distributions:
            if dist == 'Uniform':
                polynomials.append(LegendrePolynomial())
            elif dist == 'Normal':
                polynomials.append(HermitePolynomial())
            else:
                raise ValueError(f"Unknown distribution: {dist}")
        
        # Compute design matrix Psi (N, P)
        # Psi[i, j] = Phi_j(x_i)
        
        max_orders = jnp.max(basis_indices, axis=0)
        univariate_evals = []
        for i, poly in enumerate(polynomials):
            evals = poly.evaluate(parameters[:, i], max_orders[i].item())
            univariate_evals.append(evals) # (N, max_order_i + 1)
            
        # Gather terms
        # term_values = jnp.ones((num_samples, num_terms)) # This might be memory intensive for large N, P
        # Better to do it column by column or vectorized
        
        # Pre-allocate design matrix
        # For efficiency, we can use vmap? 
        # But indices look up is fast.
        
        # Let's reuse the logic from evaluate but return the matrix
        # Vectorized gather:
        # We need to construct Psi of shape (num_samples, num_terms)
        # Psi[:, k] = prod_d univariate_evals[d][:, basis_indices[k, d]]
        
        # Loop over dimensions to accumulate product
        Psi = jnp.ones((num_samples, num_terms))
        for d in range(num_dims):
            # univariate_evals[d] has shape (samples, max_order+1)
            # basis_indices[:, d] has shape (terms,)
            
            # We want to select columns from univariate_evals[d] based on basis_indices[:, d]
            # Resulting shape should be (samples, terms)
            
            # Use fancy indexing / gathering
            # evals_d: (samples, max_order+1)
            # indices_d: (terms,)
            # We want out: (samples, terms)
            # out[i, j] = evals_d[i, indices_d[j]]
            
            # This is effectively evals_d[:, indices_d]
            evals_d = univariate_evals[d]
            indices_d = basis_indices[:, d]
            
            # Check if this creates a copy or view. JAX handles it.
            columns = evals_d[:, indices_d] 
            
            Psi = Psi * columns
            
        # 3. Solve Least Squares: Psi * c = values
        coefficients, residuals, rank, s = jnp.linalg.lstsq(Psi, values)
        
        return cls(coefficients, basis_indices, distributions)

    @classmethod
    def create_basis_indices(cls, num_dims: int, total_order: int) -> Int[Array, " num_terms num_dims"]:
        """
        Creates basis indices for total order truncation.
        Sum of indices <= total_order.
        """
        # This is a classic combinatorics problem. 
        # For small dims and order, simple recursion is fine.
        indices = []
        
        # Use simple recursion
        def generate(current_index, remaining_order):
            if len(current_index) == num_dims:
                indices.append(current_index)
                return
            
            # Optimization: for last dimension, iterate up to remaining_order
            if len(current_index) == num_dims - 1:
                # Last dim can take any value from 0 to remaining_order
                for i in range(remaining_order + 1):
                    generate(current_index + [i], 0)
            else:
                for i in range(remaining_order + 1):
                    generate(current_index + [i], remaining_order - i)
                    
        generate([], total_order)
        return jnp.array(indices, dtype=jnp.int32)

