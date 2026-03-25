
import jax
import jax.numpy as jnp
import optax
from jax.scipy.linalg import cho_factor, cho_solve
from dmipy_jax.utils.spherical_harmonics import sh_basis_real
from dmipy_jax.reconstruction.csd.classic.solvers import _get_response_matrix

class SpectralBayesianCSD:
    """
    Bayesian CSD (Agent 3).
    Models the FOD as a Gaussian Process on the Sphere, solved in the Spectral Domain.
    
    y = K c + epsilon
    c ~ N(0, Sigma_p(theta))
    epsilon ~ N(0, sigma_n^2 I)
    
    We optimize Marginal Likelihood w.r.t theta (lengthscale) and sigma_n.
    Then compute posterior p(c|y).
    """
    
    def __init__(self, lmax=8):
        self.lmax = lmax
        
    def angular_power_spectrum(self, lengthscale):
        """
        Matérn-like power spectrum for FOD.
        A_l ~ (1 + (l * lengthscale)^2 ) ^ -nu
        Smoothness prior.
        """
        ls = jnp.arange(0, self.lmax + 1, 2)
        # Repeat for m
        # l=0 (1), l=2 (5), ...
        
        # Matern-like spectral decay
        # A_l = exp(- l * l * lengthscale) ? (Heat kernel)
        # Using Heat Kernel (Squared Exponential equivalent) for simplicity and stability.
        # SE on sphere -> A_l ~ exp(-l(l+1) * ls^2 / 2)
        
        spectrum_vals = jnp.exp( - ls * (ls + 1.0) * (lengthscale**2) / 2.0 )
        
        # Construct diagonal
        diag = []
        for i, l in enumerate(range(0, self.lmax + 1, 2)):
             n_m = 2 * l + 1
             diag.extend([spectrum_vals[i]] * n_m)
             
        return jnp.array(diag)

    def marginal_likelihood(self, params, signal, K, inv_fn=jnp.linalg.pinv):
        """
        -log p(y | theta)
        y ~ N(0, K Sigma_p K.T + sigma_n^2 I)
        """
        lengthscale, noise_var = params
        lengthscale = jax.nn.softplus(lengthscale) # Ensure positive
        noise_var = jax.nn.softplus(noise_var)
        
        # Prior Covariance
        Sigma_p_diag = self.angular_power_spectrum(lengthscale) # (N_sh,)
        
        # Data Covariance: C = K diag(S_p) K.T + S_n I
        # K is (N_meas, N_sh).
        # Inner term: K_scaled = K * sqrt(S_p)
        # C = K_scaled @ K_scaled.T + S_n I
        
        K_scaled = K * jnp.sqrt(Sigma_p_diag)[None, :]
        C = jnp.dot(K_scaled, K_scaled.T) + noise_var * jnp.eye(K.shape[0])
        
        # Log Likelihood
        # L = -0.5 * (y.T C^-1 y + log|C| + N log 2pi)
        
        # Cholesky
        # Use simple solve for now (or robust Cholesky)
        # C is (N_meas, N_meas). N=60. Cheap.
        L = jnp.linalg.cholesky(C + 1e-6 * jnp.eye(C.shape[0]))
        
        # log|C| = 2 sum log diag(L)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        
        # y.T C^-1 y
        # solve L x = y then L.T z = x? No.
        # alpha = cho_solve((L, True), signal)
        # term = y.T alpha
        alpha = jax.scipy.linalg.cho_solve((L, True), signal)
        mahalanobis = jnp.dot(signal, alpha)
        
        nll = 0.5 * (mahalanobis + log_det + len(signal) * jnp.log(2*jnp.pi))
        return nll

    def predict(self, params, signal, K):
        """
        Compute Posterior Mean and Covariance of coeffs c.
        p(c|y) = N(mu_c, Sigma_c)
        """
        lengthscale, noise_var = params
        lengthscale = jax.nn.softplus(lengthscale)
        noise_var = jax.nn.softplus(noise_var)
        
        Sigma_p_diag = self.angular_power_spectrum(lengthscale)
        
        # Posterior Covariance
        # Sigma_c^-1 = K.T Sigma_n^-1 K + Sigma_p^-1
        # Sigma_c = (1/sn K.T K + diag(1/Sp))^-1
        
        N_sh = len(Sigma_p_diag)
        prior_prec = jnp.diag(1.0 / (Sigma_p_diag + 1e-9))
        
        H = (1.0 / noise_var) * jnp.dot(K.T, K) + prior_prec
        
        # Invert H (45x45)
        L_h = jnp.linalg.cholesky(H + 1e-6 * jnp.eye(N_sh))
        Sigma_c = jax.scipy.linalg.cho_solve((L_h, True), jnp.eye(N_sh))
        
        # Posterior Mean
        # mu_c = Sigma_c K.T Sigma_n^-1 y
        projected_y = (1.0 / noise_var) * jnp.dot(K.T, signal)
        mu_c = jnp.dot(Sigma_c, projected_y)
        
        return mu_c, Sigma_c, Sigma_p_diag

    def fit_and_predict(self, signal, bvecs, response_coeffs):
        # 1. Setup K
        # Acq Matrix
        r = jnp.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-8
        bvecs_norm = bvecs / r
        theta_acq = jnp.arccos(bvecs_norm[:, 2]) 
        phi_acq = jnp.arctan2(bvecs_norm[:, 1], bvecs_norm[:, 0])
        Y_acq = sh_basis_real(theta_acq, phi_acq, lmax=self.lmax)
        R_diag = _get_response_matrix(response_coeffs, self.lmax)
        K = Y_acq * R_diag[None, :] 
        
        # 2. Optimize Hyperparams (Lengthscale, Noise)
        optimizer = optax.adam(0.1)
        
        # params: [log_lengthscale, log_noise]
        init_params = jnp.array([-1.0, -2.0]) # LS=0.3, sigma=0.1
        opt_state = optimizer.init(init_params)
        
        @jax.jit
        def step(params, opt_st):
            loss, grads = jax.value_and_grad(self.marginal_likelihood)(params, signal, K)
            updates, opt_st = optimizer.update(grads, opt_st, params)
            params = optax.apply_updates(params, updates)
            return params, opt_st, loss
            
        # Optimization Loop (20 steps is enough usually for 2 params)
        params = init_params
        for _ in range(30):
            params, opt_state, loss = step(params, opt_state)
            
        # 3. Predict
        mu_c, Sigma_c, prior = self.predict(params, signal, K)
        
        # Return Mean and Uncertainty (Trace of Covariance or Diagonal)
        uncertainty = jnp.sqrt(jnp.diag(Sigma_c))
        
        return mu_c, uncertainty

