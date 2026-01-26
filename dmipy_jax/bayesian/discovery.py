import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from typing import Optional, Dict, Any, List
import equinox as eqx

from dmipy_jax.models.super_tissue_model import SuperTissueModel
# We assume SuperTissueModel is available.
# We also need JaxAcquisition but it will be passed to the model.

class BayesianDiscovery:
    """
    Automated Model Discovery using Bayesian Inference / Sparse Regression.
    
    Uses a Regularized Horseshoe Prior to induce sparsity on compartment volume fractions.
    """
    
    def __init__(self, acquisition_model=None):
        if acquisition_model is None:
            self.super_model = SuperTissueModel()
        else:
            self.super_model = acquisition_model
            
        self.param_metadata = {
            'names': self.super_model.parameter_names,
            'ranges': self.super_model.parameter_ranges,
            'cardinality': self.super_model.parameter_cardinality
        }
        
    def _regularized_horseshoe(self, name: str, size: int, tau0: float = 1e-3):
        """
        Implements the Regularized Horseshoe Prior for sparsity.
        
        Args:
            name: Name of the parameter site.
            size: Number of dimensions (covariates).
            tau0: Global shrinkage parameter guess.
        
        Returns:
            The sampled weights (positive).
        """
        # Global shrinkage
        # tau ~ HalfCauchy(tau0) - heuristic usually tau0 is small
        tau = numpyro.sample(f"{name}_tau", dist.HalfCauchy(tau0))
        
        # Local shrinkage
        # lambda ~ HalfCauchy(1)
        lam = numpyro.sample(f"{name}_lambda", dist.HalfCauchy(1.0), sample_shape=(size,))
        
        # Slab regularization (c^2)
        # Prevents the "tails" from being too heavy for large signals
        c2 = numpyro.sample(f"{name}_c2", dist.InverseGamma(2.0, 8.0)) # Slab w/ roughly Normal(0, 2)
        
        # Reparameterization of the weights
        # z ~ Normal(0, 1)
        z = numpyro.sample(f"{name}_z", dist.Normal(0.0, 1.0), sample_shape=(size,))
        
        # Calculate Horseshoe scales
        # kappa = 1 / (1 + lam^2 * tau^2)
        # But we use the regularized version:
        # lambda_tilde = sqrt( (c^2 * lam^2 * tau^2) / (c^2 + lam^2 * tau^2) )
        # Using the standard horseshoe algebra for stable computation
        
        lt = (tau * lam)
        
        # Robust calculation: Clamp lt to avoid overflow in square
        # If lt is large, lambda_tilde approaches c.
        # 1e4^2 = 1e8, plenty large given c^2 approx 4-100.
        lt_safe = jnp.clip(lt, max=1e4)
        
        lt_sq = jnp.square(lt_safe)
        
        final_sq = (c2 * lt_sq) / (c2 + lt_sq)
        lambda_tilde = jnp.sqrt(final_sq)
        
        # We need POSITIVE weights for volume fractions.
        # Standard horseshoe is for regression coefficients beta \in Real.
        # Here we model w = |beta| or use HalfNormal wrapper?
        # A common approach for sparse non-negative:
        # beta ~ HS(), w = |beta|.
        # Or w ~ HalfNormal(scale=lambda_tilde)
        
        # Let's use w = lambda_tilde * |z|
        w = lambda_tilde * jnp.abs(z)
        
        # Deterministic site for tracking
        numpyro.deterministic(f"{name}_weights", w)
        
        return w

    def model(self, bvals, bvecs, signal=None, acquisition_kwargs=None):
        """
        NumPyro model definition.
        
        Args:
            bvals: Array of b-values (s/m^2 SI units usually required).
            bvecs: Array of gradients.
            signal: Observed signal (optional).
            acquisition_kwargs: Dict with 'delta', 'Delta' etc.
        """
        
        # 1. Sample Internal Parameters
        # Iterate over parameters excluding 'fraction'
        
        param_vector_parts = []
        fraction_count = 0
        
        # Build acquisition object if needed, or pass raw if model supports it.
        # SuperTissueModel expects an acquisition object or kwargs compatible with compose_models.
        # The simplest is to pass a Mock object or just kwargs if the underlying models handle it.
        # But SuperTissueModel.__call__ signature is (params, acquisition).
        # We should probably mock the acquisition structure here to match expected API.
        
        # We need to reconstruct the flat parameter vector in the correct order.
        full_param_vector = []
        
        # Separate fraction params from model params
        model_p_names = [n for n in self.param_metadata['names'] if not n.startswith('fraction')]
        fraction_names = [n for n in self.param_metadata['names'] if n.startswith('fraction')]
        
        model_params_values = []
        
        for name in model_p_names:
            ranges = self.param_metadata['ranges'][name]
            cardinality = self.param_metadata['cardinality'][name]
            
            # Constraints for internal parameters
            # Use Uniform within bounds or TruncatedNormal
            # Vanishing gradient strategy: Even if weight is 0, these must be sampled.
            # We use meaningful priors.
            
            if cardinality == 1:
                low, high = ranges
                # Uninformative prior within bounds
                val = numpyro.sample(name, dist.Uniform(low, high))
                model_params_values.append(val.reshape(1))
            else:
                # Vector parameter (e.g. mu)
                # Structure detected: ranges is list of ranges per dimension
                # ranges[k] = (low, high)
                vec_parts = []
                for k in range(cardinality):
                    dim_range = ranges[k]
                    low_k = float(dim_range[0])
                    high_k = float(dim_range[1])
                    val_k = numpyro.sample(f"{name}_{k}", dist.Uniform(low_k, high_k))
                    vec_parts.append(val_k)
                
                # Stack them back to (cardinality,)
                val_vec = jnp.stack(vec_parts)
                model_params_values.append(val_vec)
                
        # 2. Sample Weights (Fractions) using Horseshoe
        n_compartments = len(fraction_names)
        
        # Sparsity on the SIMPLEX is tricky.
        # Standard Multi-Compartment models require sum(w) = 1 (or <= 1).
        # Discovery usually allows sum(w) free ? Or sum(w) <= 1?
        # If we use Stick + Ball + ... usually we want sum(w) approx 1 if normalized signal.
        # "Bayesian Lasso" or "Bayesian Horseshoe" is for unconstrained regression usually.
        # We can normalize them: w_norm = w / sum(w).
        # But then sparsity on w translates to sparsity on w_norm.
        
        # Strategy:
        # Sample raw weights from Horseshoe.
        raw_weights = self._regularized_horseshoe("tissue_fractions", n_compartments, tau0=0.1)
        
        # Enforce Sum-to-One or Sum-Linear signal?
        # If S0 is 1.0 (normalized), then sum(w) should be 1.0.
        # But discovery means "some weights are zero".
        # If we enforce sum(w)=1, one compartment must take up the slack.
        # Let's assume we model S = sum(w_i S_i).
        # We can add a constraint/penalty or just let sigma handle it.
        # Or better: Dirichlet prior? No, Dirichlet is not sparse enough (no exact zeros).
        # Horseshoe gives exact zeros (or extremely small).
        
        # Implementation: Use the raw weights directly.
        # If the user supplied normalized signal, the model presumably aggregates to ~1.
        # If we want to strictly enforce sum<=1, we can clamp or scale.
        # For now, let's stick to the raw linear combination as requested: S = sum(w_i M_i)
        
        # Add weights to valid param list
        # We need to map them to the named fractions in order
        fraction_values = []
        for i in range(n_compartments):
            fraction_values.append(raw_weights[i].reshape(1))
            
        # 3. Concatenate all parameters
        # The SuperTissueModel expects them in the order of self.parameter_names
        # We have iterated model_p_names then fraction_names.
        # Check if this matches self.parameter_names order.
        # code in super_tissue_model.py:
        #   for i, model... add params
        #   for i, model... add fractions
        # Yes, it matches.
        
        full_params = jnp.concatenate(model_params_values + fraction_values)
        
        # 4. Predict Signal
        # Use a dummy acquisition wrapper or just pass arrays if supported
        # We need to construct a lightweight 'acquisition' object because SuperTissueModel
        # calls `self.composite_fn(params, acquisition)`
        # `composite_fn` typically unpacks acquisition.
        
        # Let's create a minimal namedtuple or class for acquisition
        from collections import namedtuple
        Acq = namedtuple('Acq', ['bvalues', 'gradient_directions', 'kwargs'])
        # Add any extra kwargs into the valid attributes or strictly passed kwargs
        # The models usually access acquisition.bvalues etc.
        # We might need to check how `compose_models` accesses it.
        # Usually: bvals = acquisition.bvalues
        
        # We can create a dynamic class to be safe
        class SimpleAcq:
            def __init__(self, b, g, k):
                self.bvalues = b
                self.gradient_directions = g
                for key, val in k.items():
                    setattr(self, key, val)
                    
        acq_obj = SimpleAcq(bvals, bvecs, acquisition_kwargs or {})
        
        # Predict
        S_pred = self.super_model(full_params, acq_obj)
        
        # 5. Likelihood
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
        
        with numpyro.plate("data", len(bvals)):
             numpyro.sample("obs", dist.Normal(S_pred, sigma), obs=signal)

    def fit(self, signal, bvals, bvecs, acquisition_kwargs=None, 
            num_samples=1000, num_warmup=1000, num_chains=1, key=None,
            init_params=None):
        if key is None:
            key = jax.random.PRNGKey(42)
            
        # Kernel with robust initialization
        if init_params is not None:
            init_strategy = numpyro.infer.init_to_value(values=init_params)
        else:
            init_strategy = numpyro.infer.init_to_sample
            
        nuts_kernel = NUTS(self.model, init_strategy=init_strategy)
        
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        
        mcmc.run(key, bvals=bvals, bvecs=bvecs, signal=signal, acquisition_kwargs=acquisition_kwargs)
        
        return mcmc

    def discover_microstructure(self, signal, bvals, bvecs, acquisition_kwargs=None, epsilon=0.01,
                                init_params=None, num_samples=1000, num_warmup=1000):
        """
        Runs the discovery pipeline and returns marginal probabilities of existance.
        """
        mcmc = self.fit(signal, bvals, bvecs, acquisition_kwargs, 
                        init_params=init_params, num_samples=num_samples, num_warmup=num_warmup)
        samples = mcmc.get_samples()
        
        # Analyze weights
        weights = samples["tissue_fractions_weights"] # Shape (Samples, N_compartments)
        
        # Probability that weight > epsilon
        probs = jnp.mean(weights > epsilon, axis=0)
        
        result = {
            "probabilities": probs,
            "mean_weights": jnp.mean(weights, axis=0),
            "median_weights": jnp.median(weights, axis=0),
            "samples": samples
        }
        return result
