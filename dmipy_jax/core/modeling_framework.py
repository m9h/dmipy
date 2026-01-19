import jax
import jax.numpy as jnp
from dmipy_jax.composer import compose_models
from dmipy_jax.fitting.optimization import OptimistixFitter

class JaxMultiCompartmentModel:
    """
    A class for combining multiple signal models into a Multi-Compartment Model (MCM).
    
    This class wraps the `compose_models` functional composition with an object-oriented
    interface compatible with the `OptimistixFitter` and user-friendly parameter
    management (dictionaries).
    """
    
    def __init__(self, models):
        """
        Args:
            models (list): List of JAX-based signal model instances.
                           Each model must have `parameter_names` and `parameter_cardinality`.
        """
        self.models = models
        self.model_func = compose_models(models)
        
        self.parameter_names = []
        self.parameter_cardinality = {}
        self.parameter_ranges = {}
        self.parameter_map = [] # To map flat array back to names
        
        # Build parameter metadata
        # We need to map the flat array index to (name, sub_index)
        
        # 1. Model Parameters
        for i, model in enumerate(models):
            # Resolve name collisions if any (though typically users might construct models 
            # with distinct names or we assign them).
            # Here we just iterate. If multiple models have same param names, 
            # we need to disambiguate for the dictionary representation.
            
            # Simple disambiguation strategy: append _{index} if collision
            # OR better: usage of model name prefix?
            # For this implementation, we will check for collisions and append _{model_index}
            
            for pname in model.parameter_names:
                unique_name = pname
                if unique_name in self.parameter_names:
                    # Collision
                    unique_name = f"{pname}_{i+1}"
                
                self.parameter_names.append(unique_name)
                card = model.parameter_cardinality[pname]
                self.parameter_cardinality[unique_name] = card
                
                # Handle Ranges
                # Expecting model.parameter_ranges to be a dict: {name: (min, max)} or {name: [(min, max), ...]}
                if hasattr(model, 'parameter_ranges') and pname in model.parameter_ranges:
                    rng = model.parameter_ranges[pname]
                    self.parameter_ranges[unique_name] = rng
                else:
                    # Default defaults
                    if card == 1:
                        self.parameter_ranges[unique_name] = (-jnp.inf, jnp.inf)
                    else:
                        self.parameter_ranges[unique_name] = [(-jnp.inf, jnp.inf)] * card

        # 2. Partial Volume Fractions
        # compose_models appends N fractions at the end
        for i in range(len(models)):
            fname = f'partial_volume_{i}'
            # Check collision
            if fname in self.parameter_names:
                fname = f'partial_volume_{i}_frac' # Fallback
            
            self.parameter_names.append(fname)
            self.parameter_cardinality[fname] = 1
            self.parameter_ranges[fname] = (0.0, 1.0)

    def get_flat_bounds(self):
        """
        Returns flat lists of lower and upper bounds for all parameters.
        Returns:
            (jnp.ndarray, jnp.ndarray): lower_bounds, upper_bounds
        """
        lows = []
        highs = []
        
        for name in self.parameter_names:
            card = self.parameter_cardinality[name]
            rng = self.parameter_ranges[name]
            
            if card == 1:
                l, h = rng
                lows.append(l)
                highs.append(h)
            else:
                if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], (int, float)):
                    # Uniform range for vector
                    l, h = rng
                    lows.extend([l] * card)
                    highs.extend([h] * card)
                else:
                    # List of ranges
                    for r in rng:
                        l, h = r
                        lows.append(l)
                        highs.append(h)
                        
        return jnp.array(lows), jnp.array(highs)



    def parameter_dictionary_to_array(self, parameter_dictionary):
        """
        Converts a dictionary of parameters to the flat array expected by the kernel.
        """
        params_list = []
        for name in self.parameter_names:
            if name not in parameter_dictionary:
                raise ValueError(f"Missing parameter '{name}' in dictionary.")
            
            val = parameter_dictionary[name]
            card = self.parameter_cardinality[name]
            
            if card == 1:
                # Scalar check
                # Check for vector input where scalar expected
                # Use jnp.size to catch arrays with >1 element safely
                if jnp.size(val) > 1:
                     raise ValueError(f"Parameter '{name}' expects scalar, got {val} with size {jnp.size(val)}")
                
                # Append as scalar (squeeze handles 0-d or 1-d of size 1)
                params_list.append(jnp.squeeze(val))
            else:
                # Vector
                # Flatten
                if isinstance(val, (list, tuple)):
                    params_list.extend(val)
                else:
                    params_list.extend(jnp.ravel(val))
                    
        return jnp.concatenate([jnp.atleast_1d(p) for p in params_list])


    def parameter_array_to_dictionary(self, parameter_array):
        """
        Converts the flat result array back to a user-friendly dictionary.
        """
        ret = {}
        idx = 0
        for name in self.parameter_names:
            card = self.parameter_cardinality[name]
            if card == 1:
                ret[name] = parameter_array[idx]
                idx += 1
            else:
                ret[name] = parameter_array[idx : idx + card]
                idx += card
        return ret


    def fit(self, acquisition, data, method="Levenberg-Marquardt", compute_uncertainty=True):
        """
        Fits the model to data.
        
        Args:
            acquisition: JaxAcquisition object
            data: Signal data array. Shape (N_meas,) for single voxel or (N_vox, N_meas) for multiple.
            method: Optimization method (default: "Levenberg-Marquardt").
            compute_uncertainty: If True, computes CRLB standard deviations (default: True).
        
        Returns:
            dict: Fitted parameters in dictionary format.
                  If compute_uncertainty is True, includes keys with '_std' suffix.
        """
        # 1. Prepare Ranges for Fitter
        # OptimistixFitter expects a list of (min, max) for every single scalar parameter in the flat array.
        
        flat_ranges = []
        scales_list = []
        for name in self.parameter_names:
            card = self.parameter_cardinality[name]
            rng = self.parameter_ranges[name]
            
            # Determine scale for this parameter
            # Heuristic: use order of magnitude of the upper bound, or 1.0
            # If (low, high), scale = high if high < 1e-2 or high > 1e2
            
            current_scales = []
            
            if card == 1:
                # rng is (min, max)
                flat_ranges.append(rng)
                low, high = rng
                # Pick scale
                if not jnp.isinf(high) and (high != 0):
                    s = high
                elif not jnp.isinf(low) and (low != 0):
                    s = low
                else:
                    s = 1.0
                current_scales.append(s)
            else:
                # rng is list
                if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], (int, float)):
                     # Replicate
                     flat_ranges.extend([rng] * card)
                     low, high = rng
                     s = high if not jnp.isinf(high) and high!=0 else 1.0
                     current_scales.extend([s] * card)
                else:
                     flat_ranges.extend(rng)
                     for r in rng:
                         l, h = r
                         s = h if not jnp.isinf(h) and h!=0 else 1.0
                         current_scales.append(s)
            
            scales_list.extend(current_scales)

        scales = jnp.array(scales_list)

        # 2. Instantiate Fitter
        # Pass scales to OptimistixFitter
        fitter = OptimistixFitter(self.model_func, flat_ranges, scales=scales)
        
        # 3. Initial Guess
        # Use GlobalBruteInitializer (Random Search in this implementation)
        from dmipy_jax.fitting.initialization import GlobalBruteInitializer
        
        # Determine number of initialization points.
        # For single voxel: 50 points? For 1M voxels, maybe fewer per voxel or shared?
        # If we use random grid, we can generate ONE grid and check it against all voxels (vmapped)
        
        initializer = GlobalBruteInitializer(self)
        
        # Generate random candidates
        lows, highs = self.get_flat_bounds()
        
        # Handle infinities for random sampling
        # Replace inf with practical limits
        safe_lows = jnp.where(jnp.isinf(lows), -10.0, lows) # Arbitrary safe
        safe_highs = jnp.where(jnp.isinf(highs), 10.0, highs)
        
        # Replace 0-inf range (diffusivity) with 0-3e-9 approx if not specified? 
        # Actually user ranges should be good.
        
        n_candidates = 2000 
        key = jax.random.PRNGKey(42)
        
        # random uniform (N_cand, N_params)
        rand_uni = jax.random.uniform(key, (n_candidates, len(lows)))
        
        # Scale to bounds: low + rand * (high - low)
        candidates = safe_lows + rand_uni * (safe_highs - safe_lows)
        
        if data.ndim == 1:
            init_params = initializer.compute_initial_guess(data, acquisition, candidates)
        else:
            # Multi-voxel
            # vmap compute_initial_guess over data
            # candidates are shared (None)
            selector = jax.vmap(initializer.compute_initial_guess, in_axes=(0, None, None))
            init_params = selector(data, acquisition, candidates)
            # init_params shape (N_vox, N_params)

        
        # 4. Run Fit
        from dmipy_jax.core.uncertainty_utils import compute_crlb_std
        
        # Helper to compute residual sigma for CRLB
        def estimate_sigma(params, data, acquisition):
            pred = self.model_func(params, acquisition)
            mse = jnp.mean((data - pred)**2)
            return jnp.sqrt(mse)

        if data.ndim == 1:
            fitted, _ = fitter.fit(data, acquisition, init_params)
            ret = self.parameter_array_to_dictionary(fitted)
            
            if compute_uncertainty:
                sigma_est = estimate_sigma(fitted, data, acquisition)
                # Compute CRLB
                # jacobian needs model_func, params, acq
                # But compute_crlb_std separates jacobian calc? No, I defined `compute_jacobian` in utils as separate.
                # Let's reuse compute_jacobian from utils or just do it here if simple.
                from dmipy_jax.core.uncertainty_utils import compute_jacobian, compute_crlb_std
                
                J = compute_jacobian(self.model_func, fitted, acquisition)
                stds = compute_crlb_std(J, sigma=sigma_est)
                
                # Unpack stds
                idx = 0
                for name in self.parameter_names:
                    card = self.parameter_cardinality[name]
                    if card == 1:
                        ret[f"{name}_std"] = stds[idx]
                        idx += 1
                    else:
                        ret[f"{name}_std"] = stds[idx:idx+card]
                        idx += card
            return ret

        else:
            # VMAP over voxels
            fit_vmapped = jax.vmap(fitter.fit, in_axes=(0, None, None))
            fitted_batch, _ = fit_vmapped(data, acquisition, init_params)
            
            ret = {}
            idx = 0
            for name in self.parameter_names:
                card = self.parameter_cardinality[name]
                if card == 1:
                    ret[name] = fitted_batch[:, idx]
                    idx += 1
                else:
                    ret[name] = fitted_batch[:, idx:idx+card]
                    idx += card
            
            if compute_uncertainty:
                # Vmap uncertainty calculation
                from dmipy_jax.core.uncertainty_utils import compute_jacobian, compute_crlb_std
                
                def single_pixel_uncertainty(params, data_voxel):
                    sigma_est = estimate_sigma(params, data_voxel, acquisition)
                    J = compute_jacobian(self.model_func, params, acquisition)
                    return compute_crlb_std(J, sigma=sigma_est)
                
                stds_batch = jax.vmap(single_pixel_uncertainty)(fitted_batch, data)
                
                idx = 0
                for name in self.parameter_names:
                    card = self.parameter_cardinality[name]
                    if card == 1:
                        ret[f"{name}_std"] = stds_batch[:, idx]
                        idx += 1
                    else:
                        ret[f"{name}_std"] = stds_batch[:, idx:idx+card]
                        idx += card
                        
            return ret
