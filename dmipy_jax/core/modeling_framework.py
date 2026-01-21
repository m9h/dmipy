import jax
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.composer import compose_models
from dmipy_jax.fitting.optimization import OptimistixFitter, VoxelFitter
import equinox as eqx

class CompartmentModel(eqx.Module):
    """
    Abstract base class for all signal models (compartments).
    """
    def __call__(self, bvals, gradient_directions, **kwargs):
        raise NotImplementedError

    @property
    def parameter_names(self):
        raise NotImplementedError
    
    @property
    def parameter_cardinality(self):
        raise NotImplementedError

    @property
    def parameter_ranges(self):
        raise NotImplementedError

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

    def __call__(self, parameters, acquisition_scheme):
        """
        Simulate signal.
        Args:
            parameters: dict or flat array
            acquisition_scheme: acquisition object
        """
        if isinstance(parameters, dict):
            params_flat = self.parameter_dictionary_to_array(parameters)
        else:
            params_flat = parameters
            
        return self.model_func(params_flat, acquisition_scheme)

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

    def __call__(self, parameter_dictionary, acquisition):
        """
        Simulates signal for the given parameters and acquisition scheme.
        Supports both single-voxel (scalars) and multi-voxel (arrays) parameter inputs.
        
        Args:
            parameter_dictionary (dict): Dictionary of parameters.
            acquisition (JaxAcquisition): Acquisition scheme.
            
        Returns:
            jnp.ndarray: Simulated signal (N_meas,) or (N_vox, N_meas).
        """
        # 1. Determine if inputs are batched
        # Check first parameter in dictionary
        first_key = self.parameter_names[0]
        if first_key not in parameter_dictionary:
             # Try finding any valid key
             found = False
             for k in self.parameter_names:
                 if k in parameter_dictionary:
                     first_key = k
                     found = True
                     break
             if not found:
                 raise ValueError(f"No valid parameters found in dictionary. Expected {self.parameter_names}")

        val = parameter_dictionary[first_key]
        card = self.parameter_cardinality[first_key]
        
        # Check dimensionality relative to cardinality
        # If card=1, scalar is ndim=0. Batch is ndim=1.
        # If card>1, vector is ndim=1 (or flattened). Batch is ndim=2.
        # We assume vectors are never flattened to scalars in the input dict, they should be arrays.
        
        if card == 1:
            is_batched = jnp.ndim(val) > 0
        else:
            is_batched = jnp.ndim(val) > 1
        
        # 2. Convert to Array
        if is_batched:
            # Setup vmapped conversion
            # parameter_dictionary_to_array expects scalars. 
            # We vmap it.
            # But parameter_dictionary is a dict of arrays. 
            # jax.vmap(func)(dict_of_arrays) works if structure matches.
            
            converter = jax.vmap(self.parameter_dictionary_to_array)
            params_flat = converter(parameter_dictionary) # (N, n_params)
            
            # 3. Simulate (Vmapped)
            simulator = jax.vmap(self.model_func, in_axes=(0, None))
            return simulator(params_flat, acquisition)
            
        else:
            # Single voxel
            params_flat = self.parameter_dictionary_to_array(parameter_dictionary)
            return self.model_func(params_flat, acquisition)


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


    def fit(self, acquisition, data, method="Levenberg-Marquardt", compute_uncertainty=True, batch_size=None):
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
                if not np.isinf(high) and (high != 0):
                    s = high
                elif not np.isinf(low) and (low != 0):
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
                     s = high if not np.isinf(high) and high!=0 else 1.0
                     current_scales.extend([s] * card)
                else:
                     flat_ranges.extend(rng)
                     for r in rng:
                         l, h = r
                         s = h if not np.isinf(h) and h!=0 else 1.0
                         current_scales.append(s)
            
            scales_list.extend(current_scales)

        scales = jnp.array(scales_list)

        # 2. Instantiate Fitter
        # Pass scales to OptimistixFitter
        if method in ["LBFGSB", "VoxelFitter"]:
            fitter = VoxelFitter(self.model_func, flat_ranges, scales=scales)
        else:
            # Default to Optimistix (Levenberg-Marquardt)
            fitter = OptimistixFitter(self.model_func, flat_ranges, scales=scales)
        
        # 3. Initial Guess
        # Use GlobalBruteInitializer (Random Search in this implementation)
        from dmipy_jax.fitting.initialization import GlobalBruteInitializer
        
        # Determine number of initialization points.
        # For single voxel: 50 points? For 1M voxels, maybe fewer per voxel or shared?
        # If we use random grid, we can generate ONE grid and check it against all voxels (vmapped)
        
        initializer = GlobalBruteInitializer(self)
        
        # Generate random candidates using helper
        # Use simple fixed key or allow user key? For now fixed for reproducibility
        key = jax.random.PRNGKey(42)
        candidates = initializer.generate_random_grid(n_samples=2000, key=key)
        
        if data.ndim == 1:
            init_params = initializer.compute_initial_guess(data, acquisition, candidates)
        else:
            # Multi-voxel
            data = data.reshape(-1, data.shape[-1])
            N_vox = data.shape[0]
            
            # Multi-voxel
            data = data.reshape(-1, data.shape[-1])
            N_vox = data.shape[0]
            
            # Precompute predictions to avoid N_vox x N_cand simulations
            # vmap over candidates (0), acquisition fixed (None)
            simulator = jax.vmap(self.model_func, in_axes=(0, None))
            candidate_predictions = simulator(candidates, acquisition)
            
            # Select best
            # vmap select_best over data (0)
            # candidate_predictions fixed (None), candidates fixed (None)
            selector = jax.jit(jax.vmap(initializer.select_best_candidate, in_axes=(0, None, None)))
            
            # Helper for fitting a batch
            fit_vmapped = jax.jit(jax.vmap(fitter.fit, in_axes=(0, None, 0)))
            
            # Helper for uncertainty
            from dmipy_jax.core.uncertainty_utils import compute_jacobian, compute_crlb_std
            def single_pixel_uncertainty(params, data_voxel):
                # Calculate sigma from residuals (MSE)
                prediction = self.model_func(params, acquisition)
                residuals = data_voxel - prediction
                sigma_est = jnp.sqrt(jnp.mean(residuals**2))
                # sigma_est = estimate_sigma(params, data_voxel, acquisition)
                J = compute_jacobian(self.model_func, params, acquisition)
                return compute_crlb_std(J, sigma=sigma_est)
            
            calc_uncertainty_vmapped = jax.jit(jax.vmap(single_pixel_uncertainty))

            # Determine if batching is needed
            if batch_size is None or N_vox <= batch_size:
                # Full VMAP
                init_params = selector(data, candidate_predictions, candidates)
                fitted_batch, _ = fit_vmapped(data, acquisition, init_params)
                if compute_uncertainty:
                    stds_batch = calc_uncertainty_vmapped(fitted_batch, data)
                else:
                    stds_batch = None
            else:
                # Chunked Processing
                import math
                n_chunks = math.ceil(N_vox / batch_size)
                fitted_chunks = []
                stds_chunks = []
                
                for i in range(n_chunks):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, N_vox)
                    
                    data_chunk = data[start_idx:end_idx]
                    
                    # Compute Init for Chunk (Memory Safe)
                    init_chunk = selector(data_chunk, candidate_predictions, candidates)
                    
                    # Run Fit
                    res_chunk, _ = fit_vmapped(data_chunk, acquisition, init_chunk)
                    fitted_chunks.append(res_chunk)
                    
                    if compute_uncertainty:
                        std_chunk = calc_uncertainty_vmapped(res_chunk, data_chunk)
                        stds_chunks.append(std_chunk)
                
                fitted_batch = jnp.concatenate(fitted_chunks, axis=0)
                if compute_uncertainty:
                    stds_batch = jnp.concatenate(stds_chunks, axis=0)
                else:
                    stds_batch = None
            
            
            # 5. Pack results
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
            
            if compute_uncertainty and stds_batch is not None:
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
