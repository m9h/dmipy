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
                if hasattr(val, '__len__') and len(val) > 1:
                     raise ValueError(f"Parameter '{name}' expects scalar, got {val}")
                # If array-like of size 1, extract? 
                # jnp.array(val) handles it usually, but we want flat list.
                # If val is float, append.
                # If val is array, extend.
                
                # Careful: if val is array(0.5), extend works? No.
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


    def fit(self, acquisition, data, method="Levenberg-Marquardt"):
        """
        Fits the model to data.
        
        Args:
            acquisition: JaxAcquisition object
            data: Signal data array. Shape (N_meas,) for single voxel or (N_vox, N_meas) for multiple.
        
        Returns:
            dict: Fitted parameters in dictionary format.
        """
        # 1. Prepare Ranges for Fitter
        # OptimistixFitter expects a list of (min, max) for every single scalar parameter in the flat array.
        
        flat_ranges = []
        for name in self.parameter_names:
            card = self.parameter_cardinality[name]
            rng = self.parameter_ranges[name]
            
            if card == 1:
                # rng should be (min, max)
                flat_ranges.append(rng)
            else:
                # rng should be list of (min, max) of length card
                # If user passed single tuple for vector parameter, replicate it?
                # Usually better to be explicit.
                if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], (int, float)):
                     # Assume applies to all axes
                     flat_ranges.extend([rng] * card)
                else:
                     flat_ranges.extend(rng)

        # 2. Instantiate Fitter
        fitter = OptimistixFitter(self.model_func, flat_ranges)
        
        # 3. Initial Guess
        # Ideally, we should use a smarter init. For now, mean of bounds.
        init_params_list = []
        for low, high in flat_ranges:
            if jnp.isinf(low) and jnp.isinf(high):
                init_params_list.append(0.5) # Fallback
            elif jnp.isinf(high):
                init_params_list.append(low + 1.0)
            elif jnp.isinf(low):
                init_params_list.append(high - 1.0)
            else:
                init_params_list.append((low + high) / 2.0)
                
        init_params = jnp.array(init_params_list)
        
        # 4. Run Fit
        if data.ndim == 1:
            fitted, _ = fitter.fit(data, acquisition, init_params)
            return self.parameter_array_to_dictionary(fitted)
        else:
            # VMAP over voxels
            # data: (N_vox, N_meas)
            # init_params: (N_params,) -> replicated? 
            # jitter.fit signature: (data, acq, init)
            
            fit_vmapped = jax.vmap(fitter.fit, in_axes=(0, None, None))
            fitted_batch, _ = fit_vmapped(data, acquisition, init_params)
            
            # Helper for batch dict
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
            return ret
