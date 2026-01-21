import jax.numpy as jnp

def compose_models(models):
    """
    Composes a list of JAX-based signal models into a single differentiable function.

    This utility combines multiple microstructure models (e.g., Stick, Ball) into a
    multi-compartment model where the signal is a weighted sum of the individual
    compartment signals. It handles parameter packing and unpacking automatically.

    Args:
        models (list): List of instantiated JAX model objects (e.g. ``[C1Stick(), G1Ball()]``).
            Each model must have ``parameter_names`` and ``parameter_cardinality`` attributes defined.

    Returns:
        function: A function with signature ``composite_model(params, acquisition) -> signal``.
            
            - **params** (*jax.numpy.ndarray*): A 1D JAX array containing all flattened model parameters
              concatenated in order, followed by the partial volume fractions for each model
              (except potentially the last one if constrained, though this implementation expects N fractions).
              Shape: ``(Sum(params_per_model) + N_models,)``.
            - **acquisition** (*JaxAcquisition*): The acquisition scheme.
            
            Returns a 1D array of signal values.
    """

    # Validate models have required metadata
    for i, model in enumerate(models):
        if not hasattr(model, 'parameter_names') or not hasattr(model, 'parameter_cardinality'):
            raise ValueError(f"Model at index {i} ({type(model).__name__}) missing parameter_names or parameter_cardinality.")

    def composite_model(params, acquisition):
        """
        Predicts signal attenuation from a composite model.

        Parameters
        ----------
        params : jax.numpy.ndarray
            1D array of shape (N_total_params + N_models,).
            Structure: [Model1_Params, Model2_Params, ..., ModelN_Params, frac1, frac2, ..., fracN].
        acquisition : JaxAcquisition
            Acquisition scheme containing bvalues, gradient_directions, etc.

        Returns
        -------
        signal : jax.numpy.ndarray
            Weighted sum of signal attenuations.
        """
        
        # Calculate total signal
        signal = 0.0
        
        current_idx = 0
        n_models = len(models)
        
        # The last n_models parameters are the partial volume fractions
        # We process models one by one, extracting their parameters from the start of the array
        
        for i, model in enumerate(models):
            # Extract parameters for this model
            kwargs = {}
            for name in model.parameter_names:
                cardinality = model.parameter_cardinality[name]
                if cardinality == 1:
                    val = params[current_idx]
                    current_idx += 1
                else:
                    val = params[current_idx : current_idx + cardinality]
                    current_idx += cardinality
                kwargs[name] = val
            
            # Predict signal for this model
            # Pass acquisition attributes as arguments if the model expects them (bvals, gradient_directions)
            # Most dmipy_jax models expect 'bvals' and 'gradient_directions' in __call__ or kwargs
            # We explicitly pass them.
            
            # Note: The current JAX models (G1Ball, C1Stick) __call__ signatures vary.
            # G1Ball: (bvals, **kwargs)
            # C1Stick: (bvals, gradient_directions, **kwargs)
            # We will pass both bvals and gradient_directions to all models.
            # Python's **kwargs in __call__ should handle extra arguments if the signature allows it,
            # but G1Ball's signature is `def __call__(self, bvals, **kwargs):` which absorbs extras.
            # C1Stick's signature is `def __call__(self, bvals, gradient_directions, **kwargs):`.
            # So passing both as positional args might be risky if they don't align.
            # Safest is to pass them by name if possible, or assume a standard signature.
            # Let's check the signatures again from the file views.
            
            # G1Ball: `def __call__(self, bvals, **kwargs)`
            # C1Stick: `def __call__(self, bvals, gradient_directions, **kwargs)`
            
            # If we call `model(bvals=..., gradient_directions=..., **kwargs)`, it should work for both
            # if they handle **kwargs correctly.
            # G1Ball params validation:
            # lambda_iso = kwargs.get('lambda_iso', self.lambda_iso) -> OK
            # C1Stick params validation:
            # lambda_par = kwargs.get('lambda_par', self.lambda_par) -> OK
            
            # However, G1Ball __call__ implementation:
            # `def __call__(self, bvals, **kwargs):`
            # If we call it as `model(bvals=bvals, gradient_directions=gradients, **kwargs)`,
            # `bvals` matches `bvals` arg if passed as positional? No, if we pass keyword args, 
            # we must ensure it matches.
            
            # Wait, `__call__(self, bvals, **kwargs)` means `bvals` is a positional OR keyword argument.
            # So `model(bvals=..., gradient_directions=..., **kwargs)` works.
            
            # C1Stick: `def __call__(self, bvals, gradient_directions, **kwargs)`
            # So `model(bvals=..., gradient_directions=..., **kwargs)` works.
            
            # So we will pass bvals and gradient_directions as kwargs (or mixed).
            
            # Actually, to be safe and consistent with typical functional JAX patterns, 
            # let's pass bvals and gradient_directions as arguments.
            
            # Inject timing parameters if available in acquisition
            if acquisition.delta is not None:
                kwargs['delta'] = acquisition.delta
                kwargs['small_delta'] = acquisition.delta
            if acquisition.Delta is not None:
                kwargs['Delta'] = acquisition.Delta
                kwargs['big_delta'] = acquisition.Delta
            
            # Inject TE/TR for relaxation models
            if hasattr(acquisition, 'TE') and acquisition.TE is not None:
                kwargs['TE'] = acquisition.TE
            if hasattr(acquisition, 'TR') and acquisition.TR is not None:
                kwargs['TR'] = acquisition.TR
            
            # Pass acquisition object for advanced models
            kwargs['acquisition'] = acquisition

            sub_signal = model(
                bvals=acquisition.bvalues,
                gradient_directions=acquisition.gradient_directions,
                **kwargs
            )

            # Get partial volume fraction for this model
            # The fractions are at the end of the params array
            frac_idx = -n_models + i
            fraction = params[frac_idx]
            
            signal += fraction * sub_signal
            
        return signal

    return composite_model
