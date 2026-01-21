import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.core.modeling_framework import CompartmentModel
from typing import Any, Optional

class RelaxationModel(CompartmentModel):
    r"""
    A model for T1 and T2 relaxation kinetics.
    
    Signal = (1 - exp(-TR/T1)) * exp(-TE/T2)
    
    If TR is None (or infinite), the T1 term is 1.0.
    """
    t1: Any = None
    t2: Any = None

    parameter_names = ['t1', 't2']
    parameter_cardinality = {'t1': 1, 't2': 1}
    # T1/T2 typical ranges in seconds
    parameter_ranges = {
        't1': (0.1, 5.0),
        't2': (0.01, 1.0)
    }
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        t1 = kwargs.get('t1', self.t1)
        t2 = kwargs.get('t2', self.t2)
        
        TE = getattr(kwargs.get('acquisition', None), 'TE', None)
        TR = getattr(kwargs.get('acquisition', None), 'TR', None)
        
        # If acquisition not passed in kwargs, check if TE/TR passed directly?
        if TE is None: TE = kwargs.get('TE', None)
        if TR is None: TR = kwargs.get('TR', None)
        
        return self.relaxation_kernel(TE, TR, t1, t2)
        
    @staticmethod
    def relaxation_kernel(TE, TR, t1, t2):
        term_t2 = 1.0
        if TE is not None and t2 is not None:
            # Avoid div by zero
            # t2 = jnp.clip(t2, a_min=1e-6) # handled by ranges usually
            term_t2 = jnp.exp(-TE / t2)
            
        term_t1 = 1.0
        if TR is not None and t1 is not None:
             term_t1 = 1.0 - jnp.exp(-TR / t1)
             
        return term_t2 * term_t1


class JointModel(CompartmentModel):
    """
    Joint Diffusion-Relaxation Model.
    
    Combines a diffusion CompartmentModel with T1/T2 relaxation.
    """
    diffusion_model: CompartmentModel
    relaxation_model: RelaxationModel
        
    @property
    def parameter_names(self):
        return list(self.diffusion_model.parameter_names) + self.relaxation_model.parameter_names

    @property
    def parameter_cardinality(self):
        c = self.diffusion_model.parameter_cardinality.copy()
        c.update(self.relaxation_model.parameter_cardinality)
        return c

    @property
    def parameter_ranges(self):
        r = self.diffusion_model.parameter_ranges.copy()
        r.update(self.relaxation_model.parameter_ranges)
        return r
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        # 1. Compute Diffusion Signal
        # diffusion model might expect simple args or also kwargs
        # We pass everything down
        S_diff = self.diffusion_model(bvals, gradient_directions, **kwargs)
        
        # 2. Compute Relaxation
        # We need to make sure 'acquisition' is in kwargs if we want to access TE/TR from it
        # or pass TE/TR explicitly if they are in kwargs
        S_relax = self.relaxation_model(bvals, gradient_directions, **kwargs)
        
        return S_diff * S_relax
