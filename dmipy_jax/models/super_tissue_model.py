import jax.numpy as jnp
import equinox as eqx
from typing import Any, List
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.gaussian_models import Ball
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.signal_models.sphere_models import S1Dot
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.composer import compose_models

class SuperTissueModel(eqx.Module):
    """
    A 'Super Tissue Model' that composes multiple compartments:
    - Stick (Intra-axonal, zero radius)
    - Ball (Extra-axonal, isotropic)
    - Zeppelin (Extra-axonal, anisotropic)
    - Dot (Restricted, zero radius)
    - RestrictedCylinder (Intra-axonal, finite radius)
    
    Designed for 'Automated Model Discovery' where an L1 sparsity penalty
    drives irrelevant compartments to zero volume fraction.
    """
    
    models: List[Any] = eqx.field(static=True)
    composite_fn: Any = eqx.field(static=True)
    parameter_names: List[str] = eqx.field(static=True)
    parameter_cardinality: dict = eqx.field(static=True)
    parameter_ranges: dict = eqx.field(static=True)
    
    def __init__(self, models=None):
        # 1. Instantiate Sub-Models
        if models is None:
            self.models = [
                Stick(),
                Ball(),
                Zeppelin(),
                S1Dot(),
                RestrictedCylinder()
            ]
        else:
            self.models = models
        
        # 2. Compose
        self.composite_fn = compose_models(self.models)
        
        # 3. Aggregate Parameter Metadata
        self.parameter_names = []
        self.parameter_cardinality = {}
        self.parameter_ranges = {}
        
        # We need to flatten names carefully to avoid collisions if models share names?
        # compose_models expects a flat array.
        # But for metadata, we usually prefix?
        # Actually `compose_models` logic iterates models and consumes params.
        # It relies on `model.parameter_names`.
        # To expose a unified metadata interface, we list them in order.
        
        for i, model in enumerate(self.models):
            prefix = f"m{i}_{type(model).__name__}_"
            for name in model.parameter_names:
                unique_name = prefix + name
                self.parameter_names.append(unique_name)
                self.parameter_cardinality[unique_name] = model.parameter_cardinality[name]
                self.parameter_ranges[unique_name] = model.parameter_ranges[name]
        
        # Add Volume Fractions
        # compose_models expects N fractions at the end.
        for i, model in enumerate(self.models):
            frac_name = f"fraction_{i}_{type(model).__name__}"
            self.parameter_names.append(frac_name)
            self.parameter_cardinality[frac_name] = 1
            self.parameter_ranges[frac_name] = (0.0, 1.0)

    def __call__(self,  params, acquisition):
        """
        Predicts signal.
        
        Args:
            params: Flat array of parameters including fractions.
            acquisition: JaxAcquisition object.
        """
        return self.composite_fn(params, acquisition)
