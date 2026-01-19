
Dmipy-JAX Contribution Guidelines
=================================

The idea behind dmipy-jax is that it is a modular, GPU-accelerated port of dmipy using JAX. It leverages the "Kidger scientific stack" to provide robust, differentiable, and fast microstructure imaging tools.

To contribute your work to dmipy-jax, we ask that you adhere to the guidelines akin to the original project but adapted for the JAX ecosystem.

Prerequisite Reading (The "Kidger Stack")
-----------------------------------------

Development in `dmipy-jax` relies on a set of high-quality libraries built on top of JAX. Familiarity with these is crucial:

1.  **JAX**: The core engine for composable transformations (autodiff, JIT compilation, vectorization).
    - `JAX Documentation <https://jax.readthedocs.io/>`_
    - `JAX "The Sharp Bits" <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_

2.  **Equinox**: Used for building stateful, object-oriented models (like CompartmentModels) safely within JAX.
    - `Equinox Documentation <https://docs.kidger.site/equinox/>`_
    - Key Concept: All modules are PyTrees.

3.  **Optimistix**: Used for non-linear least squares (NLLS) fitting and root finding.
    - `Optimistix Documentation <https://docs.kidger.site/optimistix/>`_

4.  **Lineax**: Used for efficient linear solves, often as a subcommand within larger optimizations.
    - `Lineax Documentation <https://docs.kidger.site/lineax/>`_

5.  **Jaxtyping**: Used for type annotations to ensure array shape and dtype correctness.
    - `Jaxtyping Documentation <https://docs.kidger.site/jaxtyping/>`_

Developing New Models
---------------------

Models in `dmipy-jax` should be implemented as `equinox.Module` classes. This allows them to be pytrees, meaning they can be passed freely through JIT-compiled functions.

**Blueprint for a Compartment Model:**

.. code:: python

    import equinox as eqx
    import jax.numpy as jnp
    from jaxtyping import Array, Float

    class NewCompartmentModel(eqx.Module):
        # Parameters are stored as fields in the module
        parameter1: Float[Array, ""]
        parameter2: Float[Array, ""]
        
        def __init__(self, parameter1=None, parameter2=None):
            # detailed initialization logic
            self.parameter1 = parameter1
            self.parameter2 = parameter2

        def __call__(self, acquisition_scheme, **kwargs) -> Float[Array, "N"]:
            # JAX-compatible signal generation
            # self.parameter1 etc. are available directly
            pass

If you want contribute, just contact us or open a pull request on the `m9h/dmipy` repository.
