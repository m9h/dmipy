"""Neural posterior estimation backends for simulation-based inference.

This package provides the inference engines that learn the posterior
distribution p(theta | x) over tissue microstructure parameters theta
given observed diffusion signals x.

Available backends:

* :mod:`~dmipy_jax.inference.mdn` -- Mixture Density Network (MDN)
  with configurable Gaussian components.
* :mod:`~dmipy_jax.inference.flows` -- Rational-quadratic spline
  normalizing flow (custom implementation).
* :mod:`~dmipy_jax.inference.trainer` -- FlowJAX-based conditional
  normalizing flow trainer (affine and spline coupling layers).
* :mod:`~dmipy_jax.inference.mcmc` -- BlackJAX NUTS sampler for
  Bayesian inference with explicit likelihoods.
* :mod:`~dmipy_jax.inference.score_posterior` -- Score-based diffusion
  posterior estimator with E(3)-equivariant score networks.
* :mod:`~dmipy_jax.inference.variational` -- Variational inference.
* :mod:`~dmipy_jax.inference.amortized` -- Amortized posterior networks.

The MDN and flow backends are the primary SBI workhorses, wrapped by
:func:`~dmipy_jax.pipeline.train.train_sbi` into normalised modules
that handle the ``[0,1] <-> physical`` parameter mapping automatically.
"""
