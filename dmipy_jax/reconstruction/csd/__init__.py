"""
Constrained Spherical Deconvolution (CSD) Module

Submodules:
- classic: Standard and Algebraic (SOS) CSD.
- deep: Equivariant (e3nn) and Generative CSD.
- bayesian: GPJax-based Non-Parametric CSD.
"""

# Expose main solvers as they are implemented
from .classic.solvers import fit_sos_csd, fit_csd
from .classic.response import ResponseEstimator
from . import deep
from . import bayesian
