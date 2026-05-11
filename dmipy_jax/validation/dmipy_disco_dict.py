"""
dmipy-JAX dictionary matcher on DiSCo — scalar microstructure recovery.

Counterpart to `dmipy_jax/validation/force_disco.py` (which fits dipy's
upstream `dipy.reconst.force.FORCEModel`) — this module uses sbi4dwi's
own `dmipy_jax.library.matcher.DictionaryMatcher` for the same task.

Library design (matches FORCE paper §3.2's tuned DiSCo protocol):
  - 2-stick + Bingham dispersion + isotropic (small ceiling)
  - Diffusivity priors: D_∥ ∈ Uniform(0.54, 0.66) × 10⁻³ mm²/s
  - ODI ∈ Uniform(0.01, 0.30)
  - f_iso ∈ Uniform(0.0, 0.05) (paper "disabled" isotropic compartment)

dmipy-JAX NDI prediction = matched `f1 + f2 = 1 - f_iso`. This is a
proxy for "intra-cellular volume fraction" since our 2-stick model has
no extra-axonal compartment; for DiSCo's pure-cylinder phantom this is
the most defensible analogue to FORCE's `fit.nd`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.signal_models.bingham import BinghamNODDI


def build_disco_tuned_two_stick_simulator(acq: JaxAcquisition) -> ModelSimulator:
    """6-param dispersion-aware 2-stick simulator with DiSCo-aligned priors.

    Parameter layout: ``[d_par, theta1, theta2, odi, f1, f_iso]``.
    """
    bingham = BinghamNODDI(grid_points=120)

    def forward_fn(params, acq):
        d_par = params[0]
        t1, t2 = params[1], params[2]
        odi = params[3]
        f1, f_iso = params[4], params[5]
        f2 = 1.0 - f1 - f_iso
        kappa = 1.0 / jnp.tan(jnp.pi * odi / 2.0)
        mu1 = jnp.array([jnp.sin(t1), 0.0, jnp.cos(t1)])
        mu2 = jnp.array([jnp.sin(t2), 0.0, jnp.cos(t2)])
        s1 = bingham(acq.bvalues, acq.gradient_directions,
                     mu=mu1, kappa1=kappa, kappa2=kappa, lambda_par=d_par)
        s2 = bingham(acq.bvalues, acq.gradient_directions,
                     mu=mu2, kappa1=kappa, kappa2=kappa, lambda_par=d_par)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_par", "theta1", "theta2", "odi", "f1", "f_iso"],
        parameter_ranges={
            # Paper §3.2 DiSCo: D_∥ ~ Uniform(0.54, 0.66) × 10⁻³ mm²/s
            "d_par": (0.54e-9, 0.66e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "odi": (0.01, 0.30),
            # f1 = primary stick fraction within the (1 - f_iso) WM volume
            "f1": (0.1, 0.8),
            # f_iso = extracellular volume fraction. DiSCo strands are sparse:
            # GT Strand Intra Volume Fraction mean ≈ 0.18, so the per-voxel
            # extracellular fraction ranges roughly 0.4–0.95. Our pure-stick
            # WM compartment has no extra-axonal zeppelin (which FORCE uses),
            # so f_iso here doubles as the extracellular water proxy.
            "f_iso": (0.0, 0.95),
        },
        acquisition=acq,
    )


def fit_dmipy_dict_on_disco(
    subject: int = 1,
    snr: int = 30,
    library_size: int = 200_000,
    seed: int = 1,
    k_best: int = 10,
    _smoke_roi_subset: Optional[Iterable[int]] = None,
) -> dict:
    """Fit dmipy-JAX dict on DiSCo and return per-voxel parameter maps.

    Returns
    -------
    dict with keys: ``ndi``, ``fa_proxy``, ``odi``, ``params``, ``mask``,
    ``gtab``, ``snr``, ``library_size``.
    """
    from dmipy_jax.library.generator import LibraryGenerator
    from dmipy_jax.library.matcher import DictionaryMatcher
    from dmipy_jax.library.storage import SimulationLibrary

    from dmipy_jax.validation.force_disco import load_disco_subject

    out = load_disco_subject(subject=subject, snr=snr, single_shell_b=1900)
    data = out["data"]
    mask = out["mask"]
    rois = out["rois"]
    # JaxAcquisition object reconstructed from gtab
    bvals = np.asarray(out["gtab"].bvals) * 1e6  # mm²/s → m²/s (SI)
    bvecs = np.asarray(out["gtab"].bvecs)
    acq = JaxAcquisition(bvalues=jnp.asarray(bvals), gradient_directions=jnp.asarray(bvecs))

    if _smoke_roi_subset is not None:
        fit_mask = np.isin(rois, list(_smoke_roi_subset)) & mask
    else:
        fit_mask = mask

    sim = build_disco_tuned_two_stick_simulator(acq)
    print(f"  generating dmipy-JAX library: {library_size:,} entries...")
    gen = LibraryGenerator(sim, chunk_size=20_000)
    params, signals = gen.generate(library_size, key=jax.random.PRNGKey(seed))
    jax.block_until_ready(signals)
    lib = SimulationLibrary(
        params=params, signals=signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=k_best)

    # Volume-level matching: returns {param_name: (X,Y,Z)} maps
    print(f"  matching {int(fit_mask.sum()):,} voxels against library...")
    maps = matcher.match_volume(data, mask=fit_mask, batch_size=4096)

    ndi = np.zeros(mask.shape, dtype=np.float32)
    odi = np.zeros(mask.shape, dtype=np.float32)
    fa_proxy = np.zeros(mask.shape, dtype=np.float32)

    f1 = maps["f1"]
    f_iso = maps["f_iso"]
    # dmipy NDI = total stick fraction = f1 + f2 = 1 - f_iso (in our 2-stick
    # + iso model, the entire WM compartment is sticks).
    ndi_map = (1.0 - f_iso).astype(np.float32)
    ndi[fit_mask] = ndi_map[fit_mask]
    odi[fit_mask] = maps["odi"][fit_mask].astype(np.float32)

    # FA proxy from the angle between the two recovered fibres.
    # When fibres are aligned (single-fibre voxel) → high FA; when
    # orthogonal (crossing) → low FA. Not strictly FA but tracks the
    # tensor-FA pattern on this kind of phantom.
    t1 = maps["theta1"]
    t2 = maps["theta2"]
    cos_diff = np.cos(t1 - t2)
    f1m = maps["f1"]
    f2m = 1.0 - f1m - f_iso
    # Bingham concentration also matters; a single sharp fibre = high FA
    fa_proxy_arr = np.abs(cos_diff) * np.maximum(f1m, f2m)
    fa_proxy[fit_mask] = fa_proxy_arr[fit_mask].astype(np.float32)

    return {
        "ndi": ndi,
        "odi": odi,
        "fa_proxy": fa_proxy,
        "params": maps,
        "mask": fit_mask,
        "gtab": out["gtab"],
        "rois": rois,
        "snr": snr,
        "library_size": library_size,
    }
