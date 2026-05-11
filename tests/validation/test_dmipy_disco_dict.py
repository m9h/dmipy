"""
Tests for the dmipy-JAX dictionary matcher on DiSCo.

This is the dmipy-JAX-vs-FORCE head-to-head on the scalar microstructure
recovery task (paper-aligned §16 metric). The §17-§19 connectivity work
applies to FORCE through dipy; dmipy-JAX has its own DictionaryMatcher
in `dmipy_jax/library/matcher.py` which has never been pointed at DiSCo
before this commit.

Tests pin:
  - build_disco_tuned_two_stick_simulator: 6-param simulator with
    DiSCo-aligned diffusivity priors per FORCE paper §3.2.
  - fit_dmipy_dict_on_disco: end-to-end smoke fit + library round-trip.
"""

from pathlib import Path

import numpy as np
import pytest


_helper = pytest.importorskip("dmipy_jax.validation.dmipy_disco_dict")
DISCO_ROOT = Path.home() / ".dipy" / "disco" / "disco_1"


@pytest.fixture(scope="module")
def disco_ready():
    if not DISCO_ROOT.exists():
        pytest.skip(f"DiSCo not at {DISCO_ROOT}")


# --------------------------------------------------------------------------- #
# 1. DiSCo-tuned simulator
# --------------------------------------------------------------------------- #

class TestBuildDiSCoTunedSimulator:
    def test_parameter_ranges_match_paper(self):
        """Parallel diffusivity range must match FORCE paper §3.2:
        Uniform(0.54, 0.66) × 10⁻³ mm²/s = (0.54e-9, 0.66e-9) m²/s in SI."""
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )
        sim = _helper.build_disco_tuned_two_stick_simulator(acq)
        d_par_lo, d_par_hi = sim.parameter_ranges["d_par"]
        assert abs(d_par_lo - 0.54e-9) < 1e-12
        assert abs(d_par_hi - 0.66e-9) < 1e-12

    def test_f_iso_spans_extracellular_range(self):
        """DiSCo has sparse strands → GT Intra Volume Fraction mean ≈ 0.18,
        so per-voxel extracellular volume is roughly 0.4–0.95. Our 2-stick
        model has no extra-axonal zeppelin (FORCE does), so the `f_iso`
        parameter doubles as the extracellular water proxy and must span
        a wide range to recover GT NDI values across the dataset.

        (Original version of this test asserted ``f_iso_max ≤ 0.05`` based
        on a misreading of paper §3.2 "isotropic compartment was disabled".
        That phrase refers to FORCE's GM/CSF ball compartments — FORCE
        keeps its extra-axonal zeppelin. Our 2-stick model has nothing
        analogous, so we widen f_iso to cover the extracellular range
        directly.)"""
        from dmipy_jax.acquisition import JaxAcquisition
        import jax.numpy as jnp
        acq = JaxAcquisition(
            bvalues=jnp.array([0.0, 1.9e9]),
            gradient_directions=jnp.array([[1, 0, 0], [0, 0, 1]], dtype=float),
        )
        sim = _helper.build_disco_tuned_two_stick_simulator(acq)
        _, f_iso_max = sim.parameter_ranges["f_iso"]
        assert f_iso_max >= 0.8, (
            f"f_iso_max={f_iso_max} too narrow to recover DiSCo's GT "
            "Intra Volume Fraction range; 2-stick model needs f_iso as "
            "the extracellular proxy."
        )


# --------------------------------------------------------------------------- #
# 2. End-to-end smoke fit
# --------------------------------------------------------------------------- #

class TestFitDmipyDictOnDiSCo:
    def test_smoke_fit_recovers_ndi_per_voxel(self, disco_ready):
        """Run the dmipy-JAX dictionary fit on a small DiSCo ROI subset
        and verify the returned NDI map has consistent shape + range."""
        result = _helper.fit_dmipy_dict_on_disco(
            subject=1, snr=30, library_size=10_000,
            _smoke_roi_subset=(1, 2, 3),
        )
        for k in ("ndi", "fa_proxy", "mask"):
            assert k in result
        mask = result["mask"]
        ndi = result["ndi"]
        assert ndi.shape == mask.shape
        ndi_in = ndi[mask]
        assert (ndi_in >= 0).all() and (ndi_in <= 1.0 + 1e-3).all()
