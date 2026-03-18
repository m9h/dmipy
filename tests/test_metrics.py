"""
Tests for derived metrics — FA, MD, ODI, DKI (AK, RK, KFA, micro-FA),
tissue segmentation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_jax.library.derived_metrics import (
    tensor_to_fa_md,
    eigenvalues_to_ad_rd,
    watson_kappa_to_odi,
    bingham_kappas_to_odi,
    multi_compartment_to_eigenvalues,
    multi_compartment_to_mean_kurtosis,
    multi_compartment_to_axial_kurtosis,
    multi_compartment_to_radial_kurtosis,
    multi_compartment_to_kurtosis_fa,
    micro_fa,
)
from dmipy_jax.pipeline.metrics import MultiMetricExtractor


class TestDTIMetrics:

    def test_isotropic_fa_zero(self):
        """Isotropic tensor (equal eigenvalues) should have FA = 0."""
        eigs = jnp.array([1e-9, 1e-9, 1e-9])
        fa, md = tensor_to_fa_md(eigs)
        assert jnp.abs(fa) < 1e-5
        assert jnp.abs(md - 1e-9) < 1e-15

    def test_anisotropic_fa(self):
        """Highly anisotropic tensor should have high FA."""
        eigs = jnp.array([3e-9, 0.1e-9, 0.1e-9])
        fa, md = tensor_to_fa_md(eigs)
        assert fa > 0.9
        assert md > 0

    def test_fa_range(self):
        """FA should always be in [0, 1]."""
        key = np.random.RandomState(42)
        for _ in range(100):
            eigs = jnp.array(sorted(key.uniform(0.1e-9, 3e-9, 3), reverse=True))
            fa, _ = tensor_to_fa_md(eigs)
            assert 0 <= float(fa) <= 1.0

    def test_ad_rd(self):
        eigs = jnp.array([3e-9, 1e-9, 1e-9])
        ad, rd = eigenvalues_to_ad_rd(eigs)
        assert jnp.abs(ad - 3e-9) < 1e-15
        assert jnp.abs(rd - 1e-9) < 1e-15

    def test_batched_fa_md(self):
        eigs = jnp.array([[1e-9, 1e-9, 1e-9], [3e-9, 0.1e-9, 0.1e-9]])
        fa, md = tensor_to_fa_md(eigs)
        assert fa.shape == (2,)
        assert jnp.abs(fa[0]) < 1e-5
        assert fa[1] > 0.9


class TestNODDIMetrics:

    def test_watson_odi_high_kappa(self):
        """High kappa (concentrated) should give low ODI."""
        odi = watson_kappa_to_odi(jnp.array(100.0))
        assert float(odi) < 0.1

    def test_watson_odi_low_kappa(self):
        """Low kappa (dispersed) should give high ODI."""
        odi = watson_kappa_to_odi(jnp.array(0.1))
        assert float(odi) > 0.5

    def test_bingham_odi(self):
        odi = bingham_kappas_to_odi(jnp.array(10.0), jnp.array(10.0))
        assert 0 < float(odi) < 1


class TestMultiCompartmentMetrics:

    def test_effective_eigenvalues(self):
        fracs = jnp.array([0.5, 0.5])
        d_pars = jnp.array([2e-9, 1e-9])
        d_perps = jnp.array([0.5e-9, 0.5e-9])
        eigs = multi_compartment_to_eigenvalues(fracs, d_pars, d_perps)
        assert eigs.shape == (3,)
        assert jnp.abs(eigs[0] - 1.5e-9) < 1e-15

    def test_mean_kurtosis(self):
        fracs = jnp.array([0.5, 0.5])
        d_pars = jnp.array([2e-9, 0.5e-9])
        d_perps = jnp.array([1e-9, 0.3e-9])
        mk = multi_compartment_to_mean_kurtosis(fracs, d_pars, d_perps)
        assert float(mk) > 0

    def test_single_compartment_zero_kurtosis(self):
        fracs = jnp.array([1.0])
        d_pars = jnp.array([1.7e-9])
        d_perps = jnp.array([0.5e-9])
        mk = multi_compartment_to_mean_kurtosis(fracs, d_pars, d_perps)
        assert jnp.abs(mk) < 1e-10


class TestDKIMetrics:

    def test_axial_kurtosis(self):
        """Two compartments with different d_par should give AK > 0."""
        fracs = jnp.array([0.5, 0.5])
        d_pars = jnp.array([2e-9, 0.5e-9])
        ak = multi_compartment_to_axial_kurtosis(fracs, d_pars)
        assert float(ak) > 0

    def test_axial_kurtosis_single_compartment(self):
        fracs = jnp.array([1.0])
        d_pars = jnp.array([1.7e-9])
        ak = multi_compartment_to_axial_kurtosis(fracs, d_pars)
        assert jnp.abs(ak) < 1e-10

    def test_radial_kurtosis(self):
        """Two compartments with different d_perp should give RK > 0."""
        fracs = jnp.array([0.5, 0.5])
        d_perps = jnp.array([1e-9, 0.2e-9])
        rk = multi_compartment_to_radial_kurtosis(fracs, d_perps)
        assert float(rk) > 0

    def test_radial_kurtosis_equal(self):
        """Equal d_perps → RK = 0."""
        fracs = jnp.array([0.5, 0.5])
        d_perps = jnp.array([0.5e-9, 0.5e-9])
        rk = multi_compartment_to_radial_kurtosis(fracs, d_perps)
        assert jnp.abs(rk) < 1e-10

    def test_kurtosis_fa_isotropic_kurtosis(self):
        """If AK == RK, KFA should be near 0."""
        fracs = jnp.array([0.5, 0.5])
        # Same ratio of heterogeneity in all directions
        d_pars = jnp.array([2e-9, 1e-9])
        d_perps = jnp.array([2e-9, 1e-9])
        kfa = multi_compartment_to_kurtosis_fa(fracs, d_pars, d_perps)
        assert float(kfa) < 0.1

    def test_kurtosis_fa_anisotropic(self):
        """If AK >> RK, KFA should be high."""
        fracs = jnp.array([0.5, 0.5])
        d_pars = jnp.array([3e-9, 0.1e-9])   # very heterogeneous axially
        d_perps = jnp.array([0.5e-9, 0.5e-9])  # homogeneous radially
        kfa = multi_compartment_to_kurtosis_fa(fracs, d_pars, d_perps)
        assert float(kfa) > 0.3

    def test_micro_fa_isotropic(self):
        """Isotropic compartments should give micro-FA ≈ 0."""
        fracs = jnp.array([0.5, 0.5])
        d_pars = jnp.array([1e-9, 1e-9])
        d_perps = jnp.array([1e-9, 1e-9])
        ufa = micro_fa(fracs, d_pars, d_perps)
        assert float(ufa) < 0.01

    def test_micro_fa_anisotropic(self):
        """Highly anisotropic compartments should give high micro-FA."""
        fracs = jnp.array([0.5, 0.5])
        d_pars = jnp.array([2e-9, 2e-9])
        d_perps = jnp.array([0.1e-9, 0.1e-9])
        ufa = micro_fa(fracs, d_pars, d_perps)
        assert float(ufa) > 0.8

    def test_micro_fa_range(self):
        """micro-FA should be in [0, 1]."""
        key = np.random.RandomState(42)
        for _ in range(50):
            n_comp = key.randint(2, 5)
            fracs = key.dirichlet(np.ones(n_comp))
            d_pars = key.uniform(0.1e-9, 3e-9, n_comp)
            d_perps = key.uniform(0.1e-9, d_pars)
            ufa = micro_fa(jnp.array(fracs), jnp.array(d_pars), jnp.array(d_perps))
            assert 0 <= float(ufa) <= 1.0


class TestMultiMetricExtractor:

    def test_dti_extraction(self):
        params = {
            "lambda_par": np.array([2e-9, 1.5e-9]),
            "lambda_perp": np.array([0.5e-9, 0.5e-9]),
        }
        extractor = MultiMetricExtractor(["lambda_par", "lambda_perp"])
        metrics = extractor.extract(params)
        assert "FA" in metrics
        assert "MD" in metrics
        assert "AD" in metrics
        assert "RD" in metrics
        assert metrics["FA"].shape == (2,)

    def test_noddi_extraction(self):
        params = {
            "f_intra": np.array([0.5]),
            "f_iso": np.array([0.1]),
            "kappa": np.array([10.0]),
        }
        extractor = MultiMetricExtractor(["f_intra", "f_iso", "kappa"])
        metrics = extractor.extract(params)
        assert "NDI" in metrics
        assert "FW" in metrics
        assert "ODI" in metrics

    def test_dki_extraction_from_compartments(self):
        """DKI metrics from multi-compartment partial_volume/lambda naming."""
        params = {
            "partial_volume_0": np.array([0.6, 0.4]),
            "partial_volume_1": np.array([0.4, 0.6]),
            "lambda_par": np.array([1.7e-9, 1.7e-9]),
            "lambda_par_2": np.array([1.0e-9, 1.0e-9]),
            "lambda_perp": np.array([0.5e-9, 0.5e-9]),
            "lambda_perp_2": np.array([0.3e-9, 0.3e-9]),
        }
        extractor = MultiMetricExtractor(list(params.keys()))
        metrics = extractor.extract(params)
        assert "MK" in metrics
        assert "AK" in metrics
        assert "RK" in metrics
        assert "KFA" in metrics
        assert "micro_FA" in metrics
        assert metrics["MK"].shape == (2,)
        assert np.all(metrics["MK"] > 0)

    def test_tissue_segmentation_from_fractions(self):
        """Explicit fraction names should produce WM/GM/CSF maps."""
        params = {
            "f_wm": np.array([0.7, 0.1, 0.0]),
            "f_gm": np.array([0.2, 0.8, 0.0]),
            "f_csf": np.array([0.1, 0.1, 1.0]),
        }
        extractor = MultiMetricExtractor(list(params.keys()))
        metrics = extractor.extract(params)
        assert "WM" in metrics
        assert "GM" in metrics
        assert "CSF" in metrics
        np.testing.assert_array_equal(metrics["WM"], params["f_wm"])

    def test_tissue_segmentation_from_thresholds(self):
        """Threshold-based segmentation from FA and MD."""
        params = {
            "lambda_par": np.array([2e-9, 1e-9, 3.5e-9]),
            "lambda_perp": np.array([0.3e-9, 0.9e-9, 3.5e-9]),
        }
        extractor = MultiMetricExtractor(list(params.keys()))
        metrics = extractor.extract(params)
        # Voxel 0: high FA (~0.9), moderate MD → WM
        # Voxel 1: low FA (~0.1), moderate MD → GM
        # Voxel 2: isotropic, high MD → CSF
        assert "WM" in metrics
        assert "GM" in metrics
        assert "CSF" in metrics
        assert metrics["WM"][0] == 1.0  # WM
        assert metrics["GM"][1] == 1.0  # GM
        assert metrics["CSF"][2] == 1.0  # CSF

    def test_dki_extraction_from_explicit_arrays(self):
        """DKI from explicit fractions/d_pars/d_perps arrays."""
        params = {
            "fractions": np.array([[0.5, 0.5], [0.3, 0.7]]),
            "d_pars": np.array([[2e-9, 1e-9], [1.5e-9, 0.8e-9]]),
            "d_perps": np.array([[0.5e-9, 0.3e-9], [0.4e-9, 0.2e-9]]),
        }
        extractor = MultiMetricExtractor(list(params.keys()))
        metrics = extractor.extract(params)
        assert "MK" in metrics
        assert "AK" in metrics
        assert metrics["MK"].shape == (2,)
