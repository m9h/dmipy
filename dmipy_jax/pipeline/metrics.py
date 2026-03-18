"""
Phase 4: Unified multi-metric output from fitted / matched parameters.

Computes DTI (FA, MD, AD, RD), DKI (MK, AK, RK, KFA, micro-FA),
NODDI (NDI, ODI, FW), tissue segmentation (WM, GM, CSF),
and fiber peaks from multi-compartment parameter maps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from dmipy_jax.library.derived_metrics import (
    tensor_to_fa_md,
    eigenvalues_to_ad_rd,
    watson_kappa_to_odi,
    bingham_kappas_to_odi,
    multi_compartment_to_mean_kurtosis,
    multi_compartment_to_axial_kurtosis,
    multi_compartment_to_radial_kurtosis,
    multi_compartment_to_kurtosis_fa,
    micro_fa,
)


class MultiMetricExtractor:
    """Extract derived diffusion metrics from parameter maps.

    Parameters
    ----------
    parameter_names : list of str
        Names of parameters in the fitted results dict.
    """

    def __init__(self, parameter_names: List[str]):
        self.parameter_names = list(parameter_names)

    def extract(
        self,
        params: Dict[str, np.ndarray],
        vol_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute all available metrics from the parameter dictionary.

        Parameters
        ----------
        params : dict
            ``{name: np.ndarray}`` — may be 1-D (flat) or 3-D volumes.
        vol_shape : tuple, optional
            Spatial shape; used only for reshaping if needed.

        Returns
        -------
        metrics : dict ``{metric_name: np.ndarray}``
        """
        metrics: Dict[str, np.ndarray] = {}

        # ---- DTI --------------------------------------------------------- #
        self._try_dti(params, metrics)

        # ---- DKI --------------------------------------------------------- #
        self._try_dki(params, metrics)

        # ---- NODDI / Watson ---------------------------------------------- #
        self._try_noddi(params, metrics)

        # ---- Bingham ----------------------------------------------------- #
        self._try_bingham(params, metrics)

        # ---- Tissue segmentation ----------------------------------------- #
        self._try_tissue_segmentation(params, metrics)

        return metrics

    # ------------------------------------------------------------------ #
    # Metric-group helpers
    # ------------------------------------------------------------------ #

    def _try_dti(self, params, out):
        """Attempt to compute FA, MD, AD, RD from eigenvalue-like params."""
        eigenvalue_sets = [
            ("lambda_1", "lambda_2", "lambda_3"),
            ("FA", "MD"),  # already computed
        ]

        for names in eigenvalue_sets:
            if all(n in params for n in names):
                if names == ("FA", "MD"):
                    return
                eigs = np.stack([params[n] for n in names], axis=-1)
                fa, md = tensor_to_fa_md(jnp.array(eigs))
                ad, rd = eigenvalues_to_ad_rd(jnp.array(eigs))
                out["FA"] = np.asarray(fa)
                out["MD"] = np.asarray(md)
                out["AD"] = np.asarray(ad)
                out["RD"] = np.asarray(rd)
                return

        # Try from generic diffusivity names
        if "lambda_par" in params and "lambda_perp" in params:
            lp = np.asarray(params["lambda_par"])
            lt = np.asarray(params["lambda_perp"])
            eigs = np.stack([lp, lt, lt], axis=-1)
            fa, md = tensor_to_fa_md(jnp.array(eigs))
            ad, rd = eigenvalues_to_ad_rd(jnp.array(eigs))
            out["FA"] = np.asarray(fa)
            out["MD"] = np.asarray(md)
            out["AD"] = np.asarray(ad)
            out["RD"] = np.asarray(rd)

    def _try_dki(self, params, out):
        """Attempt to compute DKI metrics (MK, AK, RK, KFA, micro-FA).

        Requires multi-compartment parameters: per-compartment fractions
        and diffusivities.  Looks for naming patterns from
        ``JaxMultiCompartmentModel`` or explicit ``fractions`` / ``d_pars``
        / ``d_perps`` arrays.
        """
        fracs, d_pars, d_perps = self._gather_compartment_diffusivities(params)
        if fracs is None:
            return

        fracs_j = jnp.array(fracs)
        d_pars_j = jnp.array(d_pars)
        d_perps_j = jnp.array(d_perps)

        out["MK"] = np.asarray(
            multi_compartment_to_mean_kurtosis(fracs_j, d_pars_j, d_perps_j)
        )
        out["AK"] = np.asarray(
            multi_compartment_to_axial_kurtosis(fracs_j, d_pars_j)
        )
        out["RK"] = np.asarray(
            multi_compartment_to_radial_kurtosis(fracs_j, d_perps_j)
        )
        out["KFA"] = np.asarray(
            multi_compartment_to_kurtosis_fa(fracs_j, d_pars_j, d_perps_j)
        )
        out["micro_FA"] = np.asarray(
            micro_fa(fracs_j, d_pars_j, d_perps_j)
        )

    def _try_noddi(self, params, out):
        """Attempt to compute NDI, ODI, FW from NODDI-like params."""
        for name in ("f_intra", "partial_volume_0"):
            if name in params:
                out["NDI"] = np.asarray(params[name])
                break

        for name in ("f_iso", "partial_volume_2"):
            if name in params:
                out["FW"] = np.asarray(params[name])
                break

        for name in ("kappa", "kappa_1"):
            if name in params:
                odi = watson_kappa_to_odi(jnp.array(params[name]))
                out["ODI"] = np.asarray(odi)
                break

    def _try_bingham(self, params, out):
        """Attempt to compute ODI from Bingham kappas."""
        if "kappa1" in params and "kappa2" in params:
            odi = bingham_kappas_to_odi(
                jnp.array(params["kappa1"]),
                jnp.array(params["kappa2"]),
            )
            out["ODI_bingham"] = np.asarray(odi)

    def _try_tissue_segmentation(self, params, out):
        """Estimate WM / GM / CSF tissue fractions from microstructure params.

        Uses heuristic thresholds on FA and MD (or direct volume fractions
        when available) following the FORCE approach of reading tissue class
        directly from the matched multi-compartment entry.

        Classification rules (when volume fractions are not explicit):
          - **CSF**: MD > 2.5e-9  m^2/s
          - **GM**:  FA < 0.2  AND  MD in [0.5e-9, 2.5e-9]
          - **WM**:  FA >= 0.2  AND  MD in [0.3e-9, 2.5e-9]

        These are deliberately conservative; the primary purpose is to produce
        approximate tissue masks for QC and downstream parcellation, not to
        replace dedicated segmentation tools (e.g. FAST / SynthSeg).
        """
        # Path 1: explicit volume fractions (FORCE-style)
        frac_names_wm = ("f_wm", "partial_volume_wm")
        frac_names_gm = ("f_gm", "partial_volume_gm")
        frac_names_csf = ("f_csf", "f_iso", "partial_volume_2")

        has_explicit = False
        for name in frac_names_wm:
            if name in params:
                out["WM"] = np.asarray(params[name])
                has_explicit = True
                break
        for name in frac_names_gm:
            if name in params:
                out["GM"] = np.asarray(params[name])
                has_explicit = True
                break
        for name in frac_names_csf:
            if name in params:
                out["CSF"] = np.asarray(params[name])
                has_explicit = True
                break

        if has_explicit:
            return

        # Path 2: threshold-based from FA and MD
        fa = out.get("FA")
        md = out.get("MD")
        if fa is None or md is None:
            return

        fa = np.asarray(fa)
        md = np.asarray(md)

        csf_mask = md > 2.5e-9
        gm_mask = (~csf_mask) & (fa < 0.2) & (md >= 0.5e-9)
        wm_mask = (~csf_mask) & (fa >= 0.2)

        out["WM"] = wm_mask.astype(np.float32)
        out["GM"] = gm_mask.astype(np.float32)
        out["CSF"] = csf_mask.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Compartment gathering
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gather_compartment_diffusivities(params):
        """Try to assemble (fractions, d_pars, d_perps) arrays.

        Returns ``(None, None, None)`` if insufficient data.

        Looks for:
          - Explicit arrays ``fractions``, ``d_pars``, ``d_perps``
          - ``partial_volume_*`` + ``lambda_par*`` + ``lambda_perp*`` patterns
        """
        # Path 1: explicit arrays
        if all(k in params for k in ("fractions", "d_pars", "d_perps")):
            return params["fractions"], params["d_pars"], params["d_perps"]

        # Path 2: MCM naming convention
        pv_keys = sorted(k for k in params if k.startswith("partial_volume_"))
        lp_keys = sorted(k for k in params if k.startswith("lambda_par"))
        lt_keys = sorted(k for k in params if k.startswith("lambda_perp"))

        if len(pv_keys) < 2 or len(lp_keys) < 1:
            return None, None, None

        # Need equal number of compartments across all arrays
        n_comp = min(len(pv_keys), len(lp_keys))
        if len(lt_keys) < n_comp:
            # Fall back: assume axially symmetric with d_perp = 0 for sticks
            lt_vals = [np.asarray(params[k]) for k in lt_keys]
            while len(lt_vals) < n_comp:
                lt_vals.append(np.zeros_like(lt_vals[0]))
        else:
            lt_vals = [np.asarray(params[k]) for k in lt_keys[:n_comp]]

        fracs = np.stack([np.asarray(params[k]) for k in pv_keys[:n_comp]], axis=-1)
        d_pars = np.stack([np.asarray(params[k]) for k in lp_keys[:n_comp]], axis=-1)
        d_perps = np.stack(lt_vals, axis=-1)

        return fracs, d_pars, d_perps
