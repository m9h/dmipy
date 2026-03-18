"""
Bingham-NODDI model card for SBI pipeline.

Defines prior spec, tortuosity constraint, and noise config for training
a Bingham-NODDI posterior with the unified SBI pipeline.

Free parameters: f_intra, f_iso, kappa1, kappa2, mu (5-D posterior).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.signal_models.bingham import BinghamNODDI
from dmipy_jax.acquisition import JaxAcquisition


# Fixed biophysical constants
D_PAR = 1.7e-9   # m^2/s  (intra-axonal parallel diffusivity)
D_ISO = 3.0e-9   # m^2/s  (free water / CSF)


def get_bingham_noddi_config(
    acquisition: JaxAcquisition,
    *,
    snr: float = 30.0,
    n_steps: int = 5000,
    inference_mode: str = "mdn",
) -> SBIPipelineConfig:
    """Build a pipeline config for Bingham-NODDI SBI.

    Parameters
    ----------
    acquisition : JaxAcquisition
        Multi-shell acquisition scheme (b = 700, 2000 recommended).
    snr : float
    n_steps : int
    inference_mode : str  ``"mdn"`` or ``"flow"``
    """
    acq_dict = {
        "bvalues": acquisition.bvalues.tolist(),
        "gradient_directions": acquisition.gradient_directions.tolist(),
    }
    if acquisition.delta is not None:
        acq_dict["delta"] = float(acquisition.delta)
    if acquisition.Delta is not None:
        acq_dict["Delta"] = float(acquisition.Delta)

    return SBIPipelineConfig(
        model_name="BinghamNODDI",
        parameter_names=["f_intra", "f_iso", "kappa1", "kappa2", "mu_theta"],
        parameter_ranges={
            "f_intra": (0.01, 0.99),
            "f_iso": (0.0, 0.99),
            "kappa1": (0.1, 128.0),
            "kappa2": (0.1, 128.0),
            "mu_theta": (0.0, jnp.pi),
        },
        acquisition=acq_dict,
        inference_mode=inference_mode,
        n_components=8,
        hidden_dim=256,
        depth=4,
        noise_type="rician",
        snr=snr,
        learning_rate=5e-4,
        batch_size=512,
        n_steps=n_steps,
        constraints={"tortuosity": "d_perp = d_par * (1 - f_intra)"},
    )


def build_bingham_noddi_simulator(
    acquisition: JaxAcquisition,
    snr: float = 30.0,
    grid_points: int = 100,
) -> ModelSimulator:
    """Build a ``ModelSimulator`` for the Bingham-NODDI model.

    The simulator maps a 5-D parameter vector
    ``[f_intra, f_iso, kappa1, kappa2, mu_theta]`` to a multi-shell signal.

    Parameters
    ----------
    acquisition : JaxAcquisition
    snr : float
    grid_points : int
        Bingham integration grid density.
    """
    bingham_model = BinghamNODDI(grid_points=grid_points)

    def forward_fn(params_flat, acq):
        f_intra = params_flat[0]
        f_iso = params_flat[1]
        kappa1 = params_flat[2]
        kappa2 = params_flat[3]
        mu_theta = params_flat[4]
        mu_phi = 0.0  # Fix phi to 0 (symmetry)

        mu = jnp.array([mu_theta, mu_phi])

        # Tortuosity constraint
        d_perp = D_PAR * (1.0 - f_intra)

        # Intra-cellular: Bingham-dispersed stick
        s_intra = bingham_model(
            acq.bvalues,
            acq.gradient_directions,
            mu=mu,
            kappa1=kappa1,
            kappa2=kappa2,
            lambda_par=D_PAR,
        )

        # Extra-cellular: isotropic Zeppelin (simplified)
        # S_extra = exp(-b * (d_par * cos^2 + d_perp * sin^2))
        # For the dispersed case we use a simple isotropic average
        d_mean_extra = (D_PAR + 2 * d_perp) / 3.0
        s_extra = jnp.exp(-acq.bvalues * d_mean_extra)

        # CSF / free water
        s_iso = jnp.exp(-acq.bvalues * D_ISO)

        # Volume fractions
        f_ec = (1.0 - f_iso) * (1.0 - f_intra)
        f_ic = (1.0 - f_iso) * f_intra

        signal = f_ic * s_intra + f_ec * s_extra + f_iso * s_iso
        return signal

    def prior_sampler_fn(key, n):
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        f_intra = jr.beta(k1, 5.0, 5.0, shape=(n,))
        f_iso = jr.beta(k2, 0.5, 5.0, shape=(n,))
        # Log-uniform for kappas
        log_k1 = jr.uniform(k3, (n,), minval=jnp.log(0.1), maxval=jnp.log(128.0))
        log_k2 = jr.uniform(k4, (n,), minval=jnp.log(0.1), maxval=jnp.log(128.0))
        kappa1 = jnp.exp(log_k1)
        kappa2 = jnp.exp(log_k2)
        mu_theta = jr.uniform(k5, (n,), minval=0.0, maxval=jnp.pi)
        return jnp.stack([f_intra, f_iso, kappa1, kappa2, mu_theta], axis=-1)

    parameter_names = ["f_intra", "f_iso", "kappa1", "kappa2", "mu_theta"]
    parameter_ranges = {
        "f_intra": (0.01, 0.99),
        "f_iso": (0.0, 0.99),
        "kappa1": (0.1, 128.0),
        "kappa2": (0.1, 128.0),
        "mu_theta": (0.0, float(jnp.pi)),
    }

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=parameter_names,
        parameter_ranges=parameter_ranges,
        acquisition=acquisition,
        noise_type="rician",
        snr=snr,
        prior_sampler_fn=prior_sampler_fn,
    )
