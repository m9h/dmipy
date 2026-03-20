"""
Configuration dataclass for the SBI pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable


@dataclass
class SBIPipelineConfig:
    """Full configuration for an SBI training and deployment run.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. ``"DTI"``, ``"NODDI"``).
    parameter_names : list of str
        Names of the inferred parameters in order.
    parameter_ranges : dict
        ``{name: (low, high)}`` bounds used for prior sampling and output
        de-normalisation.
    acquisition : dict
        Serialisable acquisition spec.  At minimum must contain ``bvalues``
        and ``gradient_directions`` as array-likes.  May also carry ``delta``,
        ``Delta``, etc.
    inference_mode : str
        ``"mdn"`` for Mixture Density Network or ``"flow"`` for normalising
        flow (NPE via FlowJAX).
    noise_type : str
        ``"rician"`` (default) or ``"gaussian"``.
    snr : float
        Signal-to-noise ratio at b = 0.  Noise sigma = 1 / snr.
    n_components : int
        Number of Gaussian mixture components (MDN mode only).
    hidden_dim : int
        Width of hidden layers.
    depth : int
        Number of hidden layers (MDN) or flow layers (flow).
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Samples per training step.
    n_steps : int
        Total training iterations.
    checkpoint_path : str or None
        Where to save / load checkpoints.
    constraints : dict or None
        Optional mapping of constraint names to callables or descriptors,
        e.g. ``{"tortuosity": {"d_perp": "d_par * (1 - f_intra)"}}``.
    seed : int
        Base PRNG seed.
    """

    # Model specification
    model_name: str = "DTI"
    parameter_names: List[str] = field(default_factory=lambda: ["FA", "MD"])
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Acquisition
    acquisition: Dict[str, Any] = field(default_factory=dict)

    # Network architecture
    inference_mode: str = "mdn"  # "mdn" or "flow"
    architecture: str = "mlp"  # "mlp" or "residual"
    n_components: int = 8
    hidden_dim: int = 128
    depth: int = 3

    # Noise
    noise_type: str = "rician"
    snr: float = 30.0
    snr_range: Optional[Tuple[float, float]] = None  # e.g. (5, 80); if set, overrides fixed snr
    curriculum_noise: bool = False  # if True, start with high SNR and linearly decrease to snr_range[0]

    # Training hyper-parameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_steps: int = 5000

    # LR scheduling
    lr_schedule: str = "cosine"  # "constant", "cosine", "warmup_cosine"
    warmup_steps: int = 500

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Architecture
    activation: str = "gelu"  # "relu", "gelu", "silu"

    # Validation / early stopping
    val_fraction: float = 0.1  # fraction of each batch used for validation tracking
    patience: int = 0  # 0 = disabled; stop if val loss hasn't improved in this many steps

    # Checkpointing
    checkpoint_path: Optional[str] = None

    # Constraints
    constraints: Optional[Dict[str, Any]] = None

    # Reproducibility
    seed: int = 0

    # ---- helpers ----

    @property
    def noise_sigma(self) -> float:
        return 1.0 / self.snr

    @property
    def theta_dim(self) -> int:
        return len(self.parameter_names)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SBIPipelineConfig":
        """Reconstruct from a plain dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
