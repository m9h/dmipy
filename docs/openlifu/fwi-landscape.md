# Full Waveform Inversion Landscape for Ultrasound

## The LANL→UNC Pipeline (Lin + Wohlberg)

The key research group bridging geophysical FWI and medical ultrasound CT is **Youzuo Lin** (formerly LANL, now UNC Chapel Hill SMILE lab) and **Brendt Wohlberg** (LANL, SCICO developer). Their work spans:

### OpenFWI (NeurIPS 2022)
- **Repo:** github.com/lanl/OpenFWI (166 stars, BSD-3)
- 12 large-scale benchmark datasets for ML-driven seismic FWI (~2.1 TB)
- InversionNet (encoder-decoder CNN), VelocityGAN (WGAN-GP), UPFWI baselines
- PyTorch. 2D acoustic wave equation. 70x70 grids.
- **No differentiable forward solver included** — purely data-driven.
- Forward data generated via 2-8 FD solver (KAUST MATLAB code, not released).

### OpenPros (ICLR 2026)
- First large-scale prostate USCT benchmark: 280,000+ paired samples
- Open-source FDTD + Runge-Kutta acoustic solvers
- Speed-of-sound reconstruction from limited-view ultrasound
- Clinical MRI/CT + ex vivo specimens

### BrainPuzzle (SPIE 2024)
- **Transcranial ultrasound tomography** — directly relevant
- Hybrid physics + ML: reverse time migration + transformer super-resolution
- Addresses skull aberration problem

### Learned FWI for USCT (IEEE TCI 2024)
- Lozenski, Wang, Li, Anastasio, **Wohlberg**, **Lin**, Villa
- InversionNet adapted for breast USCT
- Task-informed loss for tumor detection
- Forward model via **Devito** (not j-Wave)
- 3 orders of magnitude faster than iterative FWI

### Survey (arXiv 2024)
- Lin, Feng, Theiler, Chen, Villa, Rao, Greenhall, Pantea, Anastasio, **Wohlberg**
- arXiv:2410.08329 — comprehensive review unifying seismic, NDE, and medical USCT

---

## SCICO (JAX Inverse Problem Framework)

**Repo:** github.com/lanl/scico — v0.0.7 (Dec 2025). JOSS paper: DOI 10.21105/joss.04722.

SCICO is a JAX-native library from LANL for solving inverse problems. It is **already a dependency** of sbi4dwi (`scico>=0.0.6`) and is used in:
- `dmipy_jax/inverse/solvers.py` — MicrostructureOperator as SCICO Operator
- `dmipy_jax/inverse/amico.py` — ADMM solver
- `dmipy_jax/inverse/global_amico.py` — Spatial TV regularization

### SCICO Capabilities for FWI-US

| Feature | SCICO Class | Use in FWI |
|---------|-------------|-----------|
| Nonlinear forward operator | `operator.Operator` | Wrap j-Wave as SCICO operator |
| ADMM | `optimize.ADMM` | Split data fidelity + TV regularization |
| PDHG | `optimize.PDHG` | Primal-dual with nonlinear operators (v0.0.3+) |
| NonLinear PADMM | `optimize.NonLinearPADMM` | Constrained nonlinear inversion |
| Total Variation | `functional.IsotropicTVNorm` | Piecewise-smooth reconstruction |
| L2 data loss | `loss.SquaredL2Loss` | ||A(x)-y||² with arbitrary A |
| Non-negativity | `functional.NonNegativeIndicator` | c(r) > 0 constraint |
| PnP denoisers | `functional.BM3D`, `DnCNN` | Plug-and-Play regularization |

### Integration Path for fwi_us.py

```python
from scico.operator import Operator
from scico import functional, loss, optimize, linop

class AcousticForwardOperator(Operator):
    """j-Wave forward model as SCICO nonlinear operator."""
    def _eval(self, sound_speed):
        medium = create_medium(self.domain, sound_speed, self.density, ...)
        return run_simulation_jax(medium, ...)

# FWI with SCICO PDHG + TV
f = loss.SquaredL2Loss(y=observed, A=AcousticForwardOperator(...))
g = tv_weight * functional.IsotropicTVNorm()
C = linop.FiniteDifference(input_shape=grid_shape)
solver = optimize.PDHG(f=f, g=g, C=C, ...)
c_reconstructed = solver.solve()
```

This replaces the manual optax loop in fwi_us.py with proven proximal splitting, automatic adjoint computation, and convergence guarantees.

---

## SMILE Lab (UNC Chapel Hill)

**PI:** Youzuo Lin (yzlin@unc.edu). **Lab:** smileunc.github.io
**SMILE = Scientific Machine Intelligence Learning Education**

### Key People
- Youzuo Lin (PI, formerly LANL Senior Scientist)
- Hanchen Wang (PostDoc → Amazon)
- Shihang Feng (→ CGG)
- Shengyu Chen (BrainPuzzle first author)
- Luke Lozenski (learned FWI for USCT)
- Collaborators: Brendt Wohlberg (LANL/SCICO), Mark Anastasio (UIUC), Umberto Villa (UT Austin), Emad Boctor (JHU)

### Simulation Methods
- FDTD (4th-order spatial, 2nd-order temporal) — OpenPros
- Runge-Kutta implicit iterative solver — OpenPros alternative
- 2D isotropic acoustic wave equation
- Neural network baselines: InversionNet, VelocityGAN, ViT-Inversion
- **No j-Wave usage** — uses custom LANL-heritage FDTD

### Relevance to Our Work
1. **BrainPuzzle** tackles transcranial skull aberration with hybrid physics+ML — same problem as our j-Wave pipeline
2. **OpenPros** provides open-source USCT forward solvers + 280K training samples
3. **InversionNet** architecture transfers to ultrasound: input=multi-channel waveforms → output=tissue property map
4. **Survey paper** (arXiv:2410.08329) provides the theoretical framework connecting all these approaches

---

## MIDA Head Model (IT'IS Foundation)

- 115 segmented structures at 0.5mm isotropic
- **5 CHF (~$5.50) handling fee** via PayPal
- URL: itis.swiss/virtual-population/regional-human-models/mida-model/
- Reference: Iacono et al., PLoS ONE 10(4):e0124126, 2015
- Includes: detailed skull layers (outer table, diploe, inner table), all cranial nerves, deep brain structures, vasculature
- Formats: NIfTI, MAT, RAW, STL

### Free Alternative: SimNIBS Ernie
- 10 tissue types including compact/spongy bone
- Direct download: github.com/simnibs/example-dataset
- CHARM pipeline for generating custom head models from T1+T2

---

## Device Array Geometries

### Oxford-UCL 256-Element Helmet (Martin, Stagg, Treeby 2025)
- Semi-ellipsoidal: 206mm L × 157mm W × 96mm H
- 256 individually controllable elements at 555kHz
- Random element distribution (minimize grating lobes)
- 3mm diameter apertures
- Focal: 1.3mm lateral × 3.4mm axial (-3dB)

### Kernel Flow2 (40 modules, fNIRS)
- 6 plates: Front, Top, Left, Right, Left Nape, Right Nape
- 40 module positions (52 sources, 312 detectors in Flow1)
- 10mm source-detector separation within modules
- Optode coordinates extractable via SolidWorks API + Python script
