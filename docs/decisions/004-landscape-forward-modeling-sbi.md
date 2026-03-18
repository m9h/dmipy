# Landscape Analysis: Forward Modeling & SBI for Diffusion MRI Microstructure

## Date
2025-03-17

## Status
Research survey — informing dmipy-JAX roadmap

## Summary

This document surveys three recent bodies of work that converge on the same core idea:
**use forward simulation of diffusion MRI signals as the primary inference engine**,
replacing or augmenting classical inverse fitting. We evaluate overlap with dmipy-JAX
and identify concrete opportunities.

The three projects are:

1. **FORCE** (Indiana/DIPY) — dictionary-based cosine-similarity matching
2. **SBI for dMRI** (Nottingham/CoNI Lab) — neural posterior estimation for Ball-and-Sticks
3. **SBIDTI** (Alicante/CSIC-UMH) — neural posterior estimation for DTI, DKI, and AxCaliber

A fourth relevant tool, **cuDIMOT** (Nottingham/CoNI Lab), provides CUDA-accelerated
classical fitting with Bingham-NODDI models.

---

## 1. FORCE — FORward modeling for Complex microstructure Estimation

**Paper:** Shah AJ, Henriques RN, Ramirez-Manzanares A, Filipiak P, Baete S, Deka K,
Gor M, Koudoro S, Garyfallidis E. Research Square preprint, November 2025.
DOI: 10.21203/rs.3.rs-8151109/v1

**Code:** https://github.com/Atharva-Shah-2298/FORCE (to be integrated into DIPY)

### What it does

FORCE replaces inverse fitting with forward simulation + signal-space matching:

1. **Simulate** a library of 500K biologically plausible voxel configurations using
   stick + zeppelin + Bingham orientation distribution + gray matter ball + free water ball.
   Each configuration is a mixture of up to 3 fiber populations with tissue fractions
   drawn from Dirichlet(2,1,1), orientations sampled on a 724-vertex electrostatic grid
   (~4.1 degree angular resolution), and biophysical parameters from literature-informed
   uniform/equispaced priors.

2. **Match** each measured voxel to the library via penalized cosine similarity:
   `i_hat = argmax [ cos(S_voxel, S_i) - alpha * n_fibers(i) ]` with alpha = 10^-5.
   Top K=50 candidates retained. Accelerated variant (FORCE-ACC) uses locality-sensitive
   hashing with Hadamard projection for approximate nearest-neighbor search.

3. **Read off** all parameters from the matched simulation entry. DTI and DKI tensors
   are computed analytically from the multi-compartment mixture (closed-form in Appendix A).

### What it outputs (from a single matching operation)

- Multi-fiber tractography peaks (up to 3 fibers per voxel)
- DTI metrics (FA, MD, AD, RD)
- DKI metrics (MK, AK, RK, KFA, micro-FA)
- NODDI-like metrics (NDI, ODI, FW fraction)
- Tissue segmentation maps (WM, GM, CSF volume fractions)
- Uncertainty maps (IQR of K-nearest-neighbor similarity) and ambiguity maps (FWHM)

### Key results

- **Synthetic:** Highest and most uniform peak-detection rates across all crossing-angle
  bins (10-90 degrees), especially at shallow crossings (10-40 degrees) where ODF-based
  methods (CSA, CSD, GQI, ODFFP) fail.
- **HCP in vivo:** DTI metrics virtually identical to conventional fitting. DKI maps
  smoother and more anatomically consistent (especially RK). NODDI ODI correlation with
  inverted T1w: r=0.93 (FORCE) vs r=0.82 (AMICO). Cleaner FW maps than AMICO.
- **Ex vivo:** Reproduced DTI contrasts on mouse brain (55 um) and generated NODDI maps
  from single-shell data.
- **Clinical:** Glioma and Parkinson's disease applications validated.
- **Runtime:** ~15 min/subject (HCP) on CPU, ~929s on GPU. Competitive with running
  DTI + DKI + NODDI + CSD separately.

### Architecture

- Simulation: Cython extensions with OpenMP via ProcessPoolExecutor, memory-mapped arrays.
- Matching: FAISS (CPU or GPU). FORCE-ACC uses LSH with Hadamard projection.
- Language: Python 3.8+, Cython, numpy, scipy, nibabel, dipy.
- No differentiability. No gradient-based refinement possible.

### Limitations

- Fixed biophysical model (stick + zeppelin + Bingham + GM + FW). Cannot estimate axon
  diameter, soma density, or perfusion.
- Discrete parameter space bounded by library resolution (~4.1 degree angular, finite
  parameter grid). Cannot interpolate between library entries.
- Uncertainty metrics are heuristic (K-NN similarity spread), not proper Bayesian posteriors.
- Library must be regenerated for each acquisition protocol.
- Memory-intensive (500K entries x signal dimensionality).

---

## 2. SBI for dMRI — Nottingham (CoNI Lab)

**Paper:** Manzano-Patron JP, Deistler M, Schroder C, Kypraios T, Goncalves PJ,
Macke JH, Sotiropoulos SN. "Uncertainty mapping and probabilistic tractography using
Simulation-Based Inference in diffusion MRI." Medical Image Analysis 103:103580, 2025.
DOI: 10.1016/j.media.2025.103580

**Code:** https://github.com/SPMIC-UoN/SBI_dMRI

### What it does

Neural Posterior Estimation (NPE) for the Ball-and-Sticks model family:

1. **Forward model:** Multi-compartment Ball-and-Sticks (1-3 fiber populations).
   Isotropic "ball" + N anisotropic "sticks" with gamma-distributed diffusivities
   for multi-shell variants.

2. **Training data generation:** 2M-6M parameter-signal pairs from carefully designed
   restricted priors. A classifier identifies valid parameter combinations (62% speedup
   over rejection sampling). Rician noise at varying SNR levels (3-80).

3. **Inference:** Neural Spline Flows (K=10 transformations) learn the mapping from
   observed diffusion signals to full posterior parameter distributions. Amortized —
   once trained, inference is a single forward pass per voxel.

4. **Model selection:** Two approaches:
   - SBI_ClassiFiber: feed-forward classifier determines fiber count, then appropriate
     NPE model applied.
   - SBI_joint: single NPE trained on all fiber counts; model selection post-hoc via
     volume fraction thresholding (f_cutoff = 5%).

5. **Signal representations:** Acquisition-specific (raw signal) and acquisition-agnostic
   (spherical harmonics L=6 for single-shell, MAP-MRI for multi-shell).

### Key results

- Orders of magnitude speedup over FSL BedpostX MCMC with comparable accuracy.
- Posterior mean correlations r > 0.95 for diffusivity and volume fractions vs MCMC.
- Median orientation differences ~2 degrees (primary fiber), ~6 degrees (secondary).
- Probabilistic tractography scan-rescan reproducibility: SBI median r=0.95-0.96
  vs MCMC median r=0.87.
- 15% higher correlation with UK Biobank population-average atlas than MCMC.

### Architecture

- Python, `sbi` toolbox v0.22, Neural Spline Flows.
- No CUDA/GPU for inference itself (Python neural network forward pass).
- Does NOT use Bingham distributions (Ball-and-Sticks only).
- Does NOT use cosine similarity matching.

---

## 3. cuDIMOT — CUDA Diffusion Modeling Toolbox (Nottingham/CoNI Lab)

**Paper:** Hernandez-Fernandez M, Reguly I, Jbabdi S, Giles M, Smith S, Sotiropoulos SN.
"Using GPUs to accelerate computational diffusion MRI." NeuroImage 188:598-615, 2019.

**Code:** https://github.com/SPMIC-UoN/fdt

### What it does

A model-independent CUDA framework for classical fitting. Users define nonlinear MRI
models via a C header file, compiled into CUDA kernels. Two-level parallelization:
different voxels assigned to CUDA warps (32 threads), within-voxel computations
distributed among threads.

### Key models

- **NODDI-Watson:** Symmetric fiber dispersion.
- **NODDI-Bingham:** Anisotropic fiber dispersion using two concentration parameters
  (kappa_1, kappa_2) plus roll parameter psi. Uses saddlepoint approximation for the
  Bingham hypergeometric function. Optimization pipeline: DTI -> Grid-Search -> LM (x2).
- **Ball-and-Rackets:** Bingham-dispersed sticks (Sotiropoulos et al., 2012).

### Relevance

This is the CUDA Bingham project. It demonstrates that GPU-accelerated Bingham
distribution fitting is tractable and fast (7x over 72-core MATLAB). Pre-compiled
binaries for NODDI-Bingham across CUDA 9.1-12. Relevant as a performance baseline
for any JAX-based Bingham implementation.

---

## 4. SBIDTI — Spanish SBI Work (CSIC-UMH, Alicante)

**Paper:** Eggl MF, De Santis S. "Simulation-Based Inference at the Theoretical Limit:
Fast, Accurate Microstructural MRI with Minimal diffusion MRI Data." bioRxiv preprint,
v3 July 2025. DOI: 10.1101/2024.11.11.622925. PMC12324183 (preprint pilot).

**Code:** https://github.com/TIB-Lab/SBIDTI

**Lab:** Translational Imaging Biomarkers (TIB) Lab, Instituto de Neurociencias,
CSIC-UMH, Alicante, Spain. PI: Silvia De Santis. Lead author Maximilian Eggl holds
a La Caixa Junior Leader fellowship.

### What it does

NPE via normalizing flows for three model families, with a focus on pushing toward
theoretical minimum acquisition requirements:

1. **DTI:** 6 tensor parameters + S0. Minimum 7 acquisitions (from 69 full) — close to
   the theoretical information-theoretic limit.
2. **DKI:** Full diffusion + kurtosis tensor. Minimum 22 acquisitions (from 138 full).
3. **AxCaliber/CHARMED:** Two-compartment biophysical model (hindered + restricted) with
   Poisson radius distribution. Minimum 19 acquisitions (from 271 full).

### Technical details

- Uses `sbi` Python toolbox with normalizing flows.
- 300K synthetic training samples per model (much less than Nottingham's 2-6M).
- Raw diffusion signals as input (no summary statistics).
- Rician noise corruption at SNR 2, 5, 10, 20, 30.
- Minimum acquisition schemes selected via electrostatic repulsion for optimal angular
  coverage.
- Priors: uniform for DTI/AxCaliber, log-normal fitted to HCP subject for DKI.
- Forward simulation uses DIPY functions (not dmipy).

### Key results

- DTI robust down to SNR=2 (NLLS degrades below SNR=10).
- DKI minimum-set accuracy loss ~5.6% vs NLLS 65%.
- AxCaliber angular error: 0.065 rad (SBI) vs 0.65 rad (NLLS reduced set).
- SBI never produces physically unrealistic negative values (unlike NLLS for DKI).
- SSIM > 0.9 on minimum acquisitions in vivo; NLLS drops below 0.66.
- Networks trained on HCP generalize to MS cohorts with different b-values.
- Up to **90% reduction in acquisition requirements** with maintained accuracy.

### Key innovation

The central contribution is demonstrating that SBI can approach the theoretical
information-theoretic minimum for number of acquisitions needed. This has direct
clinical translation implications — faster scans = more patients, less motion artifact,
pediatric/clinical feasibility.

---

## 5. Overlap with dmipy-JAX

### What dmipy-JAX already has

| Capability | dmipy-JAX status | FORCE | Nottingham SBI | Alicante SBI |
|------------|-----------------|-------|----------------|--------------|
| Stick model | C1Stick | Yes | Yes (Ball-Sticks) | Yes (AxCaliber) |
| Zeppelin model | G2Zeppelin | Yes | No | Partial (hindered) |
| Ball model | G1Ball | Yes (GM+FW) | Yes | No |
| Full tensor | G2Tensor | No | No | Yes (DTI/DKI) |
| Restricted cylinders | C2Cylinder (Callaghan, Soderman) | No | No | Yes (AxCaliber) |
| Sphere models (SANDI) | SphereGPD, SphereCallaghan | No | No | No |
| Bingham distribution | BinghamNODDI (JAX, differentiable) | Yes (Cython) | No | No |
| Watson distribution | SD1Watson | No | No | No |
| IVIM | Yes | No | No | No |
| Multi-compartment composition | Modular compose_models() | Fixed template | Fixed | Fixed |
| Levenberg-Marquardt | OptimistixFitter | No (no fitting) | No | No |
| L-BFGS-B | VoxelFitter | No | No | No |
| MCMC (NUTS) | Blackjax | No | Comparison target | No |
| Variational inference | NumPyro SVI | No | No | No |
| SBI / NPE | SBITrainer (FlowJAX), MDN examples | No | Yes (sbi toolbox) | Yes (sbi toolbox) |
| Amortized neural inference | MDN for NODDI, AxCaliber, DTI | No | Yes | Yes |
| Monte Carlo simulation | DifferentiableWalker, mesh FEM | No | No | No |
| Differentiable tractography | Yes | No | Probabilistic | No |
| GPU acceleration | Native JAX JIT/vmap | FAISS-GPU | CPU only | CPU only |
| Differentiability | Full (JAX autodiff) | None | Partial (NN only) | Partial (NN only) |
| Cosine similarity matching | Tractography only | Core method | No | No |

### What FORCE does that dmipy-JAX does not (yet)

1. **Dictionary-based forward matching** — no equivalent FAISS/ANN pipeline.
2. **Unified multi-metric output** — simultaneous DTI+DKI+NODDI+peaks+segmentation
   from one operation.
3. **Single-shell NODDI** — extracting NODDI-like metrics from single-shell data via
   matching against a multi-parameter library.
4. **Shallow crossing detection (10-40 degrees)** — superior to all ODF-based methods
   at acute fiber crossings.
5. **Heuristic uncertainty/ambiguity maps** — IQR and FWHM from K-NN similarity profile.

---

## 6. What dmipy-JAX can contribute to the forward-modeling landscape

### 6.1 Richer compartment models for simulation libraries

FORCE is locked to stick+zeppelin+Bingham+ball. dmipy-JAX's modular `compose_models()`
enables arbitrary compartment combinations. Plugging in dmipy-JAX forward models would
let FORCE (or any dictionary/SBI approach) generate signals with:

- **Axon diameter sensitivity** via restricted cylinder models (Callaghan, Soderman).
  Enables diameter estimation that FORCE currently cannot do.
- **Soma compartments** via sphere models (SphereGPD for SANDI). Enables soma density
  estimation.
- **Perfusion** via IVIM. Enables perfusion fraction extraction.
- **Anisotropic dispersion** via the differentiable BinghamNODDI already implemented.
- **Full diffusion tensors** via G2Tensor for richer extra-axonal modeling.

### 6.2 SBI as a continuous replacement for discrete dictionary lookup

dmipy-JAX already has the SBI infrastructure:

- `SBITrainer` with FlowJAX masked autoregressive normalizing flows
  (`dmipy_jax/inference/trainer.py`)
- Model-specific NPE pipelines: NODDI (`train_noddi.py`), AxCaliber (`train_axcaliber.py`),
  DTI (`train_dti.py`), SANDI (`run_wand_sandi_sbi.py`)
- Mixture Density Networks as a lighter alternative
- uGUIDE integration (`train_uguide.py`)

FORCE's simulation library is essentially training data for an NPE. Replacing the FAISS
lookup with a trained normalizing flow gives:

- **Continuous parameter estimates** instead of discrete library matches
- **Proper Bayesian posteriors** instead of heuristic IQR/FWHM uncertainty
- **Better angular resolution** (not bounded by ~4.1 degree grid)
- **Smaller memory footprint** (trained network vs 500K-entry library)

This directly bridges FORCE (Indiana) and the Nottingham/Alicante SBI work, with
dmipy-JAX as the differentiable backbone.

### 6.3 Differentiable refinement (hybrid FORCE + gradient optimization)

The most powerful combination:

1. FORCE-style cosine-similarity match → robust coarse initialization (no local minima)
2. dmipy-JAX Levenberg-Marquardt or L-BFGS-B refinement → precise continuous estimate

This is impossible with FORCE alone (Cython forward model, no gradients) but natural
with dmipy-JAX. Benefits:

- Combines FORCE's robustness to initialization with dmipy-JAX's precision
- Eliminates discretization artifacts
- Enables joint estimation of parameters FORCE doesn't model (diameter, soma density)

### 6.4 GPU-accelerated simulation library generation

FORCE generates its library with Cython+OpenMP. dmipy-JAX can generate the same signals
on GPU with `jax.vmap` over the forward model — potentially orders of magnitude faster,
and differentiable for acquisition optimization.

### 6.5 Monte Carlo ground truth for validation

FORCE validates against its own analytical forward model (circular). dmipy-JAX's
DifferentiableWalker and mesh-based FEM simulation provide independent particle-level
ground truth, giving any of these methods a much stronger validation story.

### 6.6 Acquisition optimization

dmipy-JAX's end-to-end differentiability enables gradient-based optimization of
acquisition protocols. For FORCE-style approaches: which b-values, gradient directions,
and pulse timings maximize the discriminability of the simulation library? For SBI
approaches: which acquisitions maximize expected posterior information gain?

The Alicante group's finding that SBI can approach theoretical minimum acquisitions
makes this especially relevant — dmipy-JAX could compute those limits analytically
via Fisher information, then verify them with SBI.

---

## 7. Comparative landscape summary

| | FORCE | Nottingham SBI | Alicante SBI | cuDIMOT | **dmipy-JAX** |
|---|---|---|---|---|---|
| **Paradigm** | Dictionary match | Neural posterior | Neural posterior | Classical fitting | Modular fitting + SBI |
| **Inference** | FAISS cosine sim | Neural Spline Flow | Normalizing flow | CUDA MCMC/LM | LM, MCMC, VI, NPE, MDN |
| **Forward model** | Fixed (stick+zep+Bingham) | Fixed (Ball-Sticks) | Fixed (DTI/DKI/AxCal) | Fixed (NODDI-Bingham) | **Any composable** |
| **Differentiable** | No | NN only | NN only | No | **End-to-end** |
| **GPU** | FAISS-GPU | CPU | CPU | Full CUDA | **Native JAX** |
| **Uncertainty** | Heuristic (KNN) | Bayesian posterior | Bayesian posterior | MCMC posterior | **Bayesian posterior** |
| **Diameter estimation** | No | No | Yes (AxCaliber) | No | **Yes (restricted cyl)** |
| **Bingham dispersion** | Yes (Cython) | No | No | Yes (CUDA) | **Yes (JAX)** |
| **Monte Carlo sim** | No | No | No | No | **Yes** |
| **Acquisition opt** | No | No | Empirical min | No | **Gradient-based** |
| **Clinical readiness** | High (one-shot) | Medium | Medium | High | Medium |

---

## 8. Recommended next steps for dmipy-JAX

### High-impact, buildable now

1. **FORCE-style dictionary matching module** — implement cosine-similarity matching
   against dmipy-JAX-generated libraries, using JAX for GPU-accelerated search.
   Demonstrate with richer compartments (restricted cylinders, SANDI) that FORCE
   cannot currently support.

2. **SBI benchmark paper** — compare dmipy-JAX's FlowJAX NPE against FORCE dictionary
   matching, Nottingham SBI, and Alicante SBI on identical forward models and data.
   dmipy-JAX is the only framework that can run all paradigms (dictionary, NPE, MDN,
   classical fitting, MCMC) on the same models.

3. **Hybrid initialization** — implement FORCE-style matching as an initialization
   strategy for Optimistix LM fitting. Measure convergence improvement and reduction
   in local minima vs random/DTI initialization.

### Medium-term

4. **Acquisition-optimal SBI** — use Fisher information (computable via JAX autodiff)
   to design minimal acquisition schemes, then train NPE on those. Compare to
   Alicante's empirical electrostatic approach.

5. **Bingham-NODDI SBI** — train NPE on the existing BinghamNODDI forward model.
   Neither FORCE nor either SBI paper offers Bingham-based SBI.

6. **Monte Carlo validation suite** — use DifferentiableWalker to generate ground truth
   for all methods. Publish as a community benchmark.

---

## References

1. Shah AJ et al. FORCE: FORward modeling for Complex microstructure Estimation.
   Research Square preprint, 2025. DOI: 10.21203/rs.3.rs-8151109/v1

2. Manzano-Patron JP et al. Uncertainty mapping and probabilistic tractography using
   Simulation-Based Inference in diffusion MRI. Medical Image Analysis 103:103580, 2025.
   DOI: 10.1016/j.media.2025.103580

3. Eggl MF, De Santis S. Simulation-Based Inference at the Theoretical Limit: Fast,
   Accurate Microstructural MRI with Minimal diffusion MRI Data. bioRxiv preprint v3,
   July 2025. DOI: 10.1101/2024.11.11.622925

4. Hernandez-Fernandez M et al. Using GPUs to accelerate computational diffusion MRI.
   NeuroImage 188:598-615, 2019.
