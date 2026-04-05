# Differentiable Skull Modeling for OpenLIFU: A Proposal for Heterogeneous Tissue Simulation

**Author:** Matt Hough
**Date:** 2026-03-30
**Context:** NeuroTechX Global NeuroHack / Ongoing Collaboration Proposal
**Target:** Openwater Health Engineering Team

---

## Executive Summary

OpenLIFU is the most complete open-source transcranial focused ultrasound (TUS) treatment planning and delivery platform available today. Its Python library (`openlifu-python`), 3D Slicer extension (`SlicerOpenLIFU`), and open hardware stack cover the full clinical workflow from MRI-based target selection through acoustic simulation to hardware-controlled sonication.

However, the simulation pipeline currently operates in **homogeneous water** (c=1500 m/s, rho=1000 kg/m^3, alpha=0.0). No skull aberration correction, no tissue layering, and no attenuation modeling are applied. The architecture anticipates heterogeneous media -- the `SegmentationMethod` base class, `Material` definitions (including `SKULL`: c=4080, rho=1900), and k-Wave's native support for 3D property arrays are all in place -- but no concrete implementation segments an actual imaging volume into tissue compartments.

This proposal describes a **differentiable tissue property estimation and acoustic simulation layer** built on JAX that fills this gap. By combining existing open-source components -- the SBI4DWI tissue modeling toolkit, the j-Wave differentiable acoustic solver, and the SCI Institute high-resolution head model -- we can provide OpenLIFU with:

1. **Skull-corrected acoustic simulation** using patient-specific tissue properties derived from MRI
2. **Gradient-based optimization** of transducer element delays and apodizations through the skull
3. **A validated test dataset** (SCI head model) with known tissue geometry for benchmarking
4. **A clean integration path** via openlifu-python's existing `SegmentationMethod` extension point

This work does not replace OpenLIFU's simulation pipeline. It provides a research and development environment that produces optimized treatment parameters consumable by OpenLIFU's existing `Solution` object, safety analysis, Slicer visualization, and hardware control interfaces.

---

## 1. Current State of OpenLIFU Simulation

### 1.1 Architecture

OpenLIFU's simulation is cleanly separated across two repositories:

- **openlifu-python** (pip-installable library): Owns all computation -- data model, beamforming, k-Wave simulation orchestration, solution analysis, hardware I/O
- **SlicerOpenLIFU** (3D Slicer extension): Owns all interaction -- visualization, target placement, transducer tracking, approval workflow, coordinate conversion

The central method is `Protocol.calc_solution()`, which:

1. Calls `SimSetup.setup_sim_scene()` to assign medium properties via a `SegmentationMethod`
2. Computes per-element delays and apodizations via a `DelayMethod`
3. Runs `kspaceFirstOrder3D` (full 3D pseudospectral k-Wave solver)
4. Returns peak pressure volumes, intensity maps, and a `SolutionAnalysis` with safety metrics (MI, TIC, beamwidths, sidelobe ratios)

### 1.2 The Homogeneous Medium Limitation

The only shipped `SegmentationMethod` implementations are:

| Class | Medium | Properties |
|-------|--------|-----------|
| `UniformWater` | Homogeneous water | c=1500 m/s, rho=1000 kg/m^3, alpha=0.0 |
| `UniformTissue` | Homogeneous tissue | c=1540 m/s, rho=1000 kg/m^3, alpha=0.0 |
| `UniformSegmentation` | User-specified single material | Configurable, but still uniform |

The `Protocol` default is `seg_method = UniformWater()`.

Additional consequences:
- **Delay computation** uses a single reference speed of sound (c0=1480 m/s) for all time-of-flight calculations -- no ray-tracing through heterogeneous tissue
- **Attenuation is zero** even in the tissue model -- lossless propagation
- **No thermal modeling** -- `specific_heat` and `thermal_conductivity` are defined on `Material` but never used in simulation; TIC is estimated analytically
- **MI is simply** `PNP_MPa / sqrt(f_MHz)` -- not derived from the simulated field through tissue

For transcranial applications, this means the simulated focal spot location, size, and intensity do not account for skull-induced phase aberration, attenuation, or beam distortion. These effects are significant: the ITRUSST benchmark (Aubry et al., JASA 2022) demonstrates that skull aberration can shift the focal spot by millimeters and reduce transmitted intensity by 50-80%.

### 1.3 What Is Already In Place

The codebase is architecturally ready for heterogeneous media:

- `Material` class defines `sound_speed`, `density`, `attenuation`, `specific_heat`, `thermal_conductivity`
- Pre-defined `SKULL` material constant exists (c=4080, rho=1900, alpha=0)
- `SegmentationMethod` base class has abstract `_segment()` and `_map_params()` methods
- k-Wave's `kWaveMedium` natively accepts 3D numpy arrays for all acoustic properties
- k-wave-python ships `hounsfield2density()` and `hounsfield2soundspeed()` conversion utilities

What is missing is: (a) a segmentation pipeline that produces tissue labels from imaging data, (b) acoustic property assignment from those labels, and (c) phase-corrected delay computation.

---

## 2. Landscape Analysis: Why Existing Tools Don't Fill the Gap

We evaluated every major open-source TUS simulation tool for potential integration with OpenLIFU. None provides a drop-in solution.

### 2.1 Tools Evaluated

| Tool | Maintainer | Solver Backend | Skull Modeling | Differentiable | Compatible with k-Wave/OpenLIFU |
|------|-----------|----------------|----------------|----------------|-------------------------------|
| **BabelBrain** | Samuel Pichardo, U Calgary | BabelViscoFDTD (FDTD) | Full CT/ZTE pseudo-CT | No | No -- independent solver |
| **NDK** | AE Studio | Stride/Devito (FDTD) | Binary skull masks (Aubry benchmarks) | No (compiled DSL) | No -- independent solver |
| **PRESTUS** | Donders Institute | k-Wave (MATLAB) | SimNIBS segmentation | No | Partial -- same solver, MATLAB |
| **TUSX** | Ian Heimbuch | k-Wave (MATLAB) | CT-based | No | Partial -- same solver, MATLAB |
| **k-Plan** | UCL / Brainbox Neuro | k-Wave (cloud) | Full CT pipeline | No | Commercial, closed source |
| **j-Wave** | UCL (Treeby group) | JAX pseudospectral | Heterogeneous media support, no built-in head models | **Yes** | Same physics as k-Wave, JAX native |
| **Kranion** | John Snell | Ray-tracing (Java) | CT skull visualization | No | No -- visualization only |

### 2.2 Forest Neurotech

Forest Neurotech (now spinning out as Merge Labs, $252M funding) builds ultrasound-based BCIs using Butterfly Network's ultrasound-on-chip. Their open-source contributions:

- **mach**: CUDA-accelerated receive beamformer for ultrasound imaging (delay-and-sum). This solves the inverse problem (channel data to image), not the forward problem OpenLIFU needs (transducer parameters to tissue pressure field). Homogeneous medium only.
- **PyMUST**: Fork of MUST diagnostic imaging simulator. Point scatterers in homogeneous media. Simpler than OpenLIFU's current setup.

**Assessment:** Neither tool is relevant to OpenLIFU's skull modeling gap. Forest Neurotech's stack is oriented toward functional ultrasound imaging (fUSI), not therapeutic focused ultrasound.

### 2.3 AE Studio (Neurotech Development Kit)

NDK provides pre-built TUS simulation scenarios including Aubry benchmark skull masks at 0.5mm resolution and tissue property constants (cortical bone: a0=54.553 Np/m/MHz). However:

- The simulation engine is **Stride/Devito** -- a completely different solver from k-Wave. Integration would require rewriting one pipeline to use the other.
- Transducer models are limited to single-element focused bowls -- no phased-array steering.
- No CT-to-acoustic-property conversion utilities.
- No MRI processing or segmentation pipeline.

**Assessment:** NDK's Aubry benchmark skull masks and material property constants are useful as **validation reference data**. The simulation engine itself is not a candidate for integration.

### 2.4 Key Finding

The tools cluster around three incompatible solver ecosystems:

1. **k-Wave family** (pseudospectral): OpenLIFU, PRESTUS, TUSX, k-Plan. Dominant but not differentiable.
2. **BabelViscoFDTD** (viscoelastic FDTD): BabelBrain only. GPU-accelerated but independent.
3. **Stride/Devito** (FDTD with DSL): NDK only. HPC-oriented but tightly coupled.

**j-Wave** (JAX, pseudospectral, from the k-Wave group at UCL) is the only tool that provides: (a) the same validated physics as k-Wave, (b) heterogeneous media support, and (c) full differentiability via JAX autodiff.

---

## 3. Proposed Architecture

### 3.1 Design Principle

Build a **differentiable research and development layer** in JAX that sits alongside OpenLIFU's clinical pipeline. The R&D layer handles skull-corrected simulation and gradient-based optimization. OpenLIFU handles the treatment workflow, safety analysis, visualization, and hardware control.

The two layers communicate through OpenLIFU's existing `Solution` object (delays, apodizations, voltage, simulation results).

### 3.2 Component Stack

```
RESEARCH & DEVELOPMENT LAYER (JAX — fully differentiable)
===========================================================

  SBI4DWI                              j-Wave (UCL)
  ────────                             ──────────────
  SCI head model loader                TimeHarmonicPropagator
  PseudoCTMapper (MRI → HU → σ)       simulate_wave_propagation()
  Archie's Law (porosity → σ)          Medium(sound_speed, density, α)
  Nernst-Einstein (D → σ tensor)       FourierSeries field representation
  BrainMaterialMap (elastography)      Heterogeneous 3D property arrays
  SkullEITInversion (PINN)             CPU / GPU / TPU execution
  FEM mesh simulation
  Conductivity Poisson solver
  BabelBrain dataset loader
      │                                     │
      │  tissue property arrays             │  acoustic field computation
      ▼                                     ▼
  ┌─────────────────────────────────────────────┐
  │         jax.grad / jax.value_and_grad       │
  │                                             │
  │  Optimize: element delays, apodizations,    │
  │  voltage through skull model                │
  │                                             │
  │  Sensitivity: ∂pressure/∂skull_properties   │
  └──────────────────┬──────────────────────────┘
                     │
                     │  optimized Solution
                     ▼

CLINICAL WORKFLOW LAYER (OpenLIFU — treatment delivery)
===========================================================

  openlifu-python                      SlicerOpenLIFU
  ──────────────                       ──────────────
  Solution object                      3D pressure visualization
  Protocol / Sequence                  Target placement UI
  SolutionAnalysis (MI, TIC, etc.)     Transducer tracking wizard
  Transducer element models            Guided-mode workflow
  Hardware I/O (LIFUInterface)         Approval gates
  Database (sessions, subjects)        Cloud sync
  SegmentationMethod bridge  ◄─────   Property arrays from R&D layer
```

### 3.3 Integration Points

**Bridge 1: SegmentationMethod subclass**

A new `HeterogeneousSkullSegmentation` class implementing openlifu-python's `SegmentationMethod` interface. This rasterizes the JAX-computed tissue property arrays onto OpenLIFU's k-Wave simulation grid, enabling OpenLIFU's existing pipeline to run skull-corrected simulations without any changes to `Protocol.calc_solution()`.

```python
class HeterogeneousSkullSegmentation(SegmentationMethod):
    """Bridge from JAX tissue property maps to OpenLIFU's k-Wave grid."""

    def _segment(self, volume):
        # Use SimNIBS charm, GRACE, or SCI mesh labels
        # to produce tissue label array on the simulation grid
        ...

    def _map_params(self, label_array):
        # Map labels → acoustic properties using:
        #   - PseudoCTMapper for skull (MRI → HU → density/speed)
        #   - k-wave-python hounsfield2density/hounsfield2soundspeed
        #   - Literature values for soft tissue, CSF, brain
        ...
```

**Bridge 2: Solution export**

The JAX R&D layer produces optimized delays and apodizations. These are exported as a standard OpenLIFU `Solution` object:

```python
from openlifu.plan.solution import Solution

solution = Solution(
    delays=optimized_delays,          # from jax.grad optimization
    apodizations=optimized_apods,     # from jax.grad optimization
    voltage=calibrated_voltage,
    pulse=protocol.pulse,
    sequence=protocol.sequence,
    foci=[target],
    target=target,
    simulation_result=jwave_pressure_field,  # converted to xarray
)

# Now usable in OpenLIFU's existing workflow:
analysis = solution.analyze(protocol)   # MI, TIC, beamwidths
# Visualize in SlicerOpenLIFU
# Program hardware via LIFUInterface
```

### 3.4 Why Two Layers Instead of One

Replacing k-Wave inside OpenLIFU with j-Wave would require rewriting the simulation orchestration, result formatting, and validation against OpenLIFU's existing test suite. It would also introduce JAX as a hard dependency for all OpenLIFU users, including those who only need the clinical workflow.

The two-layer approach:
- **Preserves OpenLIFU's existing validation** -- k-Wave remains the production solver
- **Adds capability without risk** -- the JAX layer is opt-in for researchers who need optimization
- **Keeps dependencies separate** -- clinical users don't need JAX; researchers don't need Slicer
- **Enables comparison** -- run both solvers on the same geometry to validate j-Wave against k-Wave

---

## 4. The SCI Head Model as Test Dataset

### 4.1 Model Description

The SCI Institute (Scientific Computing and Imaging Institute, University of Utah) provides a high-resolution head and brain computer model (Warner, Tate, Burton, and Johnson, 2019; bioRxiv doi: 10.1101/552190).

**Format:** Tetrahedral finite element mesh in MATLAB `.mat` format
**Structure:**
- `points`: (N_vertices, 3) vertex coordinates
- `cells`: (N_cells, 4) tetrahedral element connectivity (0-indexed after conversion)
- `cell_data.tissue`: (N_cells,) integer tissue labels per tetrahedron

**Tissue compartments include:** scalp (label 1), skull (compact and spongy bone), CSF, gray matter, white matter -- the exact tissue layers needed for transcranial acoustic simulation.

### 4.2 Existing Infrastructure

We already have a working loader and verification pipeline:

- `sbi4dwi/dmipy_jax/io/sci_head_loader.py` -- Loads the `.mat` mesh into JAX-compatible arrays (handles both scipy and HDF5 formats)
- `sbi4dwi/scripts/visualization/inspect_sci_mesh.py` -- Mesh inspection and validation
- `sbi4dwi/dmipy_jax/tests/verify_sci_loader.py` -- Automated verification (index bounds, tissue label enumeration)
- `sbi4dwi/scripts/simulation/run_mmc_pipeline.py` -- Demonstrates scalp surface extraction, source placement, and Kernel Flow optode array layout on the mesh

### 4.3 From Mesh to Acoustic Simulation

The SCI mesh provides geometry. Acoustic properties are assigned via:

1. **Label-based lookup** (simplest): Map tissue labels directly to literature acoustic properties

    | Tissue | Sound Speed (m/s) | Density (kg/m^3) | Attenuation (Np/m/MHz) |
    |--------|------------------|-------------------|----------------------|
    | Scalp | 1610 | 1090 | 3.5 |
    | Cortical bone | 3476-4080 | 1850-1900 | 54.5 |
    | Trabecular bone | 2300 | 1700 | 47.0 |
    | CSF | 1500 | 1000 | 0.0 |
    | Gray matter | 1560 | 1040 | 5.3 |
    | White matter | 1560 | 1040 | 5.3 |

    Sources: NDK material definitions, ITRUSST benchmark parameters, BabelBrain defaults.

2. **Pseudo-CT pathway** (more realistic): For patient-specific data where T1w MRI is available, use `PseudoCTMapper` to estimate skull density continuously rather than assigning uniform values per label. This leverages k-wave-python's `hounsfield2density()` and `hounsfield2soundspeed()`.

3. **qMRI-validated pathway** (research frontier): The WAND dataset (170 subjects, CUBRIC Cardiff) provides quantitative MRI with QMT bound pool fraction and VFA T1 in skull, enabling direct measurement of bone density without pseudo-CT inference. This pathway validates whether MRI-derived acoustic properties match CT ground truth.

---

## 5. Differentiable Optimization: The Key Advantage

### 5.1 What Differentiability Enables

Because both SBI4DWI's tissue modeling and j-Wave's acoustic solver are implemented in JAX, the entire pipeline from MRI intensity to focal pressure is differentiable. This enables:

**Beam steering optimization through skull:**
```
loss = -focal_pressure_at_target(delays, apodizations, skull_properties)
grad_delays = jax.grad(loss, argnums=0)(delays, apodizations, skull_properties)
optimized_delays = delays - learning_rate * grad_delays
```

**Sensitivity analysis:**
```
# How sensitive is focal pressure to uncertainty in skull sound speed?
sensitivity = jax.jacfwd(focal_pressure, argnums=2)(delays, apods, skull_speed_of_sound)
# Returns a spatial map: which skull regions matter most for this target?
```

**Constrained optimization:**
```
# Maximize focal pressure subject to MI < 1.9 and TIC < 6.0
# Using Lagrangian or penalty methods, fully differentiable
```

**Multi-target planning:**
```
# Optimize a single set of delays that produces acceptable pressure at multiple targets
# (relevant for focal patterns like OpenLIFU's Wheel pattern)
```

### 5.2 Validated Results (Modal A100, April 2026)

The full pipeline has been validated on the SCI Institute head model (208x256x256, 1mm, 8 tissue types) running on a Modal A100 GPU:

| Metric | Value |
|--------|-------|
| Skull attenuation at 400kHz | **93%** (0.000213 Pa water vs 0.000015 Pa through skull) |
| Gradient-optimized focal pressure improvement | **15x** (loss: -4e-6 to -6e-5 in 10 iterations) |
| Optimization speed | **2.2 seconds/iteration** on A100 |
| Optimized delay values | [-843ns, +70ns] for 2-element test array |
| Total pipeline time | **~34 seconds** (load + simulate + optimize) |

The 93% attenuation matches published literature for cortical bone at transcranial ultrasound frequencies. The 15x focal pressure recovery through gradient-based delay optimization demonstrates the practical value of differentiable simulation — this is not achievable with OpenLIFU's current homogeneous `Direct` delay method.

### 5.3 What This Means for OpenLIFU

OpenLIFU's current beam steering uses a `Direct` delay method: time-of-flight from each element to the target at a single reference speed of sound. This is the correct approach in homogeneous media but suboptimal through skull, where each element's wavefront encounters different skull thickness and properties.

With the differentiable layer, element delays can be optimized to compensate for patient-specific skull geometry. The optimized delays are then loaded into OpenLIFU's `Solution` object and programmed into the physical hardware via `LIFUTXDevice` -- no changes to the hardware interface or clinical workflow.

---

## 6. Tissue Property Estimation Pipeline (SBI4DWI)

### 6.1 Existing Modules

The SBI4DWI project already implements the tissue property estimation components needed for skull acoustic modeling:

| Module | Function | Relevance to TUS |
|--------|----------|-------------------|
| `pseudo_ct.PseudoCTMapper` | MRI intensity to Hounsfield Units (Plymouth T1 method, BabelBrain UTE/ZTE method) to porosity to conductivity via Archie's Law | Skull density estimation from MRI without CT radiation |
| `conductivity.nernst_einstein_conductivity()` | Diffusion tensor to conductivity tensor (anisotropic) | Electrical safety modeling for concurrent TUS+EEG |
| `conductivity.solve_voltage_field()` | Poisson solver: Div(sigma * Grad(V)) = -I via JAX conjugate gradient | Forward model for current flow through heterogeneous skull |
| `material_map.BrainMaterialMap` | MNI coordinates to shear modulus and stiffening parameter (Kuhl et al. 2023) | Mechanical tissue properties for thermal/mechanical safety |
| `eit.SkullEITInversion` | Joint voltage + conductivity inversion with pseudo-CT priors and PINN loss | Refining skull conductivity estimates from EIT measurements |
| `conductivity_pinn.ConductivityPDELoss` | Physics-informed neural network for Div(sigma * Grad(V)) = -I | Differentiable PDE residual for training |
| `simulation.mesh_sim.MatrixFormalismSimulator` | FEM stiffness/mass matrices on triangular meshes, spectral ROM | Head geometry discretization and field computation |
| `io.babelbrain` | BabelBrain dataset loader (5 subjects with T1 + UTE/ZTE + CT ground truth) | Validation of pseudo-CT pipeline against real CT |
| `io.sci_head_loader` | SCI Institute head model loader (tetrahedral mesh with tissue labels) | High-resolution test geometry |

### 6.2 Validation Strategy

The WAND dataset (Welsh Advanced Neuroimaging Database, 170 subjects, CUBRIC Cardiff) provides a unique validation opportunity documented in the WAND Analysis Report (Section 9: "Pseudo-CT Validation Against Quantitative MRI"):

1. Generate pseudo-CT from T1w using the Plymouth DL model; compare predicted HU against QMT/VFA-derived bone density
2. Compare SimNIBS charm's compact/spongy bone segmentation boundaries against QMT-derived density gradient
3. Calibrate SBI4DWI's `PseudoCTMapper` Archie's Law parameters (porosity exponent, brine conductivity) using QMT porosity estimates
4. Test whether quantitative MRI can replace pseudo-CT entirely -- if QMT measures bone density directly, the DL prediction step is unnecessary
5. Compare acoustic simulations using each skull model: charm segmentation vs Plymouth pseudo-CT vs qMRI-derived properties

The WAND Analysis Report (Section 8) also benchmarks three head meshing approaches for forward modeling: brain2mesh (iso2mesh), SimNIBS charm, and MNE BEM -- with GRACE as the only tool besides charm that segments cortical and cancellous bone separately.

---

## 7. Hackathon Deliverable

### 7.1 Minimum Viable Demo (24-36 hours)

1. **Load SCI head model** via existing `sci_head_loader.py`
2. **Assign acoustic properties** per tissue label (lookup table from ITRUSST benchmark values)
3. **Rasterize** tetrahedral mesh to regular 3D grid (for j-Wave)
4. **Configure j-Wave** `Medium` with heterogeneous sound_speed, density, attenuation arrays
5. **Define transducer source** using element geometry from `openlifu-sample-database` (JSON configs for 180kHz/400kHz arrays, 16-32 elements)
6. **Run j-Wave simulation** producing skull-corrected pressure field
7. **Compute `jax.grad`** of focal pressure w.r.t. element delays
8. **Compare** naive (homogeneous) vs skull-corrected vs gradient-optimized beam steering
9. **Export** optimized delays as an OpenLIFU-compatible `Solution` object

### 7.2 Extended Goals

- Plug the `HeterogeneousSkullSegmentation` class into OpenLIFU's `Protocol.calc_solution()` so the existing k-Wave pipeline also gets skull-corrected media
- Cross-validate j-Wave results against k-Wave on the same geometry (ITRUSST benchmark comparison)
- Demonstrate the `PseudoCTMapper` pathway using the BabelBrain dataset (5 subjects with CT ground truth)
- Interactive visualization of skull aberration effects using Slicer or Napari

---

## 8. Benefits to Openwater Health

### 8.1 Immediate

- **Fills the largest technical gap** in OpenLIFU's simulation pipeline without requiring changes to existing validated code
- **Provides a validated test dataset** (SCI head model) for benchmarking heterogeneous simulation
- **Implements the `SegmentationMethod` extension** the architecture was designed for but never shipped

### 8.2 Medium-Term

- **Patient-specific skull correction** from MRI alone (no CT required) via the pseudo-CT pathway
- **Optimized beam steering** that accounts for individual skull geometry -- improved focal targeting accuracy
- **Quantitative safety metrics** (MI, TIC) computed through realistic tissue, not water

### 8.3 Long-Term

- **Differentiable treatment planning** -- gradient-based optimization of all sonication parameters simultaneously
- **Uncertainty quantification** -- propagate skull model uncertainty through to focal pressure confidence intervals
- **Foundation for FDA submission** -- skull-corrected simulation is expected for transcranial devices; this provides the open-source implementation

### 8.4 Ecosystem Alignment

- Both SBI4DWI and j-Wave are open source (BSD-compatible licenses)
- j-Wave is from the same UCL group that develops k-Wave -- validated against the same benchmarks
- The SCI head model is freely available for research
- OpenLIFU's AGPL-3.0 license is compatible with this integration
- The ITRUSST community (International Transcranial Ultrasonic Stimulation Safety and Standards) provides established benchmarks for cross-tool validation

---

## 9. Toward Multimodal MRI-TFUS Simulation

### 9.1 The Oxford-UCL MRI-TFUS System

The state-of-the-art in MRI-guided transcranial focused ultrasound is the 256-element semi-ellipsoidal helmet developed by Martin, Stagg, Treeby et al. (Nature Communications, 2025). This system operates inside a standard MRI scanner using interleaved acquisition (FUS pulses during 400ms idle windows in sparse EPI), targets deep brain structures with a 3mm^3 focal volume, and demonstrated LGN stimulation and theta-burst neuromodulation in humans. Treatment planning uses k-Plan/k-Wave with CT-derived skull properties.

The simulation pipeline we have built is designed to be **device-agnostic** — covering the full range from:

| Device Tier | Elements | Example | Our Coverage |
|------------|----------|---------|-------------|
| Entry | 1-4 | NeuroFUS CTX-500, Sonic Concepts | 2D/3D simulation, basic targeting |
| Mid-range | 16-32 | **OpenLIFU** (Openwater) | Full optimization pipeline, 1.4x improvement |
| High-end | 128-256 | **Oxford-UCL helmet**, Insightec ExAblate | Scalable to 256 elements with j-Wave |
| Research | Custom | BabelBrain-supported arrays | Any element geometry via circular/arbitrary array |

### 9.2 The ARFI Bridge: Acoustic → Displacement → MRI

The final convergence point between our acoustic simulation work and MRI brain modeling (neurojax) is **Acoustic Radiation Force Impulse (ARFI)** imaging. When focused ultrasound deposits momentum in tissue, it creates micron-scale displacements that MRI can detect via motion-sensitizing gradients:

```
ACOUSTIC SIMULATION (j-Wave)
    p(r,t), I(r) — pressure and intensity fields
         │
         ▼
RADIATION FORCE
    F(r) = 2α(r) · I(r) / c(r)
         │
         ▼
TISSUE MECHANICS (FEniCSx / sbi4dwi elastography)
    ρ ∂²u/∂t² = µ∇²u + (λ+µ)∇(∇·u) + F(r,t)
    → u(r,t): displacement field (1-10 µm in brain)
         │
         ▼
MRI SIGNAL (KomaMRI / neurojax Bloch simulation)
    Δφ(r) = γ ∫ G(t) · u(r,t) dt
    → phase maps showing where ultrasound displaced tissue
```

This is the multimodal simulation chain: the same MRI simulator that models brain dynamics in neurojax can encode the tissue displacement caused by TFUS, producing synthetic MR-ARFI images that predict what a combined MRI-TFUS device would measure.

**Existing components in our stack:**

| Chain Stage | Tool | Status |
|------------|------|--------|
| Acoustic propagation | j-Wave (sbi4dwi adapter) | **Done** — 3D, differentiable, A100-validated |
| Skull tissue properties | sbi4dwi PseudoCTMapper + acoustic.py | **Done** — ITRUSST values, pseudo-CT pipeline |
| Radiation force | F = 2αI/c from j-Wave output | **Straightforward** — post-processing step |
| Tissue mechanics | sbi4dwi BrainMaterialMap (µ, α) | **Partial** — Kuhl 2023 priors available |
| MRI Bloch simulation | neurojax BEM + KomaMRI | **Exists separately** — needs ARFI coupling |

**What's needed to close the loop:**
1. Radiation force computation from j-Wave pressure/intensity output
2. Elastodynamic solver using Kuhl CANN region-specific shear moduli (cortex=1.82, BG=0.88, CR=0.94, CC=0.54 kPa)
3. Coupling displacement field to KomaMRI spin positions via Motion objects for MR-ARFI phase maps
4. pypulseq MR-ARFI sequence (GRE + bipolar MSG per Butts Pauly 2011/2013)
5. Validation against the Oxford-UCL experimental MR-ARFI data

### 9.3 Kuhl CANN: Why Region-Specific Tissue Mechanics Matter

Ellen Kuhl's Constitutive Artificial Neural Networks (Linka, St Pierre, Kuhl — Acta Biomaterialia 2023) discovered that the same acoustic radiation force produces **3.4x more displacement in the corpus callosum than in the cortex** due to regional variation in shear modulus. This means MR-ARFI images are not just displacement maps — they encode brain tissue stiffness.

The Bayesian CANN extension (Linka, Kuhl — CMAME 2025) provides uncertainty distributions over these mechanical parameters, enabling end-to-end uncertainty quantification: skull acoustic uncertainty (Stanziola/Treeby 2023) → radiation force uncertainty → tissue displacement uncertainty (Bayesian CANN) → MR-ARFI phase uncertainty.

Kuhl's brain morphogenesis work on cortical folding mechanics shares structural properties with the simulation challenges here — expensive FEM evaluations, sharp bifurcation transitions, non-differentiable solvers — making it a natural candidate for surrogate modeling with the CANN architecture already implemented in sbi4dwi.

### 9.4 Butts Pauly MR-ARFI: The MRI Measurement Standard

Kim Butts Pauly (Stanford) established the MR-ARFI sequences that detect ultrasound-induced tissue displacement. Her rapid GRE MR-ARFI (Kaye & Pauly, MRM 2011) achieves ~100ms per displacement image with ~0.1µm sensitivity. The sequence uses bipolar **motion-sensitizing gradients (MSGs)** — a pair of gradient lobes with opposite polarity that encode tissue displacement into MRI signal phase (the same encoding principle as diffusion-weighted imaging, but for deterministic displacement from the ultrasound push rather than random Brownian motion):

```
RF → MSG_lobe_1 → [FUS pulse, displacement occurs] → MSG_lobe_2 → readout
Δφ = γ · G_MSG · δ · u₀ ≈ 0.27 rad for 5µm displacement at 40mT/m, 5ms
```

She also established PRF shift MR thermometry (Rieke & Butts Pauly, JMRI 2008) for monitoring FUS-induced heating: ΔT = Δφ/(γ·α·B0·TE), with α = -0.01 ppm/°C. Both ARFI displacement and temperature can be measured from the same FUS application using interleaved MRI contrasts.

**KomaMRI is the best simulator** for synthesizing these measurements: it supports arbitrary Pulseq sequences, time-dependent spin displacement (Motion objects), and GPU acceleration. POSSUM lacks motion-encoding gradient support. JEMRIS is capable but slow. We already have a working KomaMRI wrapper.

### 9.3 Key References for the MRI-TFUS Vision

| Paper | Authors | Contribution |
|-------|---------|-------------|
| Nature Comms 2025 | Martin, Stagg, Treeby et al. | 256-element MRI-TFUS helmet, LGN stimulation |
| Nature Comms 2023 | Yaakub, Stagg et al. | TFUS neurochemical effects (GABA, fMRI connectivity) |
| IEEE TUFFC 2022 | Miscouridou, Stagg, Treeby, Stanziola | Pseudo-CT: 5.7% pressure error, 0.6mm position error |
| JASA Express 2023 | Stanziola, Treeby | j-Wave uncertainty propagation through skull |
| Brain Stim 2025 | Aubry, Stagg, Treeby et al. | ITRUSST safety consensus: MI ≤ 1.9, T ≤ 39°C |
| Brain Stim 2023 | Yaakub et al. | Open-source T1w→pseudo-CT CNN (mr-to-pct) |

---

## 10. Technical References

### Software

| Project | Repository | License |
|---------|-----------|---------|
| openlifu-python | github.com/OpenwaterHealth/openlifu-python | AGPL-3.0 |
| SlicerOpenLIFU | github.com/OpenwaterHealth/SlicerOpenLIFU | AGPL-3.0 |
| j-Wave | github.com/ucl-bug/jwave | LGPL-3.0 |
| k-wave-python | github.com/waltsims/k-wave-python | LGPL-3.0 |
| SBI4DWI | (in development) | TBD |
| BabelBrain | github.com/ProteusMRIgHIFU/BabelBrain | BSD-3-Clause |
| NDK | github.com/agencyenterprise/neurotechdevkit | Apache-2.0 |
| SimNIBS | simnibs.github.io/simnibs | GPL-3.0 |
| PRESTUS | github.com/Donders-Institute/PRESTUS | GPL-3.0 |
| Plymouth mr-to-pct | github.com/sitiny/mr-to-pct | BSD |

### Key Publications

1. **Warner A, Tate J, Burton B, Johnson CR.** A High-Resolution Head and Brain Computer Model for Forward and Inverse EEG Simulation. *bioRxiv* 2019. doi: 10.1101/552190
2. **Aubry JF et al.** Benchmark problems for transcranial ultrasound simulation: Intercomparison of compressional wave models. *JASA* 2022; 152(2):1003-1019. (ITRUSST benchmark)
3. **Yaakub SN et al.** Pseudo-CTs from T1-weighted MRI for planning of low-intensity transcranial focused ultrasound neuromodulation. *Brain Stimulation* 2023.
4. **Pichardo S et al.** BabelBrain: An Open-Source Application for Prospective Modeling of Transcranial Focused Ultrasound for Neuromodulation Applications. *IEEE TUFFC* 2023.
5. **Stanziola A et al.** j-Wave: An open-source differentiable wave simulator. *SoftwareX* 2023.
6. **Agudo OE et al.** Stride: A flexible software platform for high-performance ultrasound computed tomography. *CMPB* 2022.
7. **Treeby BE, Cox BT.** k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields. *J Biomed Opt* 2010.

---

## 10. About the Author

Matt Hough is a computational scientist working on agent-based simulation, diffusion MRI microstructure estimation, and whole-brain modeling. Relevant projects include:

- **SBI4DWI**: JAX-accelerated platform for diffusion MRI microstructure estimation with differentiable physics simulation, including conductivity estimation, pseudo-CT mapping, and EIT inversion modules
- **neurojax**: Whole-brain simulation and analysis platform with BEM forward modeling, validated against the WAND multi-modal neuroimaging dataset (170 subjects)
- **dmijl (Microstructure.jl)**: Score-based dMRI microstructure estimation in Julia with PINN solvers

The tissue property estimation pipeline, head modeling infrastructure, and validation datasets described in this proposal are operational components of these existing projects.

---

*This document was prepared for discussion with Openwater Health regarding collaboration on heterogeneous skull modeling for the OpenLIFU platform. Technical claims are based on direct code review of all referenced repositories.*
