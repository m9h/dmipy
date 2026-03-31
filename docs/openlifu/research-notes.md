# OpenLIFU + Openwater Health Research Notes

**Date:** 2026-03-30
**Context:** NeuroTechX Global NeuroHack preparation, Openwater collaboration evaluation

---

## 1. Openwater Health GitHub Organization

**URL:** https://github.com/OpenwaterHealth
**Total Public Repos:** 59 (1 archived)
**License:** AGPL-3.0 (transitioning to Apache 2.0, Week 1 of 12 as of Jan 2026)
**Mission:** "The world's first open-source, open-patent medical device company." Reduce medical device development costs from $119M to $15M through open-source collaboration.

### Four Product Platforms

#### Platform 1: OpenLIFU (Low Intensity Focused Ultrasound) — FLAGSHIP

Most actively developed. Multiple repos with commits on 2026-03-30.

| Repo | Type | Stars | Updated | Description |
|------|------|-------|---------|-------------|
| openlifu-python | Core Python lib | 18 | 2026-03-26 | pip-installable. k-Wave simulation, beamforming, treatment planning, hardware I/O |
| SlicerOpenLIFU | 3D Slicer extension | 5 | 2026-03-12 | 10 modules: Home, Database, Login, Data, PrePlanning, TransducerLocalization, SonicationPlanner, SonicationControl, ProtocolConfig, CloudSync |
| openlifu-desktop-application | Standalone app | 2 | 2026-03-12 | Slicer-based desktop wrapper. Requires NVIDIA GPU. |
| openlifu-sdk | Hardware I/O | 0 | 2026-03-27 | Low-level USB/serial to TX module, HV controller |
| opw_neuromod_sw | Legacy MATLAB | 26 | 2026-01-06 | Original open-TFUS. Being superseded by openlifu-python |
| opw_neuromod_hw | Hardware | 22 | 2026-01-20 | Complete wearable ultrasound array designs |
| opw_ustx | HW+FW | 3 | 2025-11-07 | Standalone ultrasound transmit module (STM32) |
| openlifu-transmitter-fw | Firmware | 0 | 2026-03-30 | STM32L443 TX firmware (committed today) |
| openlifu-transmitter-bl | Firmware | 0 | 2026-03-30 | Secure bootloader: ECDSA P-256, HMAC-SHA256, USB/I2C DFU |
| openlifu-console-fw | Firmware | 0 | 2026-03-30 | Console power PCB firmware |
| openlifu-test-app | Test UI | 0 | 2026-03-30 | Python/QML test app |
| openlifu-sample-database | Data/Config | 0 | 2026-01-09 | 6 transducer types (180/400kHz, 16-32 elements), protocols, example subjects |
| OpenLIFU-3DScanner | Android+desktop | 1 | 2025-11-12 | Photogrammetric 3D mesh capture for transducer localization |
| OpenLIFU-Mechanical | CAD | 1 | 2025-07-19 | Transducer housings, console drawings |
| OpenLIFU-Electronics | PCB | 0 | 2025-03-07 | PCB designs |
| openlifu-verification-tank | QA | 1 | 2026-02-09 | Automated acoustic field verification (PicoScope, calibrated hydrophone) |

#### Platform 2: Blood Flow / OpenMOTION (Optical Stroke Diagnosis)

Near-infrared speckle-based cerebral blood flow monitoring for LVO stroke detection. FDA breakthrough designation.

| Repo | Type | Stars | Description |
|------|------|-------|-------------|
| opw_bloodflow_gen2_ai | ML/AI | 1 | ResNet-1D + Transformer-1D for LVO classification. Published clinical data. |
| opw_bloodflow_gen2_sw | Software | 4 | TI TDA4VM vision app + Qt UI + Python analysis |
| opw_bloodflow_gen2_hw | Hardware | 5 | STEP/SolidWorks + Altium PCB + gerbers |
| openmotion-sdk | Python SDK | 1 | OpenMOTION 3.0 sensor/console communication |
| openmotion-camera-fpga | FPGA | 1 | Histogram pipeline + I2C camera sensor (Verilog) |
| + 8 more | FW/HW/FPGA | 0 | Sensor, console, seed laser, TA, safety FPGAs |

#### Platform 3: Acousto-Optic Imaging

| Repo | Type | Stars | Description |
|------|------|-------|-------------|
| openwater-acousto-optic-software | Software | 5 | Jupyter scanner UI + Spyder analysis |
| openwater-acousto-optic-hardware | Hardware | 4 | Full BOM: MOGLabs laser, dual AOMs, Sonic Concepts transducer |
| openwater-acousto-optic-data | Data | 3 | 14 real preclinical datasets (rat brain, kidney, tumor, 2020) |

#### Platform 4: Oncolysis

| Repo | Type | Stars | Description |
|------|------|-------|-------------|
| opw_oncolysis_data | Data | 4 | In vitro viabilities, spheroid data, in vivo mouse tumor measurements |
| opw_oncolysis_sw | Software | 0 | Controller for Rigol AWG + NI IVI + Radiall switches (Windows) |
| opw_oncolysis_hw | Hardware | 0 | In vitro + in vivo setups, 3D printed holders |

#### Supporting Infrastructure

| Repo | Description |
|------|-------------|
| openwater-regulatory | Actual FDA communications published on GitHub. Unique transparency. |
| openwater-patents | 68 patent filings (26 granted, 18 pending) + Patent Pledge |
| openwater-commons | Governance, AGPL→Apache transition roadmap |
| openwater-docs | MkDocs site (docs.openwater.health) |
| openwater-community | GitHub Pages portal |
| awesome-openwater | Curated resource list. References Mayo + Stanford partnerships. |
| openwater-phantoms | Phantom recipes (optical, ultrasound, acousto-optic) |
| AutoFiducialContest | Spring 2025 coding contest: automatic fiducial detection |

---

## 2. SlicerOpenLIFU Deep Analysis

### Architecture

Built almost entirely by **Kitware** under contract to Openwater:
- 453 commits: arhowe00 (Kitware)
- 391 commits: sadhana-r (Kitware)
- 251 commits: ebrahimebrahim (Kitware)
- 2 commits: peterhollender (Openwater)
- 1 commit: lassoan (PerkLab/Slicer core)

### Module Details

**OpenLIFUHome**: Navigation hub, guided-mode step enforcement, install requirements button.

**OpenLIFUDatabase**: Local filesystem JSON database. Structure: protocols/, subjects/, transducers/, systems/, users/. Cloud sync via api.nvpsoftware.com.

**OpenLIFULogin**: Role-based auth (admin/operator), bcrypt passwords.

**OpenLIFUData** (~2800 lines, largest module): Subject/session/volume management. DICOM + NIfTI. Photoscans, photocollections, virtual fit results.

**OpenLIFUPrePlanning**: Target placement (fiducial markups on MRI). Virtual fit: automated transducer placement using `openlifu.virtual_fit`. Skin segmentation via `openlifu.seg.skinseg` (Otsu threshold, morphological closing). Virtual fit results ranked by steering distance.

**OpenLIFUTransducerLocalization** (~3900 lines, most complex): 4-page QWizard:
1. PhotoscanMarkupPage: 3 facial landmarks (R ear, L ear, nasion) on photoscan mesh
2. SkinSegmentationMarkupPage: Same 3 landmarks on MRI skin surface
3. PhotoscanVolumeTrackingPage: Fiducial registration → ICP refinement, manual transform, scale slider (0.8-1.2x)
4. TransducerPhotoscanTrackingPage: ICP of transducer registration surface to photoscan

Photos from Android app via ADB. Meshroom (AliceVision 2023.3.0) reconstruction. MODNet for background removal. QR code generation for session URI.

**OpenLIFUSonicationPlanner**: Calls `protocol.calc_solution()` → beamforming + k-Wave → PNP/intensity volumes. Solution analysis: safety metrics, parameter constraint checking. PNP visualization with MIP 3D, foreground overlay 2D.

**OpenLIFUSonicationControl**: Hardware interface to LIFUInterface/LIFUHVController/LIFUTXDevice. State machine: NOT_CONNECTED → CONNECTED → CONFIGURED → READY → RUNNING. Abort, run logging.

**OpenLIFUProtocolConfig**: Protocol editor. Role-based access.

**OpenLIFUCloudSync**: Background thread sync to Openwater cloud.

### External Tool References Found

- **k-Wave**: Sole acoustic simulation backend. Binaries auto-downloaded.
- **Meshroom (AliceVision)**: Photogrammetric reconstruction.
- **MODNet**: Background removal neural network.
- **ADB**: Android Debug Bridge for photo transfer.

### NOT Found

Zero references to: BabelBrain, Forest Neurotech, AE Studio, k-Plan, BrainSuite, FreeSurfer, SimNIBS, FOCUS, BioHeat, or any other TUS tool.

---

## 3. openlifu-python Simulation Pipeline Deep Analysis

### Protocol.calc_solution() Flow (protocol.py:242)

1. `self.check_target(target)` — bounds check
2. `params = sim_options.setup_sim_scene(self.seg_method, volume=volume)` — medium setup
3. For each focal point from FocalPattern:
   a. `self.beamform(arr, target, params)` → delays, apodization
   b. `run_simulation()` from kwave_if.py
4. Optional scaling to match target_pressure
5. Max-aggregate pressure, mean-aggregate intensity across foci
6. `SolutionAnalysis` — safety/efficacy metrics

### Critical Finding: Homogeneous Water Medium

Only shipped SegmentationMethod implementations:
- `UniformWater`: c=1500, rho=1000, alpha=0.0
- `UniformTissue`: c=1540, rho=1000, alpha=0.0
- `UniformSegmentation`: user-specified single material

Default: `seg_method = UniformWater()`.

**No skull segmentation. No tissue layering. No attenuation. No thermal modeling.**

`SKULL` material exists as a constant (c=4080, rho=1900) but is never used.
`SegmentationMethod._segment()` abstract method is ready for heterogeneous implementation.
`_map_params()` can map label arrays to material properties.
k-Wave `kWaveMedium` natively supports 3D property arrays.

### What k-Wave Actually Computes

Grid: 60x60x64 mm default, 1mm spacing.
Medium: homogeneous (scalar c, rho, alpha) or heterogeneous (3D arrays).
Source: Each element = rectangular piston via `kWaveArray.add_rect_element()`.
Signal: Pure sinusoidal toneburst, up to 20 cycles.
Solver: `kspaceFirstOrder3D` — full 3D pseudospectral time-domain.
Output: p_max (PPP), p_min (PNP), intensity = p_min^2 / (2Z).

Hardcoded: `alpha_power = 0.9`, `alpha_mode = 'no_dispersion'`.

### Delay Computation

`Direct.calc_delays()`: uses single scalar c (ref_value from params), not ray-tracing.
`delays = max(tof) - tof` where `tof = distance / c`.

### MI and TIC

Analytical approximations, not simulation-derived:
- MI = PNP_MPa / sqrt(f_MHz)
- TIC uses emitted power + equivalent aperture diameter + fixed cranial thermal conductivity

### Solution Class

```python
Solution(id, name, protocol_id, transducer, date_created, description,
         delays, apodizations, pulse, voltage, sequence, foci, target,
         simulation_result, approved)
```
- delays: (num_foci, num_elements)
- apodizations: (num_foci, num_elements)
- simulation_result: xa.Dataset with p_max, p_min, intensity

### Virtual Fit

Geometry-only computation (no acoustics):
1. Extract skin surface from MRI (Otsu threshold)
2. Build spherical interpolator from skin mesh
3. Search (pitch, yaw) grid around target
4. At each candidate: fit tangent plane, orient transducer normal, apply standoff, check steering limits
5. Return ranked transforms by steering distance

---

## 4. Open-Source TUS Simulation Landscape

### Three Solver Ecosystems

1. **k-Wave family** (pseudospectral): OpenLIFU, PRESTUS, TUSX, BRIC tools, k-Plan. Dominant.
2. **BabelViscoFDTD** (viscoelastic FDTD): BabelBrain only. Multi-GPU (CUDA/Metal/OpenCL).
3. **Stride/Devito** (FDTD with DSL): NDK only. HPC-oriented.

**j-Wave** (JAX): Same UCL group as k-Wave. Differentiable. Heterogeneous media. The bridge.

### BabelBrain (Samuel Pichardo, U Calgary)

- GitHub: ProteusMRIgHIFU/BabelBrain. BSD-3-Clause.
- Standalone Python GUI for transcranial FUS modeling.
- Backend: BabelViscoFDTD (FDTD). CUDA/Metal/OpenCL GPU backends.
- Full CT/ZTE pseudo-CT → skull aberration correction.
- Supports: CTX 250/500, DPX 500, H317, I12378, ATAC (128 elements), REMOPD (256), hemispherical domes (1024).
- Brainsight 2.5.3 integration.
- PlanTUS (2025) integration for transducer placement optimization.
- **No connection to Openwater or OpenLIFU.**

### Forest Neurotech / Merge Labs

- Nonprofit FRO via Convergent Research (Schmidt Futures).
- Founded by Sumner Norman (ex-CSO at AE Studio).
- Building ultrasound BCIs with Butterfly Network's ultrasound-on-chip.
- Merge Labs: $252M spinout (OpenAI, Bain Capital, Gabe Newell).
- Open-source: **mach** (CUDA receive beamformer, 22 stars), **fusi-bids-pydantic**, **PyMUST** fork.
- **mach is receive-side imaging beamforming** — irrelevant to therapeutic FUS.
- **No connection to Openwater.**

### AE Studio (Neurotech Development Kit)

- GitHub: agencyenterprise/neurotechdevkit. Apache-2.0. 134 stars.
- Python library for TUS simulation using Stride/Devito.
- Built-in scenarios with Aubry benchmark skull masks (BM7 at 0.5mm, BM8 at 0.5mm).
- Material constants: cortical bone a0=54.553 Np/m/MHz, trabecular bone a0=47 Np/m/MHz.
- Transducers: single-element focused bowls only (no phased arrays).
- **No CT-to-acoustic conversion. No MRI processing. Tightly coupled to Stride.**
- Key contributor: Sumner Norman (same person, AE Studio → Forest → Merge).
- **No connection to Openwater.**

### PRESTUS (Donders Institute)

- GitHub: Donders-Institute/PRESTUS. GPL-3.0. MATLAB.
- End-to-end: MRI segmentation (SimNIBS) → k-Wave acoustic simulation.
- HPC-designed but works locally. NIfTI output.

### Other Tools

| Tool | Language | Backend | Skull Modeling | Notes |
|------|----------|---------|----------------|-------|
| TUSX | MATLAB | k-Wave | CT-based | Accessible toolbox for TUS researchers |
| BRIC tools (Plymouth) | MATLAB | k-Wave | NeuroFUS PRO CTX-500 | Companion mr-to-pct pseudo-CT tool |
| Kranion | Java | Ray-tracing | CT visualization | HIFU surgery pre-planning |
| j-Wave (UCL) | Python/JAX | Pseudospectral | Heterogeneous medium support | Differentiable. No built-in head models. |
| k-Plan | Commercial | k-Wave (cloud) | Full CT pipeline | UCL/Brainbox Neuro. Not open source. |
| HITU Simulator (FDA) | MATLAB | Axisymmetric | Heating/thermal dose | Regulatory science tool |
| PlanTUS | Publication | Heuristic | Skull feasibility | Outputs to BabelBrain or k-Wave |

### ITRUSST Benchmark (Aubry et al., JASA 2022)

11 modeling tools compared across 18 benchmark configurations.
Median focal pressure difference <10%, position difference <1mm.
Provides confidence that different solver backends produce comparable results.

---

## 5. j-Wave API Summary

### Core Classes

```python
Domain(N=(nx, ny, nz), dx=(spacing_x, spacing_y, spacing_z))  # meters

Medium(domain, sound_speed, density, attenuation, pml_size=20)
# sound_speed/density/attenuation: scalar, Array, or FourierSeries

TimeAxis(dt, t_end)
TimeAxis.from_medium(medium, cfl=0.3, t_end=None)

Sources(positions=(x_list, y_list, z_list), signals=array, dt=dt, domain=domain)

Sensors(positions=(x_list, y_list, z_list))
```

### Solvers

```python
# Time-domain
simulate_wave_propagation(medium, time_axis, sources=..., sensors=..., p0=..., u0=...)

# Frequency-domain
helmholtz_solver(medium, omega, source, method="gmres", tol=1e-3, maxiter=1000)
born_series(medium, src, omega=..., max_iter=1000, tol=1e-8)
angular_spectrum(pressure, z_pos=..., f0=..., medium=...)
rayleigh_integral(pressure, r=..., f0=..., sound_speed=1500)
```

### Heterogeneous Medium Example

```python
sound_speed = np.ones(domain.N) * 1500.0
sound_speed[50:90, 32:100] = 2300.0  # skull region
sound_speed_field = FourierSeries(np.expand_dims(sound_speed, -1), domain)
medium = Medium(domain=domain, sound_speed=sound_speed_field, density=..., pml_size=20)
```

### Key Properties

- All operations JAX-traceable → differentiable via jax.grad
- Supports jit compilation
- FourierSeries wraps arrays for PSTD method
- Validated against k-Wave MATLAB in test suite
- Dependencies: jaxdf >= 0.3.0, Python 3.11+

---

## 6. sbi4dwi Tissue Property Modules

### PseudoCTMapper (pseudo_ct.py)

```python
PseudoCTMapper(method='plymouth'|'babel')
  .mri_to_hu(intensity)      # Plymouth: HU = 1700*(1-x)+300. Babel: HU = 2000*x+100
  .hu_to_porosity(hu)        # phi = max(1 - hu/2500, 0.05)
  .archies_law(phi, sigma_brine=2.0, m=1.5)  # sigma = sigma_brine * phi^m
  .predict_conductivity(mri) # End-to-end
```

### Conductivity Module (conductivity.py)

```python
nernst_einstein_conductivity(D, C, T=310.15, z=1)  # Diffusion tensor → conductivity tensor
create_electrode_masks(positions, shape, affine)      # Point → voxel mask
solve_voltage_field(sigma, source_map, voxel_size)    # JAX CG Poisson solver
tdcs_objective_function(currents, masks, roi, dir, sigma)  # Differentiable tDCS optimization
```

### SCI Head Loader (sci_head_loader.py)

```python
load_sci_head_mesh(file_path) → {
    'points': (N_vertices, 3) float32,
    'cells': {'tetra': (N_cells, 4) int32},  # 0-indexed
    'cell_data': {'tissue': (N_cells,) int32}  # 1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM
}
```

Supports scipy.io (v5/v6) and h5py (v7.3) .mat files.
Data at: `sbi4dwi/data/SCI_headmodel/extracted/HeadMesh.mat`

### BrainMaterialMap (material_map.py)

```python
BrainMaterialMap()
  .get_priors(mni_coords)  # (N,3) → (N,2) [shear_modulus_kPa, stiffening_param]
  # WM: mu=1.68, alpha=0.45. GM: mu=1.12, alpha=0.23. CSF: mu=0.01, alpha=0.0
```

### EIT Module (eit.py)

```python
EITModel(eqx.Module):  # Joint V+sigma neural network
  v_net: MLP   # V: R^3 → R (potential)
  sigma_net: MLP  # sigma: R^3 → R+ (conductivity, softplus)

SkullEITInversion:  # Manager
  evaluate(model, batch) → PDE_residual + data_mismatch + prior_reg
```

### ConductivityPDELoss (conductivity_pinn.py)

```python
ConductivityPDELoss(PDELoss):
  pde_residual(model, points)  # |Div(sigma*Grad(V)) + I|^2
  evaluate(model, batch)       # JINNS-compatible training
```

### BabelBrain Dataset (io/babelbrain.py)

5 subjects with T1 + UTE/ZTE/PETRA + CT ground truth.
Zenodo DOI: 10.5281/zenodo.7894431 (~1-2GB).

### MMC Pipeline (scripts/simulation/run_mmc_pipeline.py)

Uses SCI head loader → centers mesh → extracts scalp surface → places Kernel Flow optode array (hexagonal rings at 3/5/10/25mm) → exports to TetGen → runs Monte Carlo photon transport.

---

## 7. WAND Analysis Report Key Findings

### Dataset

170 healthy volunteers, CUBRIC Cardiff. Ultra-strong gradient (300 mT/m) Connectom scanner + CTF 275ch MEG + TMS-EEG.

### Section 8: FEM Head Mesh Comparison

Three-way comparison: brain2mesh vs SimNIBS charm vs MNE BEM.

| Tool | Tissues | Notes |
|------|---------|-------|
| SimNIBS charm | 10 (incl. compact/spongy bone) | DL-based, TMS-optimized, no FreeSurfer needed |
| GRACE | 11 (cortical/cancellous bone separate) | MONAI U-Net, only alternative to charm for separate bone layers |
| brain2mesh | 5 (approx skull) | iso2mesh/TetGen, customizable |
| MNE BEM | 3 shells | Standard for MEG/EEG, fast |
| T1Prep | GM/WM/CSF + thickness | CAT12 rewritten in Python (Christian Gaser), Apache 2.0 |
| FastSurfer | 95 brain classes | <1 min GPU |
| SynthSeg | 40 brain structures | 6s GPU |

### Section 9: Pseudo-CT Validation

WAND provides ground truth for skull modeling via:
- QMT bound pool fraction → bone mineral density
- VFA quantitative T1 in skull → bone density (shorter T1 = denser)
- Multi-echo GRE R2* → susceptibility/mineral content
- MP2RAGE quantitative T1 → independent T1 cross-validation

Planned analyses:
1. Plymouth DL pseudo-CT from T1w vs QMT/VFA bone density
2. charm compact/spongy boundaries vs QMT density gradient
3. Calibrate sbi4dwi PseudoCTMapper Archie's Law params using QMT porosity
4. Test: can qMRI replace pseudo-CT entirely?
5. Compare E-field simulations: charm vs Plymouth pseudo-CT vs qMRI-derived conductivity

**Potential finding:** qMRI-based skull conductivity may be more accurate than pseudo-CT.

---

## 8. SCI Institute Head Model

**Reference:** Warner A, Tate J, Burton B, Johnson CR. 2019. "A High-Resolution Head and Brain Computer Model for Forward and Inverse EEG Simulation." bioRxiv doi: 10.1101/552190

**Format:** Tetrahedral FEM mesh in .mat (MATLAB)
- `points`: (N_vertices, 3) vertex coordinates
- `cells.tetra`: (N_cells, 4) tetrahedral elements (1-indexed in file, 0-indexed after loader)
- `cell_data.tissue`: (N_cells,) integer tissue labels

**Tissue compartments:** scalp (1), skull (2), CSF (3), gray matter (4), white matter (5)

**Loader:** `sbi4dwi/dmipy_jax/io/sci_head_loader.py`
**Data:** `sbi4dwi/data/SCI_headmodel/extracted/HeadMesh.mat` (Google Drive)
**Verification:** `sbi4dwi/dmipy_jax/tests/verify_sci_loader.py`

---

## 9. NeuroTechX Global NeuroHack Context

### Format

24-36 hour hybrid event (online + in-person).
2022: 600+ participants, 48 countries, 50 projects.
2023: 1000+ participants, 10 cities, 300+ in-person.

### Tracks

1. Tech/Hardware/Hands-on
2. Data/Algorithms
3. Design/Strategy/Creative
4. Ethics

### Judging

8-11 industry experts (IDUN, Neeuro, Elemind, Timeflux, Upside Down Labs, Neurable, Cleveland Clinic, Meta Reality Labs).
Criteria: innovation, technical execution, track alignment.

### Past Winners

- **NeuroMatrix** (2023 global winner): Meditation + Alzheimer's with Neuphony EEG
- **FractedMinds** (2023 2nd): Fractal art from brain waves, sub-EUR20 Arduino EEG
- **Neuroexon** (2022): Hybrid BCI stroke rehab exoskeleton, MI+SSVEP, 24ch EEG

### What Wins

- Clear healthcare/accessibility application
- Working prototype (even minimal)
- Creative use of affordable hardware
- Real-time signal processing demo
- Strong visual presentation (5 slides, 1-2 min pitch)

---

## 10. Key Technical Decisions

### Why Not Integrate Forest Neurotech

- **mach**: Receive beamforming for imaging (inverse problem). Not therapeutic FUS (forward problem). Homogeneous medium only.
- **PyMUST**: Diagnostic imaging simulator. Point scatterers in homogeneous media. Simpler than openlifu.
- Entire stack oriented toward fUSI BCI, not therapeutic FUS.

### Why Not Integrate AE Studio NDK

- Simulation engine is **Stride/Devito** — completely different from k-Wave. Would require full rewrite.
- Transducer models limited to single-element focused bowls (no phased arrays).
- No CT/MRI processing pipeline.
- **Useful artifacts**: Aubry benchmark skull masks (.mat), material property constants.

### Why sbi4dwi + j-Wave

- Both JAX → end-to-end differentiable
- sbi4dwi has PseudoCTMapper, SCI loader, conductivity solver, EIT inversion
- j-Wave from same UCL group as k-Wave → same physics, validated
- j-Wave supports heterogeneous 3D media natively
- openlifu's SegmentationMethod extension point is ready
- Solution object bridges R&D layer to clinical workflow

### Why Two Layers (R&D + Clinical)

- Preserves openlifu's existing k-Wave validation
- JAX is opt-in (import-guarded)
- Keeps AGPL boundary clean (sbi4dwi has no openlifu imports)
- Enables cross-validation (k-Wave vs j-Wave on same geometry)

---

## 11. Acoustic Property Reference Values

### ITRUSST Benchmark + Literature

| Tissue | c (m/s) | rho (kg/m^3) | alpha (dB/cm/MHz) | Source |
|--------|---------|-------------|-------------------|--------|
| Water | 1500 | 1000 | 0.0 | Standard |
| Scalp | 1610 | 1090 | 3.5 | Connor 2002 |
| Cortical bone | 3476-4080 | 1850-1900 | 4.7-8.0 | Aubry 2022, NDK |
| Trabecular bone | 2300 | 1700 | 4.1 | NDK (alpha0=47 Np/m/MHz) |
| CSF | 1500 | 1000 | 0.0 | Standard |
| Gray matter | 1560 | 1040 | 5.3 | Connor 2002 |
| White matter | 1560 | 1040 | 5.3 | Connor 2002 |

### NDK Material Constants (for reference)

```python
cortical_bone = {'vp': 3476, 'rho': 1850, 'alpha_0': 54.553, 'alpha_power': 1.0}  # Np/m/MHz
trabecular_bone = {'vp': 2300, 'rho': 1700, 'alpha_0': 47.0, 'alpha_power': 1.2}
brain = {'vp': 1560, 'rho': 1040, 'alpha_0': 5.3, 'alpha_power': 1.0}
skin = {'vp': 1610, 'rho': 1090, 'alpha_0': 3.5, 'alpha_power': 1.0}
water = {'vp': 1500, 'rho': 1000, 'alpha_0': 0.0, 'alpha_power': 0.0}
```

### Unit Conversions

- 1 Np/m = 8.686 dB/m = 0.08686 dB/cm
- NDK cortical bone: 54.553 Np/m/MHz = 54.553 * 8.686 dB/m/MHz = 473.8 dB/m/MHz = 4.738 dB/cm/MHz
- openlifu Material uses dB/cm/MHz
- k-wave-python attenuation also in dB/cm/MHz^y (with alpha_power)
