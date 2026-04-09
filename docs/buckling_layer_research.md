# Spectral Growth Model for Cortical Folding: Theory, Implementation, and Multi-Modal Validation

Morgan G. Hough

## Abstract

We present a differentiable spectral growth model for cortical folding implemented in JAX, connecting the geometric scaling theories of Toro with the continuum mechanics of Kuhl through a unified spherical harmonic framework. The model decomposes cortical surface geometry into spectral modes, applies frequency-selective growth amplification to simulate buckling instabilities, and computes shape-theoretic curvature measures (mean, Gaussian, shape index) via discrete differential geometry — all within a fully differentiable pipeline. A companion C++ FEM simulator (Brains) provides high-fidelity nonlinear stress fields for inverse problems connecting growth parameters to dMRI-derived fractional anisotropy. We describe the implementation, derive the spectral growth equations, characterize the parameter space, and outline a validation program using the WAND multi-modal neuroimaging dataset (170 subjects, ultra-strong gradient dMRI, sub-millimeter structural MRI, quantitative MRI, MEG).

## 1. Introduction

### 1.1 The cortical folding problem

The human cortex folds into a characteristic pattern of gyri and sulci during the third trimester of gestation. This folding is not merely geometric ornamentation — it determines cortical surface area, local thickness, fiber architecture, and ultimately the computational properties of the underlying neural circuits. Understanding the mechanics of folding requires models that bridge three scales: (i) tissue-level growth and mechanics, (ii) surface-level geometry and topology, and (iii) microstructural organization measurable by diffusion MRI.

Two complementary theoretical frameworks have shaped the field:

**Geometric scaling approach (Toro & Burnod 2005; Toro & Perron 2010).** Treats folding as a spectral instability on a growing elastic shell. Differential tangential growth amplifies specific spatial frequency modes, with the dominant folding wavelength set by the ratio of cortical thickness to growth rate. This approach predicts cross-species scaling laws (cortical surface area vs. brain volume) and connects folding patterns to reaction-diffusion morphogen dynamics.

**Continuum mechanics approach (Budday, Steinmann & Kuhl 2014; Tallinen et al. 2016).** Treats the brain as a bilayer — a growing cortical plate on an elastic subcortical substrate — and solves for buckling instabilities using finite element methods. This approach predicts specific folding morphologies on patient-specific anatomies and reproduces the malformation spectrum (lissencephaly, polymicrogyria) from mechanical parameter perturbations.

### 1.2 The missing link: microstructure

Both frameworks require material parameters (stiffness ratio, growth rate, cortical thickness) that are typically assumed uniform or calibrated from ex vivo experiments. However, in vivo brain tissue is mechanically heterogeneous: regional shear moduli vary 3-fold across brain regions (Budday et al. 2017; Linka, St Pierre & Kuhl 2023), and this heterogeneity correlates with cytoarchitecture, myelination, and fiber organization — all measurable by quantitative MRI and diffusion MRI.

The key insight connecting folding mechanics to dMRI is from Garcia, Wang & Kroenke (2021): mechanical stress generated during cortical folding drives fiber reorientation in the underlying white matter. Fibers beneath sulci rotate from radial to tangential as the cortex buckles inward, producing a characteristic pattern in diffusion tensor orientation that is directly measurable by DTI. This means dMRI serves both as a *constraint* on folding model parameters (via fiber orientation priors) and a *validation target* (via predicted vs. observed orientation maps).

### 1.3 Contribution

We present a unified implementation that:
1. Implements Toro's spectral growth framework as a JAX-differentiable module (BucklingLayer)
2. Connects to Kuhl's material properties via a BrainMaterialMap with region-specific shear moduli from Constitutive Artificial Neural Networks (CANNs)
3. Provides discrete differential geometry on triangulated surfaces for curvature and shape analysis
4. Wraps a C++ FEM simulator for high-fidelity stress field computation
5. Solves the inverse problem: recover growth parameters from dMRI-derived fractional anisotropy

## 2. Theory

### 2.1 Spectral growth on a spherical surface

We represent the cortical surface as a scalar radius function on the unit sphere:

$$\mathbf{x}(\theta, \varphi) = R(\theta, \varphi) \cdot \hat{\mathbf{r}}(\theta, \varphi)$$

where $\hat{\mathbf{r}} = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)^T$ is the radial unit vector and $R(\theta, \varphi)$ is the distance from the center to the surface.

The radius function is decomposed into real spherical harmonics:

$$R(\theta, \varphi) = \sum_{l=0}^{l_{\max}} \sum_{m=-l}^{l} c_l^m \, Y_l^m(\theta, \varphi)$$

with integral normalization $\int |Y_l^m|^2 \, d\Omega = 1$ (via e3nn convention), yielding $(l_{\max}+1)^2$ coefficients.

### 2.2 Spectral growth operator

Growth is modeled as frequency-selective amplification of the spherical harmonic coefficients. Following Toro's insight that folding arises from amplification of high-spatial-frequency modes, we define a spectral filter:

$$\sigma(l) = \frac{1}{1 + \exp\bigl(-k(l - l_0)\bigr)}$$

where $l_0$ is the critical degree (soft cutoff) below which modes are preserved and above which they are amplified, and $k$ controls the sharpness of the transition.

The grown coefficients are:

$$c_l^{m\prime} = c_l^m \bigl(1 + g \cdot \sigma(l)\bigr)$$

where $g > 0$ is the growth ratio parameter. This operator has three important properties:

1. **Low-frequency preservation.** For $l \ll l_0$: $\sigma(l) \approx 0$, so $c_l^{m\prime} \approx c_l^m$. The overall brain shape (roughly spherical, set by $l=0$) is preserved.

2. **High-frequency amplification.** For $l \gg l_0$: $\sigma(l) \approx 1$, so $c_l^{m\prime} \approx c_l^m(1+g)$. Folding modes are amplified by factor $(1+g)$.

3. **Smooth transition.** The sigmoid provides a differentiable transition between preservation and amplification, parameterized by $k$.

### 2.3 Physical interpretation of parameters

| Parameter | Symbol | Default | Physical meaning |
|-----------|--------|---------|-----------------|
| Maximum SH degree | $l_{\max}$ | 8 | Spatial resolution of folding representation |
| Critical degree | $l_0$ | 2.0 | Folding onset frequency; relates to wavelength $\lambda \sim 2\pi R_0 / l_0$ |
| Transition steepness | $k$ | 5.0 | Sharpness of spectral selection |
| Growth ratio | $g$ | (fitted) | Overall buckling intensity; maps to cortical growth rate / subcortical relaxation |

The critical degree $l_0$ connects directly to folding wavelength via $\lambda = 2\pi R_0 / l_0$, where $R_0$ is the mean brain radius. For a typical brain ($R_0 \approx 70$ mm), $l_0 = 2$ gives $\lambda \approx 220$ mm (global shape), while $l_0 = 10$ gives $\lambda \approx 44$ mm (primary sulci), and $l_0 = 30$ gives $\lambda \approx 15$ mm (secondary/tertiary folds).

### 2.4 Connection to Budday-Kuhl buckling theory

Budday, Steinmann & Kuhl (2014) derive the critical folding wavelength for a bilayer:

$$\lambda_{\text{crit}} = h \cdot f\!\left(\frac{\mu_c}{\mu_s}, \frac{g_c}{g_s}\right)$$

where $h$ is cortical thickness, $\mu_c/\mu_s$ is the cortex-to-subcortex stiffness ratio, and $g_c/g_s$ is the growth rate ratio. Our spectral model captures this through the mapping:

- $l_0 \propto 2\pi R_0 / \lambda_{\text{crit}} \propto R_0 / h \cdot f(\mu_c/\mu_s, g_c/g_s)^{-1}$
- $g \propto (g_c/g_s - 1)$ at onset

The spectral model is thus a reduced-order approximation of the full FEM buckling problem, valid near the bifurcation onset where linear amplification of unstable modes dominates.

## 3. Discrete differential geometry

To characterize the folded surface, we compute intrinsic curvature measures on triangulated meshes using established discrete differential geometry (Meyer et al. 2003).

### 3.1 Mean curvature (cotangent Laplacian)

The discrete Laplace-Beltrami operator at vertex $i$ is:

$$\nabla^2 \mathbf{x}_i = \frac{1}{A_i} \sum_{j \in N(i)} \frac{\cot \alpha_{ij} + \cot \beta_{ij}}{2} (\mathbf{x}_j - \mathbf{x}_i)$$

where $\alpha_{ij}, \beta_{ij}$ are the angles opposite edge $(i,j)$ in the two incident triangles, and $A_i$ is the barycentric vertex area (1/3 of each incident face). Mean curvature is:

$$H_i = -\frac{1}{2} \nabla^2 \mathbf{x}_i \cdot \hat{\mathbf{n}}_i$$

where $\hat{\mathbf{n}}_i$ is the vertex normal (area-weighted average of incident face normals).

### 3.2 Gaussian curvature (angle deficit)

By the discrete Gauss-Bonnet theorem:

$$K_i = \frac{2\pi - \sum_f \theta_i^f}{A_i}$$

where $\theta_i^f$ is the interior angle at vertex $i$ in face $f$.

### 3.3 Shape index

The shape index (Koenderink & van Doorn 1992) maps the principal curvatures to a single scale:

$$\text{SI} = \frac{2}{\pi} \arctan\!\left(\frac{H}{\sqrt{H^2 - K}}\right)$$

ranging from $-1$ (concave cup, sulcal fundus) through $0$ (saddle, sulcal wall) to $+1$ (convex dome, gyral crown). The distribution of shape index over the cortical surface encodes the folding pattern's geometric character and serves as a summary statistic for simulation-based inference.

## 4. Implementation

### 4.1 BucklingLayer (JAX, differentiable)

Implemented as an Equinox module with $(l_{\max}+1)^2$ SH coefficients as state and $(l_0, k, g)$ as parameters. The forward pass:

1. Applies spectral growth operator to SH coefficients
2. Samples 2000 points on the sphere via Fibonacci spiral (quasi-uniform)
3. Evaluates surface position via e3nn SH basis
4. Computes first and second fundamental forms via `jax.jacfwd` and `jax.hessian`
5. Returns shape index distribution, mean/Gaussian curvature maps

The entire pipeline is differentiable via JAX autodiff, enabling gradient-based optimization of growth parameters against target curvature statistics.

### 4.2 BucklingSimulator (C++ FEM, non-differentiable)

Wraps the "Brains" binary for high-fidelity nonlinear FEM simulation on tetrahedral meshes. Input: nodal growth map ($N_{\text{nodes}} \times 1$). Output: elemental stress tensors ($N_{\text{elements}} \times 3 \times 3$). The stress tensors are converted to fractional anisotropy via eigendecomposition for comparison with dMRI:

$$\text{FA} = \sqrt{\frac{3}{2}} \frac{\sqrt{\sum_i (\lambda_i - \bar{\lambda})^2}}{\sqrt{\sum_i \lambda_i^2}}$$

where $\lambda_i$ are eigenvalues of the stress tensor.

### 4.3 Constitutive Artificial Neural Networks (CANNs)

Following Linka & Kuhl (2023, 2025), we implement a physics-informed neural network that maps strain invariants $(I_1, I_2)$ to strain energy density $\Psi$:

$$P = \frac{\partial \Psi}{\partial F} \qquad \text{via } \texttt{jax.grad}$$

where $F$ is the deformation gradient and $P$ is the first Piola-Kirchhoff stress. The architecture enforces thermodynamic consistency through non-negative weights (softplus activation) and physics-inspired hidden layer activations (square, exponential, logarithmic). Region-specific shear moduli from the Kuhl lab (cortex: 1.82 kPa, corona radiata: 0.94 kPa, corpus callosum: 0.54 kPa) serve as Bayesian priors via a dedicated prior loss term.

### 4.4 Inverse solver

The inverse problem — recover growth parameters from target FA — is solved via scipy.optimize (Powell or Nelder-Mead) wrapping the C++ simulator:

$$\hat{g} = \arg\min_g \| \text{FA}_{\text{target}} - \text{FA}\bigl(\text{Brains}(g)\bigr) \|^2$$

For the differentiable BucklingLayer path, JAX gradients replace scipy optimization, enabling faster convergence and integration with neural posterior estimation (SBI).

## 5. Validation program: WAND geometric tests

The WAND multi-modal neuroimaging dataset (170 healthy volunteers) provides simultaneous per-vertex estimation of the quantities needed to constrain and validate the folding model:

### 5.1 Test 1: Thickness predicts folding wavelength

**Measurement:** Sub-millimeter (0.67 mm) structural MRI provides per-vertex cortical thickness $h$ and curvature (via FreeSurfer or the discrete geometry module).

**Prediction:** Budday-Kuhl theory: $\lambda \propto h \cdot f(\mu_c/\mu_s)$. With stiffness ratio held constant, local thickness should predict local folding wavelength. Across 170 subjects, inter-subject thickness variation should correlate with inter-subject folding wavelength variation at matched cortical locations.

**Analysis:** Compute per-vertex folding wavelength from the power spectrum of the mean curvature field (peak spatial frequency on the cortical surface mesh). Correlate with per-vertex thickness. Test whether the Budday-Kuhl relationship holds within-subject (across cortical regions) and across-subject (at matched atlas locations).

### 5.2 Test 2: Axon diameter modulates stiffness ratio

**Measurement:** WAND's 300 mT/m ultra-strong gradient dMRI enables AxCaliber axon diameter estimation unavailable on standard scanners. Larger axon diameters produce stiffer white matter.

**Prediction:** Regions with larger axon diameters (higher WM stiffness, lower stiffness ratio $\mu_c/\mu_s$) should show less folding (larger wavelength, shallower sulci). This is a direct test of the mechanical model's stiffness-ratio parameter.

**Analysis:** Map AxCaliber axon diameter to each cortical vertex via volume-to-surface projection. Correlate with local folding intensity (gyrification index, sulcal depth). Test whether the relationship follows the Budday-Kuhl stiffness-ratio dependence.

### 5.3 Test 3: Myelin maps as stiffness proxy

**Measurement:** QMT + T1/T2 ratio provides per-voxel myelin content. Myelinated tissue is stiffer.

**Prediction:** Regional myelin variation from quantitative MRI should predict regional folding patterns via the stiffness-ratio mechanism. Individual myelin maps should predict individual folding patterns better than population-average material maps.

**Analysis:** Replace the BrainMaterialMap's atlas-based shear moduli with individual-specific estimates derived from quantitative MRI. Compare folding predictions (BucklingLayer forward model) against observed folding (structural MRI) with individual vs. population-average material maps. The improvement in prediction accuracy quantifies the value of individual-specific stiffness information.

### 5.4 Test 4: Fiber orientation validates stress predictions

**Measurement:** High angular resolution dMRI provides fiber orientation distributions. Garcia et al. (2021) showed that fibers beneath sulci rotate from radial to tangential during folding.

**Prediction:** The Brains FEM simulator produces stress tensors whose principal eigenvectors should align with measured fiber orientations. Specifically, the principal stress direction beneath sulci should be tangential (matching compressed fibers), and beneath gyri should be radial (matching stretched fibers).

**Analysis:** Run the Brains simulator with growth parameters fit to each subject's thickness map. Extract principal stress directions. Compare with DTI primary eigenvectors projected to the cortical surface. Compute angular error per vertex. This tests the mechanical model's stress field predictions directly.

### 5.5 Test 5: Shape index distribution as SBI summary statistic

**Measurement:** Shape index from structural MRI surface reconstruction. The shape index distribution over the cortical surface encodes the character of folding.

**Prediction:** The BucklingLayer's spectral growth model predicts a shape index distribution that depends on $(l_0, k, g)$. Different parameter regimes produce different distributions: low $g$ gives nearly uniform SI $\approx 1$ (smooth sphere); high $g$ gives a bimodal distribution with peaks at SI $\approx \pm 0.5$ (ridges and valleys).

**Analysis:** Use neural posterior estimation (sbi library, SNPE) with the BucklingLayer as the forward simulator and the shape index distribution as the summary statistic. Infer $(l_0, k, g)$ per subject from their structural MRI surface. The posterior uncertainty quantifies parameter identifiability. Cross-validate by predicting held-out curvature statistics.

### 5.6 Test 6: Cross-modal geometric consistency

**Measurement:** All of the above, simultaneously, on the same 170 subjects.

**Prediction:** If the mechanical model is correct, then brain regions with unusual thickness (structural MRI), unusual stiffness proxy (dMRI axon diameter + QMT myelin), and unusual folding (curvature from surface reconstruction) should be the same regions — and these regions should also show unusual neural dynamics (MEG spectral peaks, via the CMC model with layer-resolved connectivity from vpjax).

**Analysis:** Multivariate correlation across modalities at each cortical vertex, across 170 subjects. Canonical correlation analysis between the folding-mechanics feature vector (thickness, axon diameter, myelin, stiffness ratio) and the dynamics feature vector (alpha peak frequency, amplitude, sp/dp ratio from CMC source localization). A significant cross-modal correlation would demonstrate that the same mechanical parameters that predict folding geometry also predict neural dynamics — unifying brain structure and function through mechanics.

## 6. Connection to neural dynamics (CMC)

The BucklingLayer's output — a folded cortical surface with per-vertex curvature — provides the geometric substrate for the canonical microcircuit (CMC) model implemented in vbjax. The connection operates through three mechanisms:

1. **Layer geometry.** On a folded surface, the cortical layers are distorted: compressed at sulcal fundi, stretched at gyral crowns. This affects the effective connectivity between CMC populations (ss, sp, ii, dp), since synaptic density per unit cortical surface area varies with local curvature.

2. **vpjax layer-resolved BOLD.** The `cmc_to_layer_activity` function maps CMC population activity to the three-layer model used by vpjax's `layer_stimulus`. On a folded surface, the ascending vein contamination pattern (vpjax's `ascending_vein_contamination`) depends on local cortical depth profiles, which vary with curvature.

3. **Forward/backward asymmetry.** The CMC's hierarchical predictive coding architecture (forward connections target ss/L4, backward connections target sp+dp/agranular layers) interacts with folding because sulcal walls contain the boundaries between cytoarchitectonic areas where forward and backward connections are most dense. The `cmc_hier_Nnode_dfun` function, coupled with a folding-informed connectivity matrix, tests whether the geometry of cortical folds affects the dynamics of predictive coding.

## References

- Bastos AM et al. (2012) Canonical microcircuits for predictive coding. Neuron 76(4):695-711.
- Budday S, Steinmann P, Kuhl E (2014) The role of mechanics during brain development. J Mech Phys Solids 72:75-92.
- Budday S et al. (2017) Mechanical properties of gray and white matter brain tissue by indentation. J Mech Behav Biomed Mater 46:318-330.
- Budday S, Raybaud C, Kuhl E (2014) A mechanical model predicts morphological abnormalities in the developing human brain. Sci Reports 4:5644.
- Douglas PK (2025) Computing with canonical microcircuits. arXiv:2508.06501.
- Garcia KE, Wang X, Bhatt Kroenke CD (2021) A model of tension-induced fiber growth predicts white matter organization during brain folding. Nature Comms 12:6681.
- Holland MA et al. (2020) Folding drives cortical thickness variations. Eur Phys J Special Topics 229:2757-2778.
- Koenderink JJ, van Doorn AJ (1992) Surface shape and curvature scales. Image Vision Comput 10(8):557-564.
- Linka K, St Pierre SR, Kuhl E (2023) Automated model discovery for human brain using CANNs. Acta Biomaterialia 160:134-151.
- Linka K, Kuhl E (2025) Bayesian CANNs. CMAME 433.
- Meyer M et al. (2003) Discrete differential-geometry operators for triangulated 2-manifolds. In: Visualization and Mathematics III, Springer.
- Tallinen T et al. (2016) On the growth and form of cortical convolutions. Nature Physics 12:588-593.
- Toro R, Burnod Y (2005) A morphogenetic model for the development of cortical convolutions. Cerebral Cortex 15(12):1900-1913.
- Toro R, Perron M (2010) [with Lefebvre] A reaction-diffusion model of human brain development. PLoS Comp Biol 6(4):e1000749.
