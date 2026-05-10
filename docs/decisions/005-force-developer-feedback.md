# FORCE developer feedback (private, not for upstream)

**Date:** 2026-05-10
**Author:** Morgan Hough (sbi4dwi)
**Context:** Replicating FORCE paper §3.2 (DiSCo phantom) and §3.3 (Stanford HARDI) on dipy 1.12.1, then comparing dmipy-JAX dictionary matching against FORCE upstream.
**Audience:** Atharva Shah, Rafael Henriques, Alonso Ramirez-Manzanares, Eleftherios Garyfallidis — the FORCE paper authors / DIPY maintainers, when we have a chance to talk to them.

The findings below are *positive about FORCE's methodology* — most surface places where the upstream packaging or tutorial could be improved so that new users don't fall into the same library-prior pitfall I did on this project.

## 1. The thing the paper says clearly but the tutorial hides

**Paper §3.2 (DiSCo phantom validation), verbatim:**

> Minor adjustments were introduced to align the forward model with the
> characteristics of this numerical phantom as it departs from the regime
> of usual biological tissue diffusion parameters. To ensure consistency
> with the DiSCo simulation model, which represents diffusion using
> stick-like compartments, **diffusivities were sampled from narrow
> uniform bands: D_∥ from Uniform(0.54, 0.66) × 10⁻³ mm²/s and D_⊥ from
> Uniform(0.32, 0.38) × 10⁻³ mm²/s. The isotropic compartment was
> disabled** to match the stick-like DiSCo model.

This is the key methodological footnote. It says: **the library prior is part of the method, and must be retuned per-dataset whenever the data depart from the in-vivo regime.**

But this insight is essentially absent from `dipy/doc/examples/reconst_force.py`, where:

- `model.generate(num_simulations=500000, num_cpus=-1, verbose=True, use_cache=False)` is called with no `diffusivity_config`.
- There is no explanation of when to retune (vs trust defaults).
- The example uses Stanford HARDI (in-vivo, where defaults work) — so a new user runs it, sees plausible output, assumes defaults are universal, and applies them to (e.g.) a phantom or ex-vivo dataset where they aren't.

## 2. Empirical demonstration of why this matters

DiSCo phantom subject 1 highRes, single-shell b=1900, SNR=30, 15,267 masked voxels:

| Method | FORCE NDI vs GT Intra Volume Fraction | FORCE FA vs DTI FA |
|---|:-:|:-:|
| Default library (`wm_d_par_range=(0.002, 0.003)`) | **r = 0.679** | r = 0.827 |
| Paper-protocol retune (`wm_d_par_range=(0.00054, 0.00066)`) | **r = 0.918** | **r = 0.990** |

**24 Pearson-r-point improvement from a ~10-line `diffusivity_config` change.**

More importantly: the **default-library run's NDI mean was 0.75** while ground truth's was ~0.3. That is a **uniform 0.4-magnitude over-estimation across the entire brain mask**. The spatial pattern still tracked GT (r=0.679 isn't zero), so visual inspection alone of the FORCE NDI map would *not* surface the problem. A new user could publish or clinically apply biased numbers in good faith.

## 3. Suggested upstream improvements

Listed in rough order of effort × value.

### (a) Add a paragraph to `dipy/doc/examples/reconst_force.py`

Right after `model.generate(...)`, something like:

```rst
.. note::
   The default diffusivity priors (``D_∥ ~ Uniform(2.0, 3.0)×10⁻³ mm²/s``,
   ``D_⊥ ~ Uniform(0.3, 1.5)×10⁻³ mm²/s``) are calibrated for in-vivo
   human brain at typical clinical resolution. If your data depart from
   this regime — e.g. ex-vivo / fixed tissue, numerical phantoms,
   high-resolution preclinical scanners — **retune via
   `diffusivity_config`**. See FORCE paper §3.2 for an example
   (DiSCo phantom, ``D_∥ ~ Uniform(0.54, 0.66)×10⁻³``).
```

This is 8 lines of docs and would have saved me a full afternoon of debugging in this project.

### (b) Add a tutorial example or section that retunes for DiSCo

DiSCo is already in dipy via `fetch_disco1_dataset()`. A second tutorial example titled "FORCE on the DiSCo phantom" that walks through:

1. Fetching DiSCo
2. Sub-selecting single-shell b≈2000
3. Generating a DiSCo-tuned library with the exact priors from paper §3.2
4. Fitting + showing the NDI ground-truth correlation

…would make the retuning step concrete and copy-pasteable. The current paper has the recipe; a tutorial would surface it.

### (c) Diagnostic: emit a warning when input data appear out-of-distribution

A simple sanity check at the top of `FORCEModel.fit`:

```python
# Heuristic: if observed mean MD outside library's library range,
# warn the user about library/data mismatch.
```

Could compute a rough DTI MD on the input and compare to the library's
expected MD range. If they're a factor of 2+ apart, emit a clear warning
pointing at `diffusivity_config`. Not strictly necessary but would
catch the failure mode early.

### (d) `FORCEModel.__init__` could store the diffusivity ranges from the simulations dict so users can inspect what they're using

Right now you have to either remember what you passed to `generate_force_simulations` or inspect `sims["wm_d_par_range"]`-like keys (if even saved). Surfacing this on the `FORCEModel` instance (`model.diffusivity_config` or `model.summary()` would help post-hoc audits and tutorial reproducibility.

## 4. Other observations from the reproduction work

These are smaller, but in the spirit of "issues a careful user trips over":

### 4a. JAX-fork incompatibility with `num_cpus > 1`

`generate_force_simulations` uses `multiprocessing` with the default `fork` start method on Linux. If JAX has been imported before this call, the fork copies JAX's background threads into the children → deadlock risk. The runtime emits this warning:

```
RuntimeWarning: os.fork() was called. os.fork() is incompatible with
multithreaded code, and JAX is multithreaded, so this will likely lead
to a deadlock.
```

In our work it silently worked at `num_cpus=20`, but the warning is a real footgun. Two cheap fixes for upstream:

- Document this clearly in the `generate_force_simulations` docstring with a recommendation: "If JAX is loaded in the calling process, use `num_cpus=1` or `multiprocessing.set_start_method('spawn')` first."
- Or, if feasible, internally use `multiprocessing.get_context('spawn').Pool(...)` so the worker processes don't inherit the parent's threading state.

### 4b. Library-stored signals use 0–100 scale, not unit-S0

`sims["signals"]` has signal range ~5–100 per voxel, not 0–1. Initially I tried to round-trip a stored signal through `model.fit` as a wiring sanity check and got `FORCEFit.label.sum() == 0`. The reason is that `model.fit` expects unit-S0-normalised input (which it then internally renormalises via cosine similarity). Stored library signals are at the raw amplitude scale.

This is **fine** as long as users always pass real DWI data (S0-normalised by acquisition), but it surprised me as a developer. Worth a comment in the docstring of `save_force_simulations` / `load_force_simulations` explaining the storage convention.

### 4c. Default sphere is 362 vertices; paper uses 724

`generate_force_simulations` doesn't expose a `sphere` argument — the
internal default is `default_sphere` (362 vertices). Paper §2.2.2 (line 182):

> Orientations were sampled uniformly over a unit sphere using a
> **724-vertex electrostatic grid**, providing an angular resolution
> of approximately 4.1° between nearest vertices.

The 362-vertex `default_sphere` has nearest-vertex error ~4.1° too (similar density). But applications requiring sub-4° angular resolution would benefit from being able to pass `sphere=Symmetric724`. Probably a one-line addition to the `generate_force_simulations` signature.

### 4d. `wm_threshold=1.0` silently suppresses ODF generation in the library

Found while reproducing paper §3.2's DiSCo connectivity-matrix protocol.
The paper explicitly says "**The isotropic compartment was disabled** to
match the stick-like DiSCo model", which I implemented as
`wm_threshold=1.0` (the parameter's documented purpose). The resulting
500K-entry library has:

- All scalar metrics (`fa`, `md`, `nd`, `dispersion`, `wm_fraction`, …)
  populated correctly — matcher works, NDI recovery r = 0.918.
- **`sims["odfs"]` is *all zeros* — 0 of 500,000 entries have any
  nonzero ODF value.**

`force_peaks(fit)` builds its `PeaksAndMetrics.peak_dirs` from the
posterior-weighted average of library ODFs, so on a tuned-library fit
it returns zero peaks per voxel — even when `fit.num_fibers > 0` says
the matcher found multi-fibre configurations. Tractography on those
peaks produces zero streamlines, which breaks any downstream
connectivity-matrix or tractography pipeline.

Compare same fixture with the default library: 249,898 / 500,000 ODF
rows nonzero, tractography works.

This is silent and dangerous: the matcher reports `num_fibers > 0`,
the user reasonably assumes peaks exist, and the empty `peak_dirs`
only surfaces when the downstream pipeline crashes (zero streamlines).
Suggestions for upstream:

- Emit a `RuntimeWarning` if `generate_force_simulations` would produce
  zero nonzero-ODF rows.
- Or: ensure ODFs are populated regardless of `wm_threshold`, since
  ODFs are needed for `force_peaks` regardless of the WM-vs-GM/CSF
  fraction policy.
- Or at minimum: document that `wm_threshold=1.0` will suppress ODFs
  and recommend using something like `wm_threshold=0.95` instead, or
  setting `compute_odfs=True` explicitly if such a flag exists.

Pinned in our tests as `test_tuned_library_suppresses_odfs_upstream_bug`
— it will turn red the day upstream fixes the issue.

### 4e. The 70%-3-fibre library composition is undocumented

`Dirichlet(2,1,1)` over (WM, GM, FW) fractions, with WM containing up to 3 fibre populations, results in:

| n_fibres | Library fraction |
|:-:|:-:|
| 1 | 10.0% |
| 2 | 20.0% |
| 3 | **69.9%** |

This means the library is dominated by 3-fibre configurations. For users analysing data with predominantly 1–2 fibre populations (most of the brain), only ~30% of the library is "useful". This is **fine** — FORCE was designed to be general — but it's not documented anywhere I can find.

A `verbose=True` printout from `generate_force_simulations` reporting the achieved n_fibre breakdown (and the total parameter coverage of the library) would help users decide whether their library is dense enough for their geometry of interest. A small histogram of `sims["num_fibers"]` is all it would take.

## 5. Net assessment of FORCE

After substantial back-and-forth in this project's docs/decisions/004, my honest assessment of FORCE upstream:

- **The method works.** §14 (Stanford HARDI maps) and §15 (FORCE vs DTI r=0.985) reproduce the paper's positive claims on its design data.
- **Library-prior alignment is critical.** §16.3 (tuned r=0.918) vs §16.2 (default r=0.679) demonstrates this at a 24-r-point swing.
- **The retuning step is documented in the paper but undersold in the tutorial.** That's the user-experience gap worth surfacing.
- **There are several small footguns** (JAX fork, signal scale, sphere quantisation, fibre-fraction prior) — all easy to document or fix.

If I had read the paper before benchmarking, the §13 / §16.2 wasted compute would not have happened. The fact that I (a careful user with the paper open) still tripped on this is the strongest evidence I have for the "document the retune step" suggestion above.

## 6. What we have running on top of FORCE in sbi4dwi

For context, when discussing with the FORCE authors: the sbi4dwi project has built a dmipy-JAX dictionary matcher (`dmipy_jax/library/matcher.py`) inspired by FORCE but with:

- **Native JAX implementation** (cosine similarity on GPU, no FAISS dependency).
- **Composable forward models** — any combination of dmipy compartments (stick, zeppelin, restricted cylinders, sphere/SANDI, Bingham, Watson, IVIM…), not just FORCE's fixed stick + zeppelin + Bingham + ball.
- **Hybrid initialisation** — dictionary match → Optimistix LM refinement for continuous parameter estimates.
- **End-to-end differentiability** — gradient-based acquisition optimisation via Fisher information / EIG.

The natural collaboration story is: dmipy-JAX adds gradient-based extensions and differentiable physics to the FORCE paradigm; FORCE provides the canonical biophysics + reference implementation. There is no competition story — these are complementary in a way the paper's §6 "future work" section gestures at.
