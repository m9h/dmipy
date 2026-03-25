# External Projects Reference

Projects that were previously vendored or referenced in this repository.
Removed during repo cleanup (2026-03-25) to reduce tracked size.

## dmipy (Bayesian fork)

- **Source**: https://github.com/AthenaEPI/dmipy
- **License**: MIT (Rutger Fick & Demian Wassermann, 2017); separate
  LICENSE-BAYESIAN for Bayesian fitting additions
- **Version vendored**: 1.0.5 (with Bayesian extensions)
- **Was located at**: `benchmarks/external/dmipy_bayesian/`
- **Purpose**: Legacy benchmarking baseline. Contained the original dmipy
  package plus Bayesian fitting notebooks (`fit_bayes.py`,
  `bayesian-fitting-HCP-example.ipynb`) used to compare against dmipy-jax
  SBI posteriors. Included pre-built eggs (`dmipy-1.0.5-py2.7.egg`,
  `dmipy-1.0.5-py3.9.egg`), Camino simulation data, and example notebooks.
- **Why removed**: 308 tracked files (~113 MB), including 54 MB of egg
  distributions. The upstream repo is publicly available and the comparison
  is reproducible by installing `dmipy==1.0.5` from PyPI or the GitHub repo.

## CATERPillar

- **Source**: https://github.com/RafaelNH/CATERPillar
- **Was located at**: `vendor/CATERPillar/`
- **Purpose**: C++ tool for generating realistic axon geometry phantoms
  (Computer-Assisted Tissue Engineering for Reproducible Phantoms in
  Localised fibre arrangements). Intended for generating ground-truth
  mesh geometries to feed into the FEM `MatrixFormalismSimulator`
  (`dmipy_jax/simulation/mesh_sim.py`).
- **Why removed**: Directory was empty (placeholder only, never populated).
  If needed in future, clone the upstream repo or add as a git submodule.

## ReMiDi

- **Source**: https://remidi.org / https://github.com/jingrebeccali/ReMiDi
- **Was located at**: `ReMiDi/` (root)
- **Purpose**: Realistic Microstructure Diffusion simulator. GPU-accelerated
  Monte Carlo random-walk simulator for diffusion MRI in complex geometries.
  Used as an oracle simulator in the multi-fidelity pipeline.
- **Integration**: The oracle wrapper lives at
  `dmipy_jax/simulation/oracles/remidi.py` and communicates with ReMiDi via
  Docker (`docker/Dockerfile.remidi`) or subprocess — it does not require
  vendored source.
- **Why removed**: Directory was empty (placeholder only, never populated).
  The Docker-based oracle integration is the intended interface.
