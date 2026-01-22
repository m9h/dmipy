---
description: "ReMiDi Educational Synergies: Lesson Plans and Prompts"
---

# ReMiDi Educational Synergies

Using ReMiDi as a **Ground Truth Oracle** to teach Differentiable Science with `dmipy-jax`.

## 1. The "Model Mismatch" Lesson
**Concept:** Demonstrate the limits of analytical models (Cylinder) on realistic geometries.
**Prompt:**
> "I want to create a demonstration of 'Model Mismatch'. 
> 1. Use ReMiDi to generate a dataset of 'Beading' cylinders with varying `beading_amplitude` (0.0 to 0.5).
> 2. Implement a fitting loop using `dmipy-jax`'s `Cylinder` model.
> 3. Plot the 'Fitting Error' and 'Estimated Radius' vs `beading_amplitude`.
> 4. Show that the model breaks down as the geometry deviates from the analytical assumption."

## 2. The "Neural Surrogate" Lesson
**Concept:** Teach how to create millisecond-fast differentiable surrogates for slow simulations.
**Prompt:**
> "I want to demonstrate the power of Neural Surrogates.
> 1. Create a `NeuralCDE` network in `dmipy-jax`.
> 2. Train it on a dataset of ReMiDi signals parameterized by `(beading_amplitude, frequency)`.
> 3. Compare the inference speed: Benchmark the time to generate 1000 signals using the Neural CDE vs the ReMiDi Monte Carlo simulation.
> 4. Visualize the accuracy of the surrogate across the parameter space."

## 3. The "Protocol Optimization" Lesson (OED)
**Concept:** Use Differentiable OED to find waveforms sensitive to specific features.
**Prompt:**
> "I want to find the optimal waveform to detect 'Beading'.
> 1. Define a binary OED task: Distinguish `amplitude=0.1` from `amplitude=0.3`.
> 2. Use `AlgebraicOED` to optimize a Free-Form Gradient Waveform (using the trained Neural Surrogate from Lesson 2 as the forward model).
> 3. Compare the Fisher Information of the Optimized Waveform vs a standard Stejskal-Tanner PGSE sequence.
> 4. Show that the optimized waveform has >10x sensitivity."

## 4. The "Global Connectivity" Lesson
**Concept:** Global fiber tracking on ground-truth phantoms.
**Prompt:**
> "I want to Validate Global Tractography.
> 1. Use the 'Fanning' ReMiDi example to generate a dataset with known fiber ground truth.
> 2. Run the `GlobalAMICOSolver` on this dataset to reconstruct the fiber density.
> 3. Vary the Total Variation regularization (`lambda_tv`) and plot the reconstruction quality (MSE vs Ground Truth).
> 4. Demonstrate how spatial regularization cleans up the reconstruction."
