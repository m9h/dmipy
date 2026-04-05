"""
Modal.com: Full TUS experiment suite.

Runs all experiments on A100 GPU, collects results, prints summary.

    modal run scripts/modal_experiments.py
"""

import modal

app = modal.App("tus-experiments")

SCI_SEG_URL = "https://sci.utah.edu/~datasets/SCI_headmodel/Segmentation.zip"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip", "wget", "git")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "jaxlib==0.4.38",
        "jwave==0.2.1",
        "optax==0.2.5",
        "equinox==0.11.12",
        "scipy",
        "numpy<2",
        "xarray",
        "h5py",
        "nibabel",
        "pynrrd",
    )
    .run_commands(
        # Cache-bust: pin to specific commit so Modal rebuilds when code changes
        "git clone --depth 1 https://github.com/m9h/sbi4dwi.git /opt/sbi4dwi"
        " && cd /opt/sbi4dwi && git log --oneline -1",
        "echo '/opt/sbi4dwi' > /usr/local/lib/python3.12/site-packages/sbi4dwi.pth",
    )
    .env({"SBI4DWI_COMMIT": "df03ecd"})  # bump this to bust cache
    .run_commands(
        f"mkdir -p /sci_head && wget -q -O /sci_head/Segmentation.zip {SCI_SEG_URL}",
        "cd /sci_head && unzip -q Segmentation.zip",
    )
)


@app.function(image=image, gpu="A100", timeout=3600)
def run_all_experiments():
    import time
    import json
    import numpy as np
    import jax
    import jax.numpy as jnp
    import importlib.util, sys, os

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    SBI = "/opt/sbi4dwi/dmipy_jax"
    acoustic = _load("acoustic", f"{SBI}/biophysics/acoustic.py")
    jwa = _load("jwave_adapter", f"{SBI}/biophysics/jwave_adapter.py")
    exp = _load("tus_experiments", f"{SBI}/biophysics/tus_experiments.py")

    print(f"JAX: {jax.__version__}, devices: {jax.devices()}, backend: {jax.default_backend()}")
    print()

    all_results = {}

    # ==================================================================
    # Experiment 1: SCI Head Segmentation Volume (from initial pipeline)
    # ==================================================================
    print("=" * 70)
    print("EXP 1: SCI Head Model — 2D Axial Simulation + Optimization")
    print("=" * 70)
    t0 = time.time()

    import glob, nrrd
    seg_files = glob.glob("/sci_head/**/*.nrrd", recursive=True)
    if seg_files:
        data, header = nrrd.read(seg_files[0])
        labels = np.array(data, dtype=np.int32)
        props = acoustic.map_labels_to_properties(jnp.array(labels))

        mid_z = labels.shape[2] // 2
        c_slice = np.array(props["sound_speed"][:, :, mid_z])
        rho_slice = np.array(props["density"][:, :, mid_z])
        spacing_m = 1e-3

        brain_mask = (c_slice > 1500) & (c_slice < 1600)
        brain_coords = np.argwhere(brain_mask)
        target = tuple(brain_coords[len(brain_coords)//2]) if len(brain_coords) > 0 else (c_slice.shape[0]//2, c_slice.shape[1]//2)
        scalp_mask = (c_slice > 1600) & (c_slice < 1620)
        scalp_coords = np.argwhere(scalp_mask)
        source_pos = tuple(scalp_coords[0]) if len(scalp_coords) > 0 else (c_slice.shape[0]//2, 5)

        r_w = jwa.run_simulation_2d(c_slice.shape, spacing_m, 1500.0, 1000.0, source_pos, 400e3, 5e-5, pml_size=10)
        r_s = jwa.run_simulation_2d(c_slice.shape, spacing_m,
                                     jnp.array(c_slice, dtype=jnp.float32),
                                     jnp.array(rho_slice, dtype=jnp.float32),
                                     source_pos, 400e3, 5e-5, pml_size=10)
        p_w = float(r_w["p_max"][target])
        p_s = float(r_s["p_max"][target])
        atten = 100*(1-p_s/p_w) if p_w > 0 else 0

        all_results["sci_head_2d"] = {
            "volume_shape": list(labels.shape),
            "tissue_types": int(len(np.unique(labels))),
            "p_water": p_w, "p_skull": p_s,
            "attenuation_pct": atten,
            "time_s": time.time() - t0,
        }
        print(f"  Attenuation: {atten:.1f}%, Water: {p_w:.6f}, Skull: {p_s:.6f}")
    else:
        print("  SKIPPED: No segmentation NRRD found")
    print()

    # ==================================================================
    # Experiment 2: Multi-Element Array Optimization (16 + 32 elements)
    # ==================================================================
    print("=" * 70)
    print("EXP 2: Multi-Element Array Optimization")
    print("=" * 70)

    for n_el in [4, 16, 32]:
        print(f"  {n_el} elements...")
        t0 = time.time()
        try:
            res = exp.run_array_optimization(
                n_elements=n_el, grid_size=64, spacing_mm=0.2,
                freq=400e3, n_iters=20, lr=1e-7, skull_thickness_voxels=8,
            )
            all_results[f"array_{n_el}el"] = res
            print(f"    Time: {res['time_s']:.1f}s ({res['time_per_iter']:.1f}s/iter)")
            print(f"    Loss: {res['loss_history'][0]:.6f} -> {res['loss_history'][-1]:.6f}")
            print(f"    Improvement: {res['improvement']:.1f}x")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[f"array_{n_el}el"] = {"error": str(e)}
    print()

    # ==================================================================
    # Experiment 3: Multi-Target
    # ==================================================================
    print("=" * 70)
    print("EXP 3: Multi-Target Simulation")
    print("=" * 70)
    t0 = time.time()
    try:
        targets = {
            "shallow_cortex": (32, 35),
            "mid_brain": (32, 45),
            "deep_thalamus": (32, 55),
        }
        mt_res = exp.simulate_at_targets(
            grid_size=64, spacing_mm=0.2, freq=400e3,
            targets=targets, skull_range=(15, 25),
        )
        all_results["multi_target"] = mt_res
        for name, r in mt_res.items():
            print(f"  {name}: p_water={r['p_water']:.6f}, p_skull={r['p_skull']:.6f}, atten={r['attenuation_pct']:.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["multi_target"] = {"error": str(e)}
    print()

    # ==================================================================
    # Experiment 4: Multi-Frequency (180kHz vs 400kHz vs 1MHz)
    # ==================================================================
    print("=" * 70)
    print("EXP 4: Frequency Comparison")
    print("=" * 70)
    try:
        freq_res = exp.compare_frequencies(
            grid_size=64, spacing_mm=0.2,
            frequencies=[180e3, 400e3, 1e6],
            skull_range=(15, 25),
            target=(32, 50),
        )
        all_results["multi_freq"] = {str(k): v for k, v in freq_res.items()}
        for freq, r in freq_res.items():
            print(f"  {freq/1e3:.0f}kHz: p_skull={r['p_skull']:.6f}, atten={r['attenuation_pct']:.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["multi_freq"] = {"error": str(e)}
    print()

    # ==================================================================
    # Experiment 5: Grid Convergence
    # ==================================================================
    print("=" * 70)
    print("EXP 5: Grid Convergence Study")
    print("=" * 70)
    try:
        gc_res = exp.grid_convergence_study(
            spacings_mm=[0.4, 0.2, 0.1],
            base_size_mm=12.8, freq=400e3,
        )
        all_results["grid_convergence"] = {str(k): v for k, v in gc_res.items()}
        for sp, r in gc_res.items():
            print(f"  {sp}mm ({r['grid_size']}^2): p_max={r['p_max']:.6f}, p_target={r['p_target']:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["grid_convergence"] = {"error": str(e)}
    print()

    # ==================================================================
    # Experiment 6: Sensitivity Analysis
    # ==================================================================
    print("=" * 70)
    print("EXP 6: Sensitivity Analysis (dp/dc)")
    print("=" * 70)
    try:
        sens = exp.compute_sensitivity_map(
            grid_size=32, spacing_mm=0.5, freq=400e3,
            target=(16, 25), skull_range=(8, 12),
        )
        skull_sens = float(jnp.abs(sens[8:12, :]).mean())
        water_sens = float(jnp.abs(sens[0:5, :]).mean())
        brain_sens = float(jnp.abs(sens[15:25, :]).mean())
        all_results["sensitivity"] = {
            "skull_mean_abs": skull_sens,
            "water_mean_abs": water_sens,
            "brain_mean_abs": brain_sens,
            "skull_to_water_ratio": skull_sens / water_sens if water_sens > 0 else float('inf'),
        }
        print(f"  Skull region: {skull_sens:.8f}")
        print(f"  Water region: {water_sens:.8f}")
        print(f"  Brain region: {brain_sens:.8f}")
        print(f"  Skull/Water ratio: {skull_sens/water_sens:.1f}x" if water_sens > 0 else "  Skull/Water: inf")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["sensitivity"] = {"error": str(e)}
    print()

    # ==================================================================
    # Experiment 7: Helmholtz vs Time-Domain
    # ==================================================================
    print("=" * 70)
    print("EXP 7: Helmholtz vs Time-Domain Comparison")
    print("=" * 70)
    try:
        hz_res = exp.compare_helmholtz_vs_timedomain(grid_size=64, spacing_mm=0.2, freq=400e3)
        all_results["helmholtz_comparison"] = hz_res
        print(f"  Time-domain p_max: {hz_res['td_p_max']:.6f}")
        print(f"  Helmholtz p_max: {hz_res['hz_p_max']:.6f}")
        print(f"  Correlation: {hz_res['correlation']:.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["helmholtz_comparison"] = {"error": str(e)}
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    n_success = sum(1 for v in all_results.values() if "error" not in v)
    n_total = len(all_results)
    print(f"  {n_success}/{n_total} experiments succeeded")
    print()

    return all_results


@app.local_entrypoint()
def main():
    import json
    results = run_all_experiments.remote()
    print("\n" + "=" * 70)
    print("RESULTS JSON")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
