"""
Modal.com: Full 3D TUS validation on SCI head model.

3D volumetric simulation through real skull anatomy, multi-element array
optimization, and focal spot analysis in 3D.

    modal run scripts/modal_3d_validation.py
"""

import modal

app = modal.App("tus-3d-validation")

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
    .env({"SBI4DWI_COMMIT": "9730995"})
    .run_commands(
        "rm -rf /opt/sbi4dwi"
        " && git clone --depth 1 https://github.com/m9h/sbi4dwi.git /opt/sbi4dwi"
        " && cd /opt/sbi4dwi && git log --oneline -1",
        "echo '/opt/sbi4dwi' > /usr/local/lib/python3.12/site-packages/sbi4dwi.pth",
    )
    .run_commands(
        f"mkdir -p /sci_head && wget -q -O /sci_head/Segmentation.zip {SCI_SEG_URL}",
        "cd /sci_head && unzip -q Segmentation.zip",
    )
)


@app.function(image=image, gpu="A100", timeout=3600, memory=32768)
def run_3d_validation():
    import time
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

    print(f"JAX: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()

    all_results = {}

    # ==================================================================
    # Step 1: Load SCI segmentation and downsample to 3D grids
    # ==================================================================
    print("=" * 70)
    print("STEP 1: Load SCI Head Segmentation")
    print("=" * 70)

    import glob, nrrd
    seg_files = glob.glob("/sci_head/**/*.nrrd", recursive=True)
    data, header = nrrd.read(seg_files[0])
    labels_full = np.array(data, dtype=np.int32)
    print(f"  Full volume: {labels_full.shape} at 1mm")
    print(f"  Tissue labels: {np.unique(labels_full)}")
    print()

    # Downsample by block-mode (most common label in each block)
    def downsample_labels(labels, factor):
        from scipy.ndimage import uniform_filter
        shape = labels.shape
        new_shape = tuple(s // factor for s in shape)
        result = np.zeros(new_shape, dtype=np.int32)
        for iz in range(new_shape[2]):
            for iy in range(new_shape[1]):
                for ix in range(new_shape[0]):
                    block = labels[
                        ix*factor:(ix+1)*factor,
                        iy*factor:(iy+1)*factor,
                        iz*factor:(iz+1)*factor,
                    ]
                    vals, counts = np.unique(block, return_counts=True)
                    result[ix, iy, iz] = vals[np.argmax(counts)]
        return result

    # Create multiple resolutions
    print("  Downsampling...")
    t0 = time.time()
    labels_4mm = downsample_labels(labels_full, 4)  # ~52x64x64
    labels_2mm = downsample_labels(labels_full, 2)  # ~104x128x128
    print(f"  4mm grid: {labels_4mm.shape} ({labels_4mm.size:,} voxels)")
    print(f"  2mm grid: {labels_2mm.shape} ({labels_2mm.size:,} voxels)")
    print(f"  Downsampled in {time.time()-t0:.1f}s")
    print()

    # ==================================================================
    # Step 2: 3D Acoustic Property Mapping
    # ==================================================================
    print("=" * 70)
    print("STEP 2: Map 3D Acoustic Properties")
    print("=" * 70)

    props_4mm = acoustic.map_labels_to_properties(jnp.array(labels_4mm))
    props_2mm = acoustic.map_labels_to_properties(jnp.array(labels_2mm))

    for name, props, labels in [("4mm", props_4mm, labels_4mm), ("2mm", props_2mm, labels_2mm)]:
        n_skull = np.sum(np.isin(labels, [2, 3]))  # skull labels in SCI model
        print(f"  {name}: c=[{float(props['sound_speed'].min()):.0f}, {float(props['sound_speed'].max()):.0f}] m/s, "
              f"skull voxels={n_skull:,}")
    print()

    # ==================================================================
    # Step 3: 3D Simulation at 4mm (fast validation)
    # ==================================================================
    print("=" * 70)
    print("STEP 3: 3D Simulation at 4mm")
    print("=" * 70)

    from jwave.geometry import TimeAxis, Sources
    from jwave import FourierSeries
    from jaxdf.geometry import Domain

    shape_4mm = labels_4mm.shape
    spacing_4mm = 4e-3  # 4mm in meters

    c_4mm = np.array(props_4mm["sound_speed"])
    rho_4mm = np.array(props_4mm["density"])

    # Find brain target (interior, c~1560)
    brain_mask_4mm = (c_4mm > 1500) & (c_4mm < 1600)
    brain_coords_4mm = np.argwhere(brain_mask_4mm)
    target_4mm = tuple(brain_coords_4mm[len(brain_coords_4mm) // 2])
    print(f"  Target (brain center): {target_4mm}")

    # Find scalp source position
    scalp_mask_4mm = (c_4mm > 1600) & (c_4mm < 1700)
    scalp_coords_4mm = np.argwhere(scalp_mask_4mm)
    if len(scalp_coords_4mm) > 0:
        source_4mm = tuple(scalp_coords_4mm[0])
    else:
        source_4mm = (shape_4mm[0] // 2, shape_4mm[1] // 2, 2)
    print(f"  Source (scalp): {source_4mm}")

    # 3D Water simulation
    print("  Running 3D water simulation...")
    t0 = time.time()
    domain_4mm = jwa.create_domain(shape_4mm, spacing_4mm)
    medium_water = jwa.create_medium(domain_4mm, 1500.0, 1000.0, pml_size=5)
    ta_4mm = TimeAxis.from_medium(medium_water, cfl=0.3, t_end=1e-4)
    dt_4mm = float(ta_4mm.dt)

    pos_src = jnp.array([[source_4mm[i] * spacing_4mm for i in range(3)]])
    sources_4mm = jwa.create_sources(domain_4mm, pos_src, 400e3, jnp.zeros(1), jnp.ones(1),
                                      dt=dt_4mm, n_cycles=5)

    p_water_3d = jwa.run_simulation_jax(medium_water, source_4mm, 400e3, 1e-4,
                                         time_axis=ta_4mm, sources=sources_4mm)
    t_water_3d = time.time() - t0
    p_w_target = float(p_water_3d[target_4mm])
    p_w_max = float(jnp.max(p_water_3d))
    print(f"  Water: {t_water_3d:.1f}s, p_target={p_w_target:.6f}, p_max={p_w_max:.6f}")

    # 3D Skull simulation
    print("  Running 3D skull simulation...")
    t0 = time.time()
    medium_skull = jwa.create_medium(
        domain_4mm,
        jnp.array(c_4mm, dtype=jnp.float32),
        jnp.array(rho_4mm, dtype=jnp.float32),
        pml_size=5,
    )
    p_skull_3d = jwa.run_simulation_jax(medium_skull, source_4mm, 400e3, 1e-4,
                                         time_axis=ta_4mm, sources=sources_4mm)
    t_skull_3d = time.time() - t0
    p_s_target = float(p_skull_3d[target_4mm])
    p_s_max = float(jnp.max(p_skull_3d))
    atten_3d = 100 * (1 - p_s_target / p_w_target) if p_w_target > 0 else 0
    print(f"  Skull: {t_skull_3d:.1f}s, p_target={p_s_target:.6f}, p_max={p_s_max:.6f}")
    print(f"  3D Attenuation: {atten_3d:.1f}%")

    # Focal spot analysis
    focal_water = np.unravel_index(np.argmax(np.array(p_water_3d)), shape_4mm)
    focal_skull = np.unravel_index(np.argmax(np.array(p_skull_3d)), shape_4mm)
    focal_shift_mm = np.linalg.norm(np.array(focal_water) - np.array(focal_skull)) * 4.0  # 4mm spacing
    print(f"  Focal spot (water): {focal_water}")
    print(f"  Focal spot (skull): {focal_skull}")
    print(f"  Focal shift: {focal_shift_mm:.1f}mm")

    all_results["3d_4mm"] = {
        "grid_shape": list(shape_4mm),
        "spacing_mm": 4.0,
        "p_water_target": p_w_target,
        "p_skull_target": p_s_target,
        "attenuation_pct": atten_3d,
        "focal_water": list(focal_water),
        "focal_skull": list(focal_skull),
        "focal_shift_mm": focal_shift_mm,
        "time_water_s": t_water_3d,
        "time_skull_s": t_skull_3d,
    }
    print()

    # ==================================================================
    # Step 4: 3D Multi-Element Array Optimization at 4mm
    # ==================================================================
    print("=" * 70)
    print("STEP 4: 3D 16-Element Array Optimization")
    print("=" * 70)

    import optax

    n_elements = 16
    # Circular array on the scalp near the source position
    array_center = tuple(source_4mm[i] * spacing_4mm for i in range(3))
    positions_3d = exp.create_circular_array(n_elements, 0.008, array_center)
    print(f"  Array center: {array_center}")
    print(f"  Elements: {n_elements}")

    initial_delays = jnp.zeros(n_elements)

    def loss_fn_3d(delays):
        sources = jwa.create_sources(
            domain_4mm, positions_3d, 400e3,
            delays=delays, apodizations=jnp.ones(n_elements),
            dt=dt_4mm, n_cycles=5,
        )
        p = jwa.run_simulation_jax(medium_skull, source_4mm, 400e3, 1e-4,
                                    time_axis=ta_4mm, sources=sources)
        return -p[target_4mm]

    optimizer = optax.adam(1e-7)
    opt_state = optimizer.init(initial_delays)
    delays = initial_delays
    loss_history_3d = []

    print("  Optimizing delays (20 iterations)...")
    t0 = time.time()
    for i in range(20):
        loss_val, grads = jax.value_and_grad(loss_fn_3d)(delays)
        updates, opt_state = optimizer.update(grads, opt_state)
        delays = optax.apply_updates(delays, updates)
        loss_history_3d.append(float(loss_val))
        if (i + 1) % 5 == 0:
            print(f"    Iter {i+1}: loss={loss_val:.6f}")
    t_opt_3d = time.time() - t0

    improvement_3d = abs(loss_history_3d[-1] / loss_history_3d[0]) if loss_history_3d[0] != 0 else 0
    print(f"  Time: {t_opt_3d:.1f}s ({t_opt_3d/20:.1f}s/iter)")
    print(f"  Loss: {loss_history_3d[0]:.6f} -> {loss_history_3d[-1]:.6f}")
    print(f"  Improvement: {improvement_3d:.2f}x")
    print(f"  Delays (ns): {[f'{d*1e9:.0f}' for d in np.array(delays)]}")

    all_results["3d_4mm_optimization"] = {
        "n_elements": n_elements,
        "loss_history": loss_history_3d,
        "improvement": improvement_3d,
        "time_s": t_opt_3d,
        "time_per_iter": t_opt_3d / 20,
        "optimized_delays_ns": [float(d * 1e9) for d in np.array(delays)],
    }
    print()

    # ==================================================================
    # Step 5: 3D Simulation at 2mm (higher resolution validation)
    # ==================================================================
    print("=" * 70)
    print("STEP 5: 3D Simulation at 2mm (higher resolution)")
    print("=" * 70)

    shape_2mm = labels_2mm.shape
    spacing_2mm = 2e-3
    c_2mm = np.array(props_2mm["sound_speed"])
    rho_2mm = np.array(props_2mm["density"])

    brain_mask_2mm = (c_2mm > 1500) & (c_2mm < 1600)
    brain_coords_2mm = np.argwhere(brain_mask_2mm)
    target_2mm = tuple(brain_coords_2mm[len(brain_coords_2mm) // 2])

    scalp_mask_2mm = (c_2mm > 1600) & (c_2mm < 1700)
    scalp_coords_2mm = np.argwhere(scalp_mask_2mm)
    source_2mm = tuple(scalp_coords_2mm[0]) if len(scalp_coords_2mm) > 0 else (shape_2mm[0]//2, shape_2mm[1]//2, 2)

    print(f"  Grid: {shape_2mm} ({np.prod(shape_2mm):,} voxels)")
    print(f"  Target: {target_2mm}, Source: {source_2mm}")

    domain_2mm = jwa.create_domain(shape_2mm, spacing_2mm)

    # Water
    print("  Running 3D water simulation (2mm)...")
    t0 = time.time()
    med_w_2mm = jwa.create_medium(domain_2mm, 1500.0, 1000.0, pml_size=5)
    ta_2mm = TimeAxis.from_medium(med_w_2mm, cfl=0.3, t_end=1e-4)
    dt_2mm = float(ta_2mm.dt)
    pos_2mm = jnp.array([[source_2mm[i] * spacing_2mm for i in range(3)]])
    src_2mm = jwa.create_sources(domain_2mm, pos_2mm, 400e3, jnp.zeros(1), jnp.ones(1), dt=dt_2mm, n_cycles=5)
    p_w_2mm = jwa.run_simulation_jax(med_w_2mm, source_2mm, 400e3, 1e-4, time_axis=ta_2mm, sources=src_2mm)
    t_w_2mm = time.time() - t0
    print(f"  Water: {t_w_2mm:.1f}s, p_target={float(p_w_2mm[target_2mm]):.6f}")

    # Skull
    print("  Running 3D skull simulation (2mm)...")
    t0 = time.time()
    med_s_2mm = jwa.create_medium(
        domain_2mm,
        jnp.array(c_2mm, dtype=jnp.float32),
        jnp.array(rho_2mm, dtype=jnp.float32),
        pml_size=5,
    )
    p_s_2mm = jwa.run_simulation_jax(med_s_2mm, source_2mm, 400e3, 1e-4, time_axis=ta_2mm, sources=src_2mm)
    t_s_2mm = time.time() - t0

    p_w_t_2mm = float(p_w_2mm[target_2mm])
    p_s_t_2mm = float(p_s_2mm[target_2mm])
    atten_2mm = 100 * (1 - p_s_t_2mm / p_w_t_2mm) if p_w_t_2mm > 0 else 0

    focal_w_2mm = np.unravel_index(np.argmax(np.array(p_w_2mm)), shape_2mm)
    focal_s_2mm = np.unravel_index(np.argmax(np.array(p_s_2mm)), shape_2mm)
    shift_2mm = np.linalg.norm(np.array(focal_w_2mm) - np.array(focal_s_2mm)) * 2.0

    print(f"  Skull: {t_s_2mm:.1f}s, p_target={p_s_t_2mm:.6f}")
    print(f"  Attenuation: {atten_2mm:.1f}%")
    print(f"  Focal shift: {shift_2mm:.1f}mm")

    all_results["3d_2mm"] = {
        "grid_shape": list(shape_2mm),
        "spacing_mm": 2.0,
        "p_water_target": p_w_t_2mm,
        "p_skull_target": p_s_t_2mm,
        "attenuation_pct": atten_2mm,
        "focal_water": list(focal_w_2mm),
        "focal_skull": list(focal_s_2mm),
        "focal_shift_mm": shift_2mm,
        "time_water_s": t_w_2mm,
        "time_skull_s": t_s_2mm,
    }
    print()

    # ==================================================================
    # Step 6: Cross-Resolution Validation (4mm vs 2mm)
    # ==================================================================
    print("=" * 70)
    print("STEP 6: Cross-Resolution Validation")
    print("=" * 70)
    print(f"  4mm attenuation: {atten_3d:.1f}%")
    print(f"  2mm attenuation: {atten_2mm:.1f}%")
    print(f"  4mm focal shift: {focal_shift_mm:.1f}mm")
    print(f"  2mm focal shift: {shift_2mm:.1f}mm")
    print(f"  4mm sim time: {t_water_3d+t_skull_3d:.1f}s")
    print(f"  2mm sim time: {t_w_2mm+t_s_2mm:.1f}s")

    all_results["cross_resolution"] = {
        "attenuation_4mm": atten_3d,
        "attenuation_2mm": atten_2mm,
        "focal_shift_4mm_mm": focal_shift_mm,
        "focal_shift_2mm_mm": shift_2mm,
        "total_time_4mm_s": t_water_3d + t_skull_3d,
        "total_time_2mm_s": t_w_2mm + t_s_2mm,
    }
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 70)
    print("3D VALIDATION COMPLETE")
    print("=" * 70)
    print(f"  4mm grid: {shape_4mm}, attenuation={atten_3d:.1f}%, shift={focal_shift_mm:.1f}mm")
    print(f"  2mm grid: {shape_2mm}, attenuation={atten_2mm:.1f}%, shift={shift_2mm:.1f}mm")
    print(f"  16-element 3D optimization: {improvement_3d:.2f}x improvement, {t_opt_3d/20:.1f}s/iter")
    print()

    return all_results


@app.local_entrypoint()
def main():
    import json
    results = run_3d_validation.remote()
    print("\n" + "=" * 70)
    print("3D VALIDATION RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
