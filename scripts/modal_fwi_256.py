"""
Modal.com: 256-element FWI simulation on SimNIBS Ernie head model.

Downloads the free Ernie head model, places a 256-element semi-ellipsoidal
array on the scalp surface (Oxford-UCL helmet geometry), and runs:
1. Forward acoustic simulation through heterogeneous skull
2. Full waveform inversion to recover sound speed from pressure data
3. Radiation force + displacement mapping
4. MR-ARFI phase prediction

    modal run scripts/modal_fwi_256.py
"""

import modal

app = modal.App("fwi-256-ernie")

ERNIE_URL = "https://github.com/simnibs/example-dataset/releases/latest/download/simnibs4_examples.zip"

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
        "scico==0.0.6",
    )
    .env({"SBI4DWI_COMMIT": "d6db016"})
    .run_commands(
        "rm -rf /opt/sbi4dwi"
        " && git clone --depth 1 https://github.com/m9h/sbi4dwi.git /opt/sbi4dwi"
        " && cd /opt/sbi4dwi && git log --oneline -1",
        "echo '/opt/sbi4dwi' > /usr/local/lib/python3.12/site-packages/sbi4dwi.pth",
    )
    .run_commands(
        f"mkdir -p /ernie && wget -q -O /ernie/simnibs4_examples.zip {ERNIE_URL}",
        "cd /ernie && unzip -q simnibs4_examples.zip"
        " && find /ernie -name '*.nii*' -o -name '*.msh' | head -20",
    )
)


@app.function(image=image, gpu="A100", timeout=3600, memory=32768)
def run_fwi_256():
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import importlib.util, sys, os, glob

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    SBI = "/opt/sbi4dwi/dmipy_jax"
    acoustic = _load("acoustic", f"{SBI}/biophysics/acoustic.py")
    jwa = _load("jwave_adapter", f"{SBI}/biophysics/jwave_adapter.py")
    rf = _load("radiation_force", f"{SBI}/biophysics/radiation_force.py")
    arfi = _load("mr_arfi", f"{SBI}/biophysics/mr_arfi.py")
    exp = _load("tus_experiments", f"{SBI}/biophysics/tus_experiments.py")

    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print()

    all_results = {}

    # ==================================================================
    # Step 1: Load Ernie head model
    # ==================================================================
    print("=" * 70)
    print("STEP 1: Load SimNIBS Ernie Head Model")
    print("=" * 70)

    # Find the segmentation volume
    nii_files = sorted(glob.glob("/ernie/**/T1*.nii*", recursive=True))
    seg_files = sorted(glob.glob("/ernie/**/*final*seg*.nii*", recursive=True)
                       + glob.glob("/ernie/**/*tissue*.nii*", recursive=True)
                       + glob.glob("/ernie/**/*labels*.nii*", recursive=True)
                       + glob.glob("/ernie/**/*mask*.nii*", recursive=True))

    print(f"  NIfTI files found: {len(nii_files)}")
    for f in nii_files[:5]:
        print(f"    {f}")
    print(f"  Segmentation files: {len(seg_files)}")
    for f in seg_files[:5]:
        print(f"    {f}")

    # Also list all files in ernie for debugging
    all_files = sorted(glob.glob("/ernie/**/*", recursive=True))
    print(f"  Total files: {len(all_files)}")
    for f in all_files[:30]:
        sz = os.path.getsize(f) if os.path.isfile(f) else 0
        print(f"    {f} ({sz/1e6:.1f} MB)")

    # Try to load a segmentation
    import nibabel as nib

    seg_loaded = False
    for candidate in seg_files + nii_files:
        try:
            img = nib.load(candidate)
            data = img.get_fdata()
            unique = np.unique(data.astype(int))
            if len(unique) >= 3:  # need at least 3 tissue types
                labels_full = data.astype(np.int32)
                voxel_size = img.header.get_zooms()
                print(f"  Loaded: {candidate}")
                print(f"  Shape: {labels_full.shape}, Voxels: {voxel_size}")
                print(f"  Labels: {unique}")
                seg_loaded = True
                break
        except Exception as e:
            continue

    if not seg_loaded:
        # Create a synthetic head phantom with skull shell
        print("  No segmentation found — creating synthetic head phantom")
        N = 128
        labels_full = np.zeros((N, N, N), dtype=np.int32)
        # Concentric shells: scalp, skull, CSF, brain
        cx, cy, cz = N//2, N//2, N//2
        xx, yy, zz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
        r = np.sqrt((xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2)
        labels_full[r < 55] = 1   # scalp
        labels_full[r < 52] = 2   # skull (compact bone → label 7 in SimNIBS, map to 2)
        labels_full[r < 46] = 3   # CSF
        labels_full[r < 43] = 4   # gray matter
        labels_full[r < 35] = 5   # white matter
        voxel_size = (1.0, 1.0, 1.0)
        print(f"  Synthetic phantom: {labels_full.shape}, labels: {np.unique(labels_full)}")

    # Downsample to 2mm for tractable simulation
    factor = max(1, int(round(2.0 / voxel_size[0])))
    if factor > 1:
        from scipy.ndimage import zoom
        labels_2mm = zoom(labels_full, 1.0/factor, order=0).astype(np.int32)
    else:
        labels_2mm = labels_full
    spacing_mm = voxel_size[0] * factor
    spacing_m = spacing_mm * 1e-3
    print(f"  Downsampled: {labels_2mm.shape} at {spacing_mm:.1f}mm")
    print()

    # ==================================================================
    # Step 2: Generate 256-element semi-ellipsoidal array
    # ==================================================================
    print("=" * 70)
    print("STEP 2: Generate 256-Element Helmet Array")
    print("=" * 70)

    def fibonacci_hemisphere(n, semi_axes=(0.103, 0.079, 0.048), center=None):
        """
        Place n points on a semi-ellipsoid using Fibonacci spiral.
        Semi-axes from Oxford-UCL helmet: 206/2=103mm, 157/2=79mm, 96mm height.
        """
        if center is None:
            center = np.array([labels_2mm.shape[0]//2,
                               labels_2mm.shape[1]//2,
                               labels_2mm.shape[2]//2]) * spacing_m

        positions = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - (i + 0.5) / n)  # polar angle [0, pi]
            phi = 2 * np.pi * i / golden_ratio     # azimuthal angle

            # Only upper hemisphere (theta < pi/2 covers dome)
            if theta > np.pi * 0.8:
                theta = np.pi * 0.8  # clip to avoid bottom

            x = semi_axes[0] * np.sin(theta) * np.cos(phi) + center[0]
            y = semi_axes[1] * np.sin(theta) * np.sin(phi) + center[1]
            z = semi_axes[2] * np.cos(theta) + center[2]
            positions.append([x, y, z])

        return np.array(positions)

    element_positions = fibonacci_hemisphere(256)
    print(f"  Elements: {element_positions.shape[0]}")
    print(f"  Position range X: [{element_positions[:,0].min()*1e3:.1f}, {element_positions[:,0].max()*1e3:.1f}] mm")
    print(f"  Position range Y: [{element_positions[:,1].min()*1e3:.1f}, {element_positions[:,1].max()*1e3:.1f}] mm")
    print(f"  Position range Z: [{element_positions[:,2].min()*1e3:.1f}, {element_positions[:,2].max()*1e3:.1f}] mm")
    print()

    # ==================================================================
    # Step 3: Map acoustic + mechanical properties
    # ==================================================================
    print("=" * 70)
    print("STEP 3: Map Acoustic + Mechanical Properties")
    print("=" * 70)

    # Take an axial slice for 2D simulation (faster on A100)
    mid_z = labels_2mm.shape[2] // 2
    labels_slice = labels_2mm[:, :, mid_z]
    print(f"  Axial slice at z={mid_z}, shape: {labels_slice.shape}")

    props = acoustic.map_labels_to_properties(jnp.array(labels_slice))
    mu_map = rf.map_labels_to_shear_modulus(jnp.array(labels_slice))

    print(f"  Sound speed: [{float(props['sound_speed'].min()):.0f}, {float(props['sound_speed'].max()):.0f}] m/s")
    print(f"  Density: [{float(props['density'].min()):.0f}, {float(props['density'].max()):.0f}] kg/m³")
    print(f"  Shear modulus: [{float(mu_map[mu_map>0].min()):.0f}, {float(mu_map.max()):.0f}] Pa")

    # Count tissue types
    for label in np.unique(labels_slice):
        count = np.sum(labels_slice == label)
        print(f"    Label {label}: {count:,} voxels ({100*count/labels_slice.size:.1f}%)")
    print()

    # ==================================================================
    # Step 4: Forward simulation (4 elements from the array on the slice)
    # ==================================================================
    print("=" * 70)
    print("STEP 4: Forward Simulation (2D slice, 4 elements)")
    print("=" * 70)

    from jwave.geometry import TimeAxis

    grid_shape = labels_slice.shape
    domain = jwa.create_domain(grid_shape, spacing_m)

    c = jnp.array(props["sound_speed"], dtype=jnp.float32)
    rho = jnp.array(props["density"], dtype=jnp.float32)
    medium = jwa.create_medium(domain, c, rho, pml_size=8)
    ta = TimeAxis.from_medium(medium, cfl=0.3, t_end=5e-5)
    dt = float(ta.dt)

    # Pick 4 element positions that fall on the 2D slice
    cx, cy = grid_shape[0]//2, grid_shape[1]//2
    source_positions_2d = jnp.array([
        [(cx - 20) * spacing_m, 3 * spacing_m],
        [(cx + 20) * spacing_m, 3 * spacing_m],
        [(cx - 10) * spacing_m, 3 * spacing_m],
        [(cx + 10) * spacing_m, 3 * spacing_m],
    ])

    # Find a brain target
    brain_mask = (labels_slice == 4) | (labels_slice == 5)
    brain_coords = np.argwhere(brain_mask)
    target = tuple(brain_coords[len(brain_coords)//2]) if len(brain_coords) > 0 else (cx, cy)
    print(f"  Target: {target}")
    print(f"  Sources: 4 elements at scalp")
    print(f"  Time step: {dt*1e9:.1f}ns")

    # Forward simulation
    t0 = time.time()
    sources = jwa.create_sources(domain, source_positions_2d, 555e3,
                                  jnp.zeros(4), jnp.ones(4), dt=dt, n_cycles=5)
    p_observed = jwa.run_simulation_jax(medium, target, 555e3, 5e-5,
                                         time_axis=ta, sources=sources)
    t_fwd = time.time() - t0
    print(f"  Forward sim: {t_fwd:.1f}s")
    print(f"  p_max at target: {float(p_observed[target]):.6f}")
    print(f"  p_max global: {float(jnp.max(p_observed)):.6f}")

    all_results["forward"] = {
        "grid_shape": list(grid_shape),
        "spacing_mm": spacing_mm,
        "n_elements": 4,
        "freq_hz": 555e3,
        "p_target": float(p_observed[target]),
        "p_max": float(jnp.max(p_observed)),
        "time_s": t_fwd,
    }
    print()

    # ==================================================================
    # Step 5: FWI — invert for sound speed (SCICO + j-Wave)
    # ==================================================================
    print("=" * 70)
    print("STEP 5: Full Waveform Inversion (SCICO PDHG + TV)")
    print("=" * 70)

    fwi = _load("fwi_us", f"{SBI}/biophysics/fwi_us.py")

    # Build AcousticForwardOperator (wraps j-Wave as SCICO-compatible operator)
    forward_op = fwi.AcousticForwardOperator(
        grid_shape=grid_shape,
        grid_spacing=spacing_m,
        density=1000.0,
        freq=555e3,
        source_position=tuple(int(x) for x in source_positions_2d[0] / spacing_m),
        t_end=5e-5,
        pml_size=8.0,
    )
    # Use the pre-computed multi-element sources
    forward_op.sources = sources

    # Try SCICO first, fall back to optax if SCICO not available
    t0 = time.time()
    try:
        print("  Method: SCICO AcceleratedPGM + IsotropicTVNorm")
        c_recon, fwi_info = fwi.invert_scico(
            p_observed, forward_op,
            initial_sound_speed=1500.0,
            tv_weight=0.1,
            n_iters=10,
        )
        fwi_history = [fwi_info.get("n_iters", 10)]
        print(f"  SCICO inversion complete")
    except Exception as e:
        print(f"  SCICO failed ({e}), falling back to optax Adam")
        c_recon, fwi_history = fwi.invert_optax(
            p_observed, forward_op,
            initial_sound_speed=1500.0,
            tv_weight=0.01,
            n_iters=10,
            lr=1.0,
        )
        for i, h in enumerate(fwi_history):
            if (i+1) % 5 == 0:
                print(f"    Iter {i+1}: loss={h:.8f}")

    t_fwi = time.time() - t0

    # Compare reconstruction to truth
    c_true = np.array(props["sound_speed"])
    c_rec = np.array(c_recon)
    skull_mask = labels_slice == 2
    brain_mask = (labels_slice == 4) | (labels_slice == 5)

    mse_total = float(np.mean((c_rec - c_true)**2))
    if skull_mask.sum() > 0:
        mse_skull = float(np.mean((c_rec[skull_mask] - c_true[skull_mask])**2))
    else:
        mse_skull = 0.0

    print(f"  FWI time: {t_fwi:.1f}s ({t_fwi/10:.1f}s/iter)")
    print(f"  Loss: {fwi_history[0]:.8f} → {fwi_history[-1]:.8f}")
    print(f"  MSE total: {mse_total:.2f}")
    print(f"  MSE skull: {mse_skull:.2f}")
    print(f"  Recon c range: [{float(c_recon.min()):.0f}, {float(c_recon.max()):.0f}]")

    all_results["fwi"] = {
        "n_iters": 10,
        "loss_history": fwi_history,
        "time_s": t_fwi,
        "mse_total": mse_total,
        "mse_skull": mse_skull,
        "c_range": [float(c_recon.min()), float(c_recon.max())],
    }
    print()

    # ==================================================================
    # Step 6: Radiation force + displacement + ARFI phase
    # ==================================================================
    print("=" * 70)
    print("STEP 6: Radiation Force → Displacement → ARFI Phase")
    print("=" * 70)

    # Intensity from pressure
    Z = props["density"] * props["sound_speed"]
    intensity = p_observed**2 / (2 * Z)

    # Radiation force
    force = rf.compute_radiation_force_from_simulation(
        intensity, props["sound_speed"], props["attenuation"]
    )

    # Displacement
    displacement = rf.solve_displacement_quasistatic(force, mu_map, spacing_m)

    # ARFI phase
    phase = arfi.predict_arfi_phase(np.array(displacement), 40e-3, 5e-3)

    max_force = float(jnp.max(jnp.abs(force)))
    max_disp_um = float(jnp.max(jnp.abs(displacement))) * 1e6
    max_phase = float(np.max(np.abs(phase)))

    print(f"  Max force: {max_force:.6e} N/m³")
    print(f"  Max displacement: {max_disp_um:.4f} µm")
    print(f"  Max ARFI phase: {max_phase:.6f} rad")
    print(f"  At target — force: {float(force[target]):.6e}, "
          f"disp: {float(displacement[target])*1e6:.4f} µm, "
          f"phase: {float(phase[target]):.6f} rad")

    # Regional displacement comparison
    if brain_mask.sum() > 0:
        gm_mask = labels_slice == 4
        wm_mask = labels_slice == 5
        if gm_mask.sum() > 0 and wm_mask.sum() > 0:
            gm_disp = float(jnp.mean(jnp.abs(displacement[jnp.array(gm_mask)])))
            wm_disp = float(jnp.mean(jnp.abs(displacement[jnp.array(wm_mask)])))
            ratio = wm_disp / gm_disp if gm_disp > 0 else 0
            print(f"  Mean GM displacement: {gm_disp*1e6:.6f} µm")
            print(f"  Mean WM displacement: {wm_disp*1e6:.6f} µm")
            print(f"  WM/GM ratio: {ratio:.2f}x")

    all_results["arfi"] = {
        "max_force": max_force,
        "max_displacement_um": max_disp_um,
        "max_phase_rad": max_phase,
    }
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Head model: {labels_2mm.shape} at {spacing_mm:.1f}mm")
    print(f"  Array: 256 elements (4 active on 2D slice)")
    print(f"  Frequency: 555 kHz")
    print(f"  Forward sim: {t_fwd:.1f}s")
    print(f"  FWI (10 iters): {t_fwi:.1f}s")
    print(f"  FWI loss: {fwi_history[0]:.8f} → {fwi_history[-1]:.8f}")

    return all_results


@app.local_entrypoint()
def main():
    import json
    results = run_fwi_256.remote()
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
