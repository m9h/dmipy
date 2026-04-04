"""
Modal.com deployment for TUS SCI head model pipeline.

Run with:
    pip install modal
    modal setup          # one-time auth
    modal run scripts/modal_tus_pipeline.py

This spins up an A100 GPU, installs deps, downloads the SCI head data
from a Modal Volume, runs the full pipeline, and shuts down.
"""

import modal

# --- Modal App ---
app = modal.App("tus-sci-head-pipeline")

# --- Container Image ---
# Downloads SCI head model (~3.5GB) during build so it's cached across runs.
# CC-BY 4.0 license: Warner, Tate, Burton, Johnson (2019). bioRxiv 10.1101/552190
# Download the pre-computed segmentation volume (much smaller than the 3.5GB mesh)
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
        "git clone --depth 1 https://github.com/m9h/sbi4dwi.git /opt/sbi4dwi",
        "echo '/opt/sbi4dwi' > /usr/local/lib/python3.12/site-packages/sbi4dwi.pth",
    )
    .run_commands(
        f"mkdir -p /sci_head && wget -q -O /sci_head/Segmentation.zip {SCI_SEG_URL}",
        "cd /sci_head && unzip -q Segmentation.zip && find /sci_head -type f | head -20",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,  # 60 min max
)
def run_tus_pipeline():
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp

    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print()

    # Direct file imports to avoid dmipy_jax.__init__ heavy dependency chain
    import importlib.util, sys

    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    SBI = "/opt/sbi4dwi/dmipy_jax"
    acoustic = _load_module("acoustic", f"{SBI}/biophysics/acoustic.py")
    jwa = _load_module("jwave_adapter", f"{SBI}/biophysics/jwave_adapter.py")
    optimizer = _load_module("tus_optimizer_mod", f"{SBI}/biophysics/tus_optimizer.py")

    # ---------------------------------------------------------------
    # Step 1: Load SCI Segmentation Volume (pre-rasterized)
    # ---------------------------------------------------------------
    print("=== Step 1: Load SCI Segmentation Volume ===")
    import glob, os, nibabel

    # Find segmentation files
    seg_files = glob.glob("/sci_head/**/*.nrrd", recursive=True) + \
                glob.glob("/sci_head/**/*.nii*", recursive=True) + \
                glob.glob("/sci_head/**/*.mat", recursive=True)
    print(f"  Segmentation files found:")
    for f in seg_files[:10]:
        print(f"    {f} ({os.path.getsize(f)/1e6:.1f} MB)")

    # Try NRRD first (SCI uses NRRD format)
    nrrd_files = [f for f in seg_files if f.endswith('.nrrd')]
    if nrrd_files:
        # Use nibabel or pynrrd to load
        try:
            import nrrd
            data, header = nrrd.read(nrrd_files[0])
            labels = np.array(data, dtype=np.int32)
            spacing = float(header.get('space directions', [[1,0,0],[0,1,0],[0,0,1]])[0][0])
            print(f"  Loaded NRRD: {nrrd_files[0]}")
        except ImportError:
            # Fallback: try nibabel
            img = nibabel.load(nrrd_files[0])
            labels = np.array(img.get_fdata(), dtype=np.int32)
            spacing = float(img.header.get_zooms()[0])
            print(f"  Loaded via nibabel: {nrrd_files[0]}")
    elif seg_files:
        # Try .mat files
        import scipy.io
        mat = scipy.io.loadmat(seg_files[0])
        print(f"  Mat keys: {[k for k in mat.keys() if not k.startswith('__')]}")
        # Find the segmentation array
        for key in mat:
            if not key.startswith('__') and hasattr(mat[key], 'shape'):
                if len(mat[key].shape) == 3 and mat[key].shape[0] > 10:
                    labels = np.array(mat[key], dtype=np.int32)
                    print(f"  Using key '{key}': shape {labels.shape}")
                    break
        spacing = 1.0  # default
    else:
        raise FileNotFoundError("No segmentation files found in /sci_head/")

    print(f"  Volume shape: {labels.shape}")
    print(f"  Spacing: {spacing} mm")
    unique_labels = np.unique(labels)
    print(f"  Tissue labels: {unique_labels} ({len(unique_labels)} types)")
    for u in unique_labels:
        count = np.sum(labels == u)
        print(f"    Label {u}: {count:,} voxels ({100*count/labels.size:.1f}%)")
    raster_time = 0.0  # Pre-computed
    grid_shape = labels.shape
    print()

    # ---------------------------------------------------------------
    # Step 3: Map Acoustic Properties
    # ---------------------------------------------------------------
    print("=== Step 3: Map Acoustic Properties ===")
    map_labels_to_properties = acoustic.map_labels_to_properties

    props = map_labels_to_properties(jnp.array(labels))
    for p in ["sound_speed", "density", "attenuation"]:
        arr = props[p]
        print(f"  {p}: [{float(arr.min()):.1f}, {float(arr.max()):.1f}]")
    print()

    # ---------------------------------------------------------------
    # Step 4: 2D Simulation (axial slice)
    # ---------------------------------------------------------------
    print("=== Step 4: j-Wave 2D Simulation ===")
    run_simulation_2d = jwa.run_simulation_2d

    mid_z = grid_shape[2] // 2
    c_slice = np.array(props["sound_speed"][:, :, mid_z])
    rho_slice = np.array(props["density"][:, :, mid_z])
    grid_spacing_m = spacing * 1e-3

    # Find brain target and scalp source
    brain_mask = (c_slice > 1500) & (c_slice < 1600)
    brain_coords = np.argwhere(brain_mask)
    target = tuple(brain_coords[len(brain_coords)//2]) if len(brain_coords) > 0 else (c_slice.shape[0]//2, c_slice.shape[1]//2)

    scalp_mask = (c_slice > 1600) & (c_slice < 1620)
    scalp_coords = np.argwhere(scalp_mask)
    source_pos = tuple(scalp_coords[0]) if len(scalp_coords) > 0 else (c_slice.shape[0]//2, 5)

    print(f"  Source: {source_pos}, Target: {target}")

    # Homogeneous
    t0 = time.time()
    r_water = run_simulation_2d(c_slice.shape, grid_spacing_m, 1500.0, 1000.0, source_pos, 400e3, 5e-5, pml_size=10)
    t_water = time.time() - t0
    p_water = float(r_water["p_max"][target])

    # Heterogeneous
    t0 = time.time()
    r_skull = run_simulation_2d(c_slice.shape, grid_spacing_m,
                                 jnp.array(c_slice, dtype=jnp.float32),
                                 jnp.array(rho_slice, dtype=jnp.float32),
                                 source_pos, 400e3, 5e-5, pml_size=10)
    t_skull = time.time() - t0
    p_skull = float(r_skull["p_max"][target])

    print(f"  Water: {t_water:.1f}s, p_target={p_water:.6f}")
    print(f"  Skull: {t_skull:.1f}s, p_target={p_skull:.6f}")
    if p_water > 0:
        print(f"  Attenuation: {100*(1-p_skull/p_water):.1f}%")
    print()

    # ---------------------------------------------------------------
    # Step 5: Delay Optimization
    # ---------------------------------------------------------------
    print("=== Step 5: Delay Optimization (10 iters) ===")
    optimize_delays = optimizer.optimize_delays
    create_domain = jwa.create_domain

    domain = create_domain(c_slice.shape, grid_spacing_m)
    if len(scalp_coords) > 1:
        element_positions = jnp.array([
            scalp_coords[0] * grid_spacing_m,
            scalp_coords[min(1, len(scalp_coords)-1)] * grid_spacing_m,
        ])
    else:
        element_positions = jnp.array([
            [source_pos[0]*grid_spacing_m, source_pos[1]*grid_spacing_m],
            [(source_pos[0]+2)*grid_spacing_m, source_pos[1]*grid_spacing_m],
        ])

    t0 = time.time()
    opt_delays, history = optimize_delays(
        domain,
        {"sound_speed": jnp.array(c_slice, dtype=jnp.float32),
         "density": jnp.array(rho_slice, dtype=jnp.float32)},
        element_positions, target, freq=400e3,
        n_iters=10, lr=1e-7, n_cycles=3, t_end=5e-5, pml_size=10,
    )
    t_opt = time.time() - t0
    print(f"  Time: {t_opt:.1f}s ({t_opt/10:.1f}s/iter)")
    print(f"  Loss: {history[0]:.6f} -> {history[-1]:.6f}")
    print(f"  Delays: {opt_delays}")
    print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Volume: {grid_shape} @ {spacing}mm")
    print(f"  Tissue types: {len(unique_labels)}")
    print(f"  Water sim: {t_water:.1f}s, p_target={p_water:.6f}")
    print(f"  Skull sim: {t_skull:.1f}s, p_target={p_skull:.6f}")
    if p_water > 0:
        print(f"  Skull attenuation: {100*(1-p_skull/p_water):.1f}%")
    print(f"  Optimization: {t_opt:.1f}s ({t_opt/10:.1f}s/iter)")
    print(f"  Focal pressure improvement: {abs(history[-1]/history[0]):.1f}x")

    return {
        "grid_shape": grid_shape,
        "tissue_types": len(unique_labels),
        "p_water": p_water,
        "p_skull": p_skull,
        "attenuation_pct": 100*(1-p_skull/p_water) if p_water > 0 else None,
        "optimized_delays": np.array(opt_delays).tolist(),
        "loss_history": history,
    }


@app.local_entrypoint()
def main():
    result = run_tus_pipeline.remote()
    print("\n=== Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
