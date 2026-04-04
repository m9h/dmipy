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
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "jaxlib",
        "jwave",
        "optax",
        "equinox",
        "scipy",
        "numpy<2",
        "xarray",
        "h5py",
    )
    .run_commands(
        "pip install git+https://github.com/m9h/sbi4dwi.git@master",
    )
)

# --- Volume for SCI head data ---
# Create once: modal volume create sci-head-data
# Upload once: modal volume put sci-head-data /local/path/to/HeadMesh.mat /HeadMesh.mat
vol = modal.Volume.from_name("sci-head-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,  # 30 min max
    volumes={"/data": vol},
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

    # ---------------------------------------------------------------
    # Step 1: Load SCI Head Model
    # ---------------------------------------------------------------
    print("=== Step 1: Load SCI Head Model ===")
    from dmipy_jax.io.sci_head_loader import load_sci_head_mesh

    t0 = time.time()
    mesh = load_sci_head_mesh("/data/HeadMesh.mat")
    points = np.array(mesh["points"])
    cells = np.array(mesh["cells"]["tetra"])
    tissues = np.array(mesh["cell_data"]["tissue"])
    print(f"  Vertices: {points.shape[0]:,}")
    print(f"  Tetrahedra: {cells.shape[0]:,}")
    print(f"  Tissue labels: {np.unique(tissues)}")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    # ---------------------------------------------------------------
    # Step 2: Rasterize to Regular Grid
    # ---------------------------------------------------------------
    print("=== Step 2: Rasterize Mesh (2mm grid) ===")
    from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    spacing = 2.0  # mm
    grid_shape = tuple(((bounds_max - bounds_min) / spacing).astype(int) + 1)
    print(f"  Grid shape: {grid_shape}")

    t0 = time.time()
    labels = rasterize_mesh(points, cells, tissues, grid_shape, spacing, bounds_min)
    raster_time = time.time() - t0
    unique_labels = np.unique(labels)
    print(f"  Labels: {unique_labels} ({len(unique_labels)} types)")
    for u in unique_labels:
        count = np.sum(labels == u)
        print(f"    Label {u}: {count:,} voxels ({100*count/labels.size:.1f}%)")
    print(f"  Rasterized in {raster_time:.1f}s")
    print()

    # ---------------------------------------------------------------
    # Step 3: Map Acoustic Properties
    # ---------------------------------------------------------------
    print("=== Step 3: Map Acoustic Properties ===")
    from dmipy_jax.biophysics.acoustic import map_labels_to_properties

    props = map_labels_to_properties(jnp.array(labels))
    for p in ["sound_speed", "density", "attenuation"]:
        arr = props[p]
        print(f"  {p}: [{float(arr.min()):.1f}, {float(arr.max()):.1f}]")
    print()

    # ---------------------------------------------------------------
    # Step 4: 2D Simulation (axial slice)
    # ---------------------------------------------------------------
    print("=== Step 4: j-Wave 2D Simulation ===")
    from dmipy_jax.biophysics.jwave_adapter import run_simulation_2d

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
    from dmipy_jax.biophysics.tus_optimizer import optimize_delays
    from dmipy_jax.biophysics.jwave_adapter import create_domain

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
    print(f"  Head: {points.shape[0]:,} verts, {cells.shape[0]:,} tets")
    print(f"  Grid: {grid_shape} @ {spacing}mm")
    print(f"  Rasterize: {raster_time:.1f}s")
    print(f"  Water sim: {t_water:.1f}s")
    print(f"  Skull sim: {t_skull:.1f}s")
    print(f"  Optimization: {t_opt:.1f}s")

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
