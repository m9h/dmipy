"""
DGX Spark: SCI Head Model full TUS pipeline.

Loads SCI head → rasterizes → maps acoustic properties → runs j-Wave
simulation → optimizes delays through skull → exports Solution.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print()

# ---------------------------------------------------------------
# Step 1: Load SCI Head Model
# ---------------------------------------------------------------
print("=== Step 1: Load SCI Head Model ===")
from dmipy_jax.io.sci_head_loader import load_sci_head_mesh

t0 = time.time()
mesh = load_sci_head_mesh("data/SCI_headmodel/extracted/HeadMesh.mat")
points = np.array(mesh["points"])
cells = np.array(mesh["cells"]["tetra"])
tissues = np.array(mesh["cell_data"]["tissue"])
print(f"  Vertices: {points.shape[0]:,}")
print(f"  Tetrahedra: {cells.shape[0]:,}")
print(f"  Tissue labels: {np.unique(tissues)}")
print(f"  Bounds: {points.min(0)} to {points.max(0)}")
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
print(f"  Grid spacing: {spacing}mm")

t0 = time.time()
labels = rasterize_mesh(points, cells, tissues, grid_shape, spacing, bounds_min)
raster_time = time.time() - t0
unique_labels = np.unique(labels)
print(f"  Labels found: {unique_labels} ({len(unique_labels)} types)")
for u in unique_labels:
    count = np.sum(labels == u)
    pct = 100.0 * count / labels.size
    print(f"    Label {u}: {count:,} voxels ({pct:.1f}%)")
print(f"  Rasterized in {raster_time:.1f}s")
print()

# ---------------------------------------------------------------
# Step 3: Map Acoustic Properties
# ---------------------------------------------------------------
print("=== Step 3: Map Acoustic Properties ===")
from dmipy_jax.biophysics.acoustic import map_labels_to_properties

props = map_labels_to_properties(jnp.array(labels))
for param in ["sound_speed", "density", "attenuation"]:
    arr = props[param]
    print(f"  {param}: min={float(arr.min()):.1f}, max={float(arr.max()):.1f}, "
          f"mean={float(arr.mean()):.1f}")
print()

# ---------------------------------------------------------------
# Step 4: Extract 2D Axial Slice for Simulation
# ---------------------------------------------------------------
print("=== Step 4: Extract Axial Slice for 2D Simulation ===")
mid_z = grid_shape[2] // 2
c_slice = np.array(props["sound_speed"][:, :, mid_z])
rho_slice = np.array(props["density"][:, :, mid_z])
print(f"  Slice at z={mid_z}, shape: {c_slice.shape}")
print(f"  Sound speed range in slice: {c_slice.min():.0f}-{c_slice.max():.0f} m/s")

# Find a brain target (interior, where c ~ 1560)
brain_mask = (c_slice > 1500) & (c_slice < 1600)
brain_coords = np.argwhere(brain_mask)
if len(brain_coords) > 0:
    target = tuple(brain_coords[len(brain_coords) // 2])
    print(f"  Target (brain): {target}")
else:
    target = (c_slice.shape[0] // 2, c_slice.shape[1] // 2)
    print(f"  Target (center, fallback): {target}")

# Find a scalp source position (exterior, where c ~ 1610)
scalp_mask = (c_slice > 1600) & (c_slice < 1620)
scalp_coords = np.argwhere(scalp_mask)
if len(scalp_coords) > 0:
    source_pos = tuple(scalp_coords[0])
    print(f"  Source (scalp): {source_pos}")
else:
    source_pos = (c_slice.shape[0] // 2, 5)
    print(f"  Source (edge, fallback): {source_pos}")
print()

# ---------------------------------------------------------------
# Step 5: j-Wave 2D Simulation (Homogeneous vs Skull-Corrected)
# ---------------------------------------------------------------
print("=== Step 5: j-Wave 2D Simulation ===")
from dmipy_jax.biophysics.jwave_adapter import run_simulation_2d

grid_spacing_m = spacing * 1e-3  # convert mm to meters

# 5a: Homogeneous water (baseline)
print("  Running homogeneous water simulation...")
t0 = time.time()
result_water = run_simulation_2d(
    grid_shape=c_slice.shape,
    grid_spacing=grid_spacing_m,
    sound_speed=1500.0,
    density=1000.0,
    source_position=source_pos,
    freq=400e3,
    t_end=5e-5,
    pml_size=10,
)
t_water = time.time() - t0
p_water_target = float(result_water["p_max"][target])
p_water_max = float(jnp.max(result_water["p_max"]))
print(f"    Time: {t_water:.1f}s")
print(f"    Max pressure: {p_water_max:.6f} Pa")
print(f"    Pressure at target: {p_water_target:.6f} Pa")

# 5b: Heterogeneous (real skull)
print("  Running heterogeneous skull simulation...")
t0 = time.time()
result_skull = run_simulation_2d(
    grid_shape=c_slice.shape,
    grid_spacing=grid_spacing_m,
    sound_speed=jnp.array(c_slice, dtype=jnp.float32),
    density=jnp.array(rho_slice, dtype=jnp.float32),
    source_position=source_pos,
    freq=400e3,
    t_end=5e-5,
    pml_size=10,
)
t_skull = time.time() - t0
p_skull_target = float(result_skull["p_max"][target])
p_skull_max = float(jnp.max(result_skull["p_max"]))
print(f"    Time: {t_skull:.1f}s")
print(f"    Max pressure: {p_skull_max:.6f} Pa")
print(f"    Pressure at target: {p_skull_target:.6f} Pa")

if p_water_target > 0:
    attenuation_pct = 100.0 * (1.0 - p_skull_target / p_water_target)
    print(f"  Skull attenuation at target: {attenuation_pct:.1f}%")
print()

# ---------------------------------------------------------------
# Step 6: Gradient-Based Delay Optimization
# ---------------------------------------------------------------
print("=== Step 6: Delay Optimization (10 iterations) ===")
from dmipy_jax.biophysics.tus_optimizer import optimize_delays
from dmipy_jax.biophysics.jwave_adapter import create_domain

domain = create_domain(c_slice.shape, grid_spacing_m)

# 2 source elements near scalp
if len(scalp_coords) > 1:
    pos1 = scalp_coords[0] * grid_spacing_m
    pos2 = scalp_coords[min(1, len(scalp_coords)-1)] * grid_spacing_m
    element_positions = jnp.array([pos1, pos2])
else:
    element_positions = jnp.array([
        [source_pos[0] * grid_spacing_m, source_pos[1] * grid_spacing_m],
        [(source_pos[0]+2) * grid_spacing_m, source_pos[1] * grid_spacing_m],
    ])

print(f"  Elements: {element_positions}")
print(f"  Target: {target}")

t0 = time.time()
optimized_delays, history = optimize_delays(
    domain,
    {"sound_speed": jnp.array(c_slice, dtype=jnp.float32),
     "density": jnp.array(rho_slice, dtype=jnp.float32)},
    element_positions,
    target,
    freq=400e3,
    n_iters=10,
    lr=1e-7,
    n_cycles=3,
    t_end=5e-5,
    pml_size=10,
)
t_opt = time.time() - t0
print(f"  Time: {t_opt:.1f}s ({t_opt/10:.1f}s/iter)")
print(f"  Loss history: {[f'{h:.6f}' for h in history]}")
print(f"  Optimized delays: {optimized_delays}")
if len(history) >= 2:
    improvement = history[0] - history[-1]
    print(f"  Loss improvement: {improvement:.6f} ({100*improvement/abs(history[0]):.1f}%)")
print()

# ---------------------------------------------------------------
# Step 7: Export Solution
# ---------------------------------------------------------------
print("=== Step 7: Export Solution ===")
from dmipy_jax.biophysics.tus_solution_export import export_simulation_result, build_solution_dict

coords = {
    "x": np.arange(c_slice.shape[0]) * spacing,
    "y": np.arange(c_slice.shape[1]) * spacing,
}
ds = export_simulation_result(
    np.array(result_skull["p_max"]),
    coords,
    density=rho_slice,
    sound_speed=c_slice,
)
sol = build_solution_dict(
    delays=np.array(optimized_delays)[np.newaxis, :],
    apodizations=np.ones((1, 2)),
    simulation_result=ds,
    freq=400e3,
)
print(f"  Dataset vars: {list(ds.data_vars)}")
print(f"  Solution keys: {list(sol.keys())}")
print(f"  Delays: {sol['delays']}")
print()

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  SCI head: {points.shape[0]:,} vertices, {cells.shape[0]:,} tets")
print(f"  Grid: {grid_shape} at {spacing}mm ({labels.size:,} voxels)")
print(f"  Tissue types: {len(unique_labels)}")
print(f"  Rasterization: {raster_time:.1f}s")
print(f"  Water sim: {t_water:.1f}s, p_target={p_water_target:.6f}")
print(f"  Skull sim: {t_skull:.1f}s, p_target={p_skull_target:.6f}")
if p_water_target > 0:
    print(f"  Skull attenuation: {attenuation_pct:.1f}%")
print(f"  Optimization: {t_opt:.1f}s for 10 iterations")
print(f"  Optimized delays: {np.array(optimized_delays)}")
print()
print("PIPELINE COMPLETE")
