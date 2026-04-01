"""
End-to-end integration test: SCI Head Model -> rasterize -> acoustic properties -> j-Wave simulation.

RED phase: Define the contract for the full TUS simulation pipeline on real anatomy.
Loads the SCI Institute head model, rasterizes to a regular grid, assigns ITRUSST
acoustic properties per tissue label, constructs a heterogeneous j-Wave medium,
places a transducer source on the scalp, and verifies that:
  1. The skull attenuates transmitted pressure compared to homogeneous water
  2. A focal spot forms at the intended target
  3. The pipeline is end-to-end differentiable (jax.grad w.r.t. element delays)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

SCI_MESH_PATH = "/data/mhough/dmipy-data/SCI_headmodel/extracted/HeadMesh.mat"
SCI_NRRD_PATH = "/data/mhough/dmipy-data/SCI_headmodel/Segmentation/HeadSegmentation.nrrd"


def _load_mesh_or_skip():
    """Load SCI head mesh or skip test if unavailable."""
    try:
        from dmipy_jax.io.sci_head_loader import load_sci_head_mesh
    except ImportError:
        pytest.skip("SCI head loader not available")
    try:
        mesh = load_sci_head_mesh(SCI_MESH_PATH)
    except FileNotFoundError:
        pytest.skip(f"SCI head data not found at {SCI_MESH_PATH}")
    return mesh


def _load_nrrd_or_skip():
    """Load the pre-rasterized SCI segmentation (NRRD) directly."""
    import gzip
    import os
    if not os.path.exists(SCI_NRRD_PATH):
        pytest.skip(f"SCI segmentation not found at {SCI_NRRD_PATH}")

    with open(SCI_NRRD_PATH, "rb") as f:
        # Skip header
        while True:
            line = f.readline()
            if line.strip() == b"":
                break
        compressed = f.read()
    data = np.frombuffer(gzip.decompress(compressed), dtype=np.uint8).reshape(208, 256, 256)

    # NRRD labels are 1-8. Remap to match SCI mesh convention:
    # NRRD 8 (68.6% = background) -> 0 (air)
    # NRRD 1-7 -> keep as-is (tissue compartments)
    labels = data.copy().astype(np.int32)
    labels[data == 8] = 0
    return labels


def _rasterize_sci_head(mesh, spacing_mm=3.0):
    """Rasterize SCI head mesh to regular grid at given spacing."""
    from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh

    points = np.array(mesh["points"])
    cells = np.array(mesh["cells"]["tetra"])
    tissues = np.array(mesh["cell_data"]["tissue"])

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    extent = bounds_max - bounds_min
    grid_shape = tuple((extent / spacing_mm).astype(int) + 1)

    labels = rasterize_mesh(
        points, cells, tissues,
        grid_shape=grid_shape,
        grid_spacing=spacing_mm,
        grid_origin=bounds_min,
    )
    return labels, grid_shape, spacing_mm, bounds_min


# ---------------------------------------------------------------------------
# Phase 1: Rasterization + Acoustic Property Assignment
# ---------------------------------------------------------------------------

class TestSCIHeadRasterizeAndMap:
    """Load SCI head segmentation and map tissue labels to acoustic properties."""

    @pytest.mark.slow
    def test_nrrd_segmentation_has_all_tissue_types(self):
        """NRRD segmentation should produce background + at least 5 tissue types."""
        labels = _load_nrrd_or_skip()

        unique = np.unique(labels)
        assert len(unique) >= 5, f"Only {len(unique)} tissue types: {unique}"
        assert 0 in unique, "Background label missing"

    @pytest.mark.slow
    def test_acoustic_property_volumes_from_sci_head(self):
        """Mapped acoustic properties should have correct shapes and physical ranges."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties

        labels = _load_nrrd_or_skip()

        props = map_labels_to_properties(jnp.array(labels))

        # Shape matches segmentation
        for key in ["sound_speed", "density", "attenuation"]:
            assert props[key].shape == labels.shape, (
                f"{key} shape {props[key].shape} != {labels.shape}"
            )

        # Physical ranges (air has c=344, so lower bound is low)
        assert jnp.all(props["sound_speed"] >= 300), "Sound speed below 300 m/s"
        assert jnp.all(props["sound_speed"] <= 5000), "Sound speed above 5000 m/s"
        assert jnp.all(props["density"] >= 1), "Density below 1 kg/m3"
        assert jnp.all(props["density"] <= 2500), "Density above 2500 kg/m3"
        assert jnp.all(props["attenuation"] >= 0), "Negative attenuation"

    @pytest.mark.slow
    def test_skull_region_has_high_sound_speed(self):
        """Voxels labeled as compact bone (3) should have sound speed > 3000 m/s."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties

        labels = _load_nrrd_or_skip()

        props = map_labels_to_properties(jnp.array(labels))
        skull_mask = labels == 3  # compact bone
        if skull_mask.sum() == 0:
            pytest.skip("No compact bone voxels in segmentation")

        skull_speed = props["sound_speed"][jnp.array(skull_mask)]
        assert jnp.all(skull_speed > 3000), (
            f"Skull sound speed min={float(skull_speed.min()):.0f} m/s, expected >3000"
        )


# ---------------------------------------------------------------------------
# Phase 2: j-Wave Simulation on SCI Head
# ---------------------------------------------------------------------------

class TestSCIHeadJWaveSimulation:
    """Run j-Wave acoustic simulation on the rasterized SCI head model."""

    @pytest.mark.slow
    def test_heterogeneous_head_simulation_runs(self):
        """j-Wave simulation on SCI head medium should produce non-zero pressure."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties
        from dmipy_jax.biophysics.jwave_adapter import (
            create_domain,
            create_medium,
            create_sources,
        )
        from jwave.acoustics.time_varying import simulate_wave_propagation
        from jwave.geometry import TimeAxis

        labels_3d = _load_nrrd_or_skip()

        # Downsample 2x for simulation speed: 256->128 per axis, 2mm spacing
        labels_ds = labels_3d[::2, ::2, ::2]
        spacing_mm = 2.0

        # Take a central sagittal slice (2D)
        mid_x = labels_ds.shape[0] // 2
        labels_2d = labels_ds[mid_x, :, :]
        ny, nz = labels_2d.shape

        props = map_labels_to_properties(jnp.array(labels_2d))

        # Convert spacing to metres for j-Wave
        spacing_m = spacing_mm * 1e-3
        domain = create_domain((ny, nz), spacing_m)

        medium = create_medium(
            domain,
            sound_speed=props["sound_speed"],
            density=props["density"],
            pml_size=10,
        )

        time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=50e-6)
        dt = float(time_axis.dt)

        # Place source on the scalp (top of head — find first non-background row)
        for row in range(ny):
            if np.any(labels_2d[row, :] > 0):
                source_row = row + 2  # just inside scalp
                break
        source_col = nz // 2

        positions = jnp.array([[source_row * spacing_m, source_col * spacing_m]])
        sources = create_sources(
            domain, positions, freq=500e3,
            delays=jnp.zeros(1), apodizations=jnp.ones(1),
            dt=dt, n_cycles=5,
        )

        pressure = simulate_wave_propagation(medium, time_axis, sources=sources)

        # Extract pressure field
        from dmipy_jax.biophysics.jwave_adapter import _extract_pressure_timeseries
        p_data = _extract_pressure_timeseries(pressure)
        p_max = jnp.max(jnp.abs(p_data), axis=0)

        assert p_max.shape == (ny, nz), f"Pressure shape {p_max.shape} != ({ny}, {nz})"
        assert jnp.max(p_max) > 0, "Zero pressure everywhere"

    @pytest.mark.slow
    def test_skull_attenuates_vs_water(self):
        """Pressure behind skull should be lower than through water alone."""
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties
        from dmipy_jax.biophysics.jwave_adapter import (
            create_domain,
            create_medium,
            create_sources,
            _extract_pressure_timeseries,
        )
        from jwave.acoustics.time_varying import simulate_wave_propagation
        from jwave.geometry import TimeAxis

        labels_3d = _load_nrrd_or_skip()
        labels_ds = labels_3d[::2, ::2, ::2]
        spacing_mm = 2.0
        mid_x = labels_ds.shape[0] // 2
        labels_2d = labels_ds[mid_x, :, :]
        ny, nz = labels_2d.shape
        spacing_m = spacing_mm * 1e-3

        props = map_labels_to_properties(jnp.array(labels_2d))

        domain = create_domain((ny, nz), spacing_m)

        # Heterogeneous (real head)
        medium_head = create_medium(
            domain,
            sound_speed=props["sound_speed"],
            density=props["density"],
            pml_size=10,
        )

        # Homogeneous water reference
        medium_water = create_medium(
            domain,
            sound_speed=1500.0,
            density=1000.0,
            pml_size=10,
        )

        t_end = 50e-6
        time_axis_head = TimeAxis.from_medium(medium_head, cfl=0.3, t_end=t_end)
        time_axis_water = TimeAxis.from_medium(medium_water, cfl=0.3, t_end=t_end)

        # Source on the scalp
        for row in range(ny):
            if np.any(labels_2d[row, :] > 0):
                source_row = row + 2
                break
        source_col = nz // 2

        freq = 500e3
        positions = jnp.array([[source_row * spacing_m, source_col * spacing_m]])

        sources_head = create_sources(
            domain, positions, freq,
            delays=jnp.zeros(1), apodizations=jnp.ones(1),
            dt=float(time_axis_head.dt), n_cycles=5,
        )
        sources_water = create_sources(
            domain, positions, freq,
            delays=jnp.zeros(1), apodizations=jnp.ones(1),
            dt=float(time_axis_water.dt), n_cycles=5,
        )

        # Run both
        p_head = simulate_wave_propagation(medium_head, time_axis_head, sources=sources_head)
        p_water = simulate_wave_propagation(medium_water, time_axis_water, sources=sources_water)

        p_max_head = jnp.max(jnp.abs(_extract_pressure_timeseries(p_head)), axis=0)
        p_max_water = jnp.max(jnp.abs(_extract_pressure_timeseries(p_water)), axis=0)

        # Find brain region (labels 4 or 5 = GM/WM)
        brain_mask = (labels_2d == 5) | (labels_2d == 6) | (labels_2d == 7)
        if brain_mask.sum() == 0:
            pytest.skip("No brain voxels in this slice")

        brain_pressure_head = jnp.mean(p_max_head[jnp.array(brain_mask)])
        brain_pressure_water = jnp.mean(p_max_water[jnp.array(brain_mask)])

        assert brain_pressure_head < brain_pressure_water, (
            f"Head should attenuate: brain_head={float(brain_pressure_head):.4f}, "
            f"brain_water={float(brain_pressure_water):.4f}"
        )


# ---------------------------------------------------------------------------
# Phase 3: Differentiable Delay Optimization on SCI Head
# ---------------------------------------------------------------------------

class TestSCIHeadDifferentiable:
    """Verify jax.grad works through the full SCI head simulation pipeline."""

    @pytest.mark.slow
    def test_grad_wrt_delays_on_head(self):
        """
        jax.grad of focal pressure w.r.t. element delays must not NaN
        when simulating through heterogeneous SCI head medium.
        """
        from dmipy_jax.biophysics.acoustic import map_labels_to_properties
        from dmipy_jax.biophysics.jwave_adapter import (
            create_domain,
            create_medium,
            create_sources,
            run_simulation_jax,
        )
        from jwave.geometry import TimeAxis

        labels_3d = _load_nrrd_or_skip()

        # Downsample 2x for tractable grad computation (256->128 per axis)
        labels_3d_ds = labels_3d[::2, ::2, ::2]
        spacing_mm = 2.0  # effectively 2mm after downsampling
        mid_x = labels_3d_ds.shape[0] // 2
        labels_2d = labels_3d_ds[mid_x, :, :]
        ny, nz = labels_2d.shape
        spacing_m = spacing_mm * 1e-3

        props = map_labels_to_properties(jnp.array(labels_2d))
        domain = create_domain((ny, nz), spacing_m)

        # Build medium, time_axis, sources with CONCRETE values (outside grad)
        medium = create_medium(
            domain,
            sound_speed=props["sound_speed"],
            density=props["density"],
            pml_size=8,
        )
        t_end = 30e-6
        time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=t_end)
        dt = float(time_axis.dt)

        # 4-element linear array on the scalp
        for row in range(ny):
            if np.any(labels_2d[row, :] > 0):
                source_row = row + 2
                break

        n_elements = 4
        center_col = nz // 2
        element_cols = np.linspace(center_col - 3, center_col + 3, n_elements).astype(int)
        element_positions = jnp.array([
            [source_row * spacing_m, col * spacing_m] for col in element_cols
        ])

        # Target: center of brain
        brain_mask = (labels_2d == 5) | (labels_2d == 6) | (labels_2d == 7)
        if brain_mask.sum() == 0:
            pytest.skip("No brain voxels in this slice")
        brain_coords = np.argwhere(brain_mask)
        target_yz = brain_coords[len(brain_coords) // 2]  # median brain voxel
        target_voxel = (int(target_yz[0]), int(target_yz[1]))

        freq = 500e3
        n_cycles = 3
        initial_delays = jnp.zeros(n_elements)

        # Pre-build sources with zero delays for the reference
        ref_sources = create_sources(
            domain, element_positions, freq,
            delays=initial_delays,
            apodizations=jnp.ones(n_elements),
            dt=dt, n_cycles=n_cycles,
        )

        def focal_pressure(delays):
            sources = create_sources(
                domain, element_positions, freq,
                delays=delays,
                apodizations=jnp.ones(n_elements),
                dt=dt, n_cycles=n_cycles,
            )
            p_max = run_simulation_jax(
                medium,
                source_position=target_voxel,
                freq=freq,
                t_end=t_end,
                time_axis=time_axis,
                sources=sources,
            )
            return p_max[target_voxel]

        grad_fn = jax.grad(focal_pressure)
        g = grad_fn(initial_delays)

        assert g.shape == (n_elements,), f"Gradient shape {g.shape} != ({n_elements},)"
        assert not jnp.any(jnp.isnan(g)), f"Gradient contains NaN: {g}"
        # Gradient should not be uniformly zero (some elements closer to target)
        assert jnp.any(g != 0.0), f"Gradient is all zeros: {g}"
