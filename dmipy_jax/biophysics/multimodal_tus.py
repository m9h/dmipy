"""
Multimodal MRI-TFUS-ARFI simulation pipeline.

Orchestrates the full chain: acoustic → radiation force → displacement → MR-ARFI.

    j-Wave acoustic sim → F=2αI/c → u=F/(µk²) → Δφ=γ·G·δ·u → MR-ARFI phase maps
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple


def run_multimodal_pipeline(
    labels: np.ndarray,
    grid_spacing_m: float,
    source_position: Tuple[int, ...],
    target: Tuple[int, ...],
    freq: float = 400e3,
    msg_amplitude: float = 40e-3,
    msg_duration: float = 5e-3,
    t_end: float = 5e-5,
    pml_size: float = 8.0,
) -> Dict:
    """
    Run the full multimodal MRI-TFUS-ARFI simulation.

    Steps:
        1. Map tissue labels → acoustic properties
        2. Map tissue labels → shear modulus (Kuhl CANN)
        3. j-Wave acoustic simulation → pressure/intensity
        4. Radiation force: F = 2αI/c
        5. Displacement: u = F/(µk²) via spectral solver
        6. MR-ARFI phase: Δφ = γ·G·δ·u
        7. Displacement recovery from phase (validation)

    Args:
        labels: 2D integer tissue label array.
        grid_spacing_m: Grid spacing in metres.
        source_position: Source grid indices.
        target: Target grid indices.
        freq: Ultrasound frequency (Hz).
        msg_amplitude: MSG amplitude (T/m).
        msg_duration: MSG lobe duration (s).
        t_end: Simulation end time (s).
        pml_size: PML thickness.

    Returns:
        Dict with all intermediate and final results.
    """
    import importlib.util, sys, os
    biophys = os.path.dirname(os.path.abspath(__file__))

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    acoustic = _load("acoustic", os.path.join(biophys, "acoustic.py"))
    jwa = _load("jwave_adapter", os.path.join(biophys, "jwave_adapter.py"))
    rf = _load("radiation_force", os.path.join(biophys, "radiation_force.py"))
    arfi = _load("mr_arfi", os.path.join(biophys, "mr_arfi.py"))

    results = {}

    # Step 1: Acoustic properties from labels
    props = acoustic.map_labels_to_properties(jnp.array(labels))
    results["sound_speed"] = props["sound_speed"]
    results["density"] = props["density"]
    results["attenuation"] = props["attenuation"]

    # Step 2: Shear modulus from labels
    mu_map = rf.map_labels_to_shear_modulus(jnp.array(labels))
    results["shear_modulus"] = mu_map

    # Step 3: j-Wave acoustic simulation
    from jwave.geometry import TimeAxis
    grid_shape = labels.shape
    domain = jwa.create_domain(grid_shape, grid_spacing_m)

    c = jnp.array(props["sound_speed"], dtype=jnp.float32)
    rho = jnp.array(props["density"], dtype=jnp.float32)
    medium = jwa.create_medium(domain, c, rho, pml_size=pml_size)
    ta = TimeAxis.from_medium(medium, cfl=0.3, t_end=t_end)
    dt = float(ta.dt)

    pos = jnp.array([[source_position[i] * grid_spacing_m for i in range(len(grid_shape))]])
    sources = jwa.create_sources(domain, pos, freq, jnp.zeros(1), jnp.ones(1), dt=dt, n_cycles=5)

    p_max = jwa.run_simulation_jax(medium, source_position, freq, t_end,
                                    time_axis=ta, sources=sources)
    results["p_max"] = p_max

    # Compute intensity: I = p²/(2Z)
    Z = props["density"] * props["sound_speed"]
    intensity = p_max**2 / (2 * Z)
    results["intensity"] = intensity

    # Step 4: Radiation force
    force = rf.compute_radiation_force_from_simulation(
        intensity, props["sound_speed"], props["attenuation"]
    )
    results["radiation_force"] = force

    # Step 5: Displacement
    displacement = rf.solve_displacement_quasistatic(force, mu_map, grid_spacing_m)
    results["displacement"] = displacement
    results["displacement_um"] = displacement * 1e6
    results["max_displacement_um"] = float(jnp.max(jnp.abs(displacement))) * 1e6

    # Step 6: MR-ARFI phase prediction
    phase = arfi.predict_arfi_phase(
        np.array(displacement), msg_amplitude, msg_duration
    )
    results["arfi_phase"] = phase

    # Step 7: Displacement recovery (validation roundtrip)
    u_recovered = arfi.recover_displacement_from_phase(
        phase, np.zeros_like(phase), msg_amplitude, msg_duration
    )
    recovery_error = float(np.max(np.abs(u_recovered - np.array(displacement))))
    results["displacement_recovery_error_m"] = recovery_error

    # Summary at target
    results["target"] = target
    results["p_at_target"] = float(p_max[target])
    results["force_at_target"] = float(force[target])
    results["displacement_at_target_um"] = float(displacement[target]) * 1e6
    results["phase_at_target_rad"] = float(phase[target])

    return results
