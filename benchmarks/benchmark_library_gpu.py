#!/usr/bin/env python3
"""
Benchmark: GPU vs CPU throughput for simulation library generation.

Generates 500K Ball+Stick+Zeppelin library entries and reports wall-clock
time on the default device (GPU if available, else CPU).

Usage::

    python benchmarks/benchmark_library_gpu.py
    JAX_PLATFORMS=cpu python benchmarks/benchmark_library_gpu.py  # force CPU
"""

import time
import jax
import jax.numpy as jnp

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator


def build_ball_stick_zeppelin_simulator():
    """3-compartment Ball+Stick+Zeppelin forward model."""
    key = jax.random.PRNGKey(0)
    n_meas = 90  # realistic multi-shell: 2 b0 + 32 dirs x b=1000 + 56 dirs x b=2000
    bvecs = jax.random.normal(key, (n_meas, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.concatenate([
        jnp.zeros(2),
        jnp.full(32, 1e9),
        jnp.full(56, 2e9),
    ])
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        d_ball = params[0]
        d_stick = params[1]
        d_perp = params[2]
        f_ball = params[3]
        f_stick = params[4]
        f_zep = 1.0 - f_ball - f_stick

        # Ball
        s_ball = jnp.exp(-acq.bvalues * d_ball)

        # Stick along z
        cos2 = acq.gradient_directions[:, 2] ** 2
        s_stick = jnp.exp(-acq.bvalues * d_stick * cos2)

        # Zeppelin (axially symmetric tensor along z)
        s_zep = jnp.exp(-acq.bvalues * (d_stick * cos2 + d_perp * (1 - cos2)))

        return f_ball * s_ball + f_stick * s_stick + f_zep * s_zep

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_ball", "d_stick", "d_perp", "f_ball", "f_stick"],
        parameter_ranges={
            "d_ball": (2.0e-9, 3.0e-9),
            "d_stick": (0.5e-9, 2.0e-9),
            "d_perp": (0.1e-9, 1.5e-9),
            "f_ball": (0.0, 0.3),
            "f_stick": (0.1, 0.7),
        },
        acquisition=acq,
    )


def benchmark(n_entries, chunk_size, n_warmup=1):
    sim = build_ball_stick_zeppelin_simulator()
    gen = LibraryGenerator(sim, chunk_size=chunk_size)

    device = jax.devices()[0]
    print(f"Device: {device}")
    print(f"Entries: {n_entries:,}   Chunk size: {chunk_size:,}")
    print(f"Signal dim: {sim.signal_dim}   Theta dim: {sim.theta_dim}")
    print()

    # Warmup
    for _ in range(n_warmup):
        _ = gen.generate(min(chunk_size, 1000), key=jax.random.PRNGKey(99))
        jax.block_until_ready(_)

    # Timed run
    key = jax.random.PRNGKey(42)
    t0 = time.perf_counter()
    params, signals = gen.generate(n_entries, key=key)
    jax.block_until_ready(signals)
    elapsed = time.perf_counter() - t0

    throughput = n_entries / elapsed
    print(f"Wall time:   {elapsed:.2f} s")
    print(f"Throughput:  {throughput:,.0f} entries/s")
    print(f"Output:      params {params.shape}, signals {signals.shape}")
    print(f"Signal range: [{float(jnp.min(signals)):.4f}, {float(jnp.max(signals)):.4f}]")

    return elapsed, throughput


def main():
    print("=" * 60)
    print("Library Generation Benchmark")
    print("=" * 60)
    print()

    for n_entries in [10_000, 100_000, 500_000]:
        chunk = min(n_entries, 50_000)
        print(f"--- {n_entries:,} entries ---")
        benchmark(n_entries, chunk)
        print()


if __name__ == "__main__":
    main()
