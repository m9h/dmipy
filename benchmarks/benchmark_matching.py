#!/usr/bin/env python3
"""
Benchmark: dictionary matching throughput on GPU.

Times cosine-similarity matching of V voxels against an N-entry library.
Reports throughput in voxels/second for various (V, N) combinations.

Usage::

    python benchmarks/benchmark_matching.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.library.matcher import DictionaryMatcher


def build_simulator():
    """Ball model simulator for benchmarking."""
    key = jax.random.PRNGKey(0)
    n_meas = 90
    bvecs = jax.random.normal(key, (n_meas, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.concatenate([
        jnp.zeros(2),
        jnp.full(32, 1e9),
        jnp.full(56, 2e9),
    ])
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        d = params[0]
        f = params[1]
        cos2 = acq.gradient_directions[:, 2] ** 2
        s_ball = jnp.exp(-acq.bvalues * d)
        s_stick = jnp.exp(-acq.bvalues * d * 1.5 * cos2)
        return f * s_ball + (1 - f) * s_stick

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d", "f"],
        parameter_ranges={"d": (0.5e-9, 3.0e-9), "f": (0.1, 0.9)},
        acquisition=acq,
    )


def benchmark_matching(n_library, n_voxels, k_best=5, batch_size=4096):
    device = jax.devices()[0]
    print(f"Device: {device}")
    print(f"Library: {n_library:,}   Voxels: {n_voxels:,}   K: {k_best}")

    sim = build_simulator()

    # Generate library
    gen = LibraryGenerator(sim, chunk_size=50_000)
    t0 = time.perf_counter()
    lib_params, lib_signals = gen.generate(n_library, key=jax.random.PRNGKey(1))
    jax.block_until_ready(lib_signals)
    t_gen = time.perf_counter() - t0
    print(f"Library generation: {t_gen:.2f} s")

    lib = SimulationLibrary(
        params=lib_params,
        signals=lib_signals,
        parameter_names=["d", "f"],
    )

    matcher = DictionaryMatcher(lib, k_best=k_best)

    # Generate test voxels
    key_test = jax.random.PRNGKey(42)
    _, test_signals = sim.sample_and_simulate(key_test, n_voxels)
    jax.block_until_ready(test_signals)

    # Warmup
    warmup_batch = test_signals[:min(100, n_voxels)]
    best, _ = matcher.match_batch(warmup_batch)
    jax.block_until_ready(best)

    # Timed batched matching
    import math
    n_batches = math.ceil(n_voxels / batch_size)

    t0 = time.perf_counter()
    for i in range(n_batches):
        s = i * batch_size
        e = min(s + batch_size, n_voxels)
        batch = test_signals[s:e]
        best, topk = matcher.match_batch(batch)
        jax.block_until_ready(best)
    t_match = time.perf_counter() - t0

    throughput = n_voxels / t_match
    print(f"Matching time: {t_match:.2f} s")
    print(f"Throughput:    {throughput:,.0f} voxels/s")
    print(f"Per-voxel:     {t_match / n_voxels * 1e6:.1f} us")
    print()

    return t_match, throughput


def main():
    print("=" * 60)
    print("Dictionary Matching Benchmark")
    print("=" * 60)
    print()

    configs = [
        (10_000, 10_000),
        (100_000, 10_000),
        (500_000, 10_000),
        (500_000, 100_000),
    ]

    results = []
    for n_lib, n_vox in configs:
        print(f"--- Library={n_lib:,}  Voxels={n_vox:,} ---")
        t, tp = benchmark_matching(n_lib, n_vox)
        results.append((n_lib, n_vox, t, tp))

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Library':>10s}  {'Voxels':>10s}  {'Time (s)':>10s}  {'Vox/s':>12s}")
    for n_lib, n_vox, t, tp in results:
        print(f"{n_lib:>10,d}  {n_vox:>10,d}  {t:>10.2f}  {tp:>12,.0f}")


if __name__ == "__main__":
    main()
