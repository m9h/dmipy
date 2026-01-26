# Vendor Directory

This directory contains external non-Python dependencies, binaries, and third-party tools that are not directly importable as Python modules within the `dmipy_jax` package.

## Convention
- **Naming**: Use `snake_case` or `PascalCase` matching the upstream project name.
- **Content**: Submodules, C++ headers, pre-compiled binaries, or data generation tools (e.g., `CATERPillar`).
- **Python Dependencies**: Pure Python dependencies bundled with the library should go in `dmipy_jax/external/`.
