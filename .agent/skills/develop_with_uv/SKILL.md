---
name: develop_with_uv
description: Instructions for developing in this project which uses `uv` for package management.
---

# Develop with UV

This project uses `uv` for python package management. 

## context
The `uv` tool replaces `pip`, `pip-tools`, and `virtualenv`. You should use `uv` commands for all package operations.

## Common Commands

### Running Scripts
Instead of `python script.py`, use:
```bash
uv run script.py
```
This ensures the script runs in the project's environment.

### Adding Dependencies
To add a new package:
```bash
uv add <package_name>
```

### Syncing Environment
To sync the environment with `uv.lock`:
```bash
uv sync
```

### Locking Dependencies
To update `uv.lock` based on `pyproject.toml`:
```bash
uv lock
```

## Important Notes
- Do not use `pip install` directly.
- The virtual environment is managed by `uv`.
- If you need to run a command inside the venv, prefix it with `uv run`.
