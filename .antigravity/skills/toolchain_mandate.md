# Toolchain Mandate: UV ONLY

**CRITICAL INSTRUCTION:** This project EXCLUSIVELY uses `uv` for Python management.
**STRICT PROHIBITION:** Do NOT use `pip`, `poetry`, `conda`, `virtualenv`, or `venv` directly.

## 1. Environment & Installation
* **Create Venv:** `uv venv`
* **Install Dependencies:** `uv sync` (or `uv pip install -r pyproject.toml`)
* **Add Package:** `uv add <package_name>` (e.g., `uv add jax`)
* **Add Dev Package:** `uv add --dev <package_name>`

## 2. Execution
* **Run Scripts:** ALWAYS use `uv run`.
    * *Correct:* `uv run python experiments/ste_test.py`
    * *Incorrect:* `python experiments/ste_test.py`
* **Run Tests:** `uv run pytest`

## 3. Dependency Resolution
* **Lock File:** Respect `uv.lock`. Never delete it manually.
* **Update:** Use `uv lock --upgrade` to update dependencies.

## 4. Performance
* `uv` is significantly faster than pip. If an installation takes >10s, verify you are using `uv` and not falling back to legacy tools.
