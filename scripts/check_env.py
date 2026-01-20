import sys
import importlib
import subprocess
import shutil

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    print(f"Checking {package_name}...", end=" ")
    try:
        pkg = importlib.import_module(import_name)
        version = getattr(pkg, "__version__", "unknown")
        print(f"OK (v{version})")
        return True, version
    except ImportError:
        print("MISSING")
        return False, None

def check_jax_backend():
    print("Checking JAX execution backend...", end=" ")
    try:
        import jax
        backend = jax.local_devices()[0].platform
        device_count = jax.device_count()
        print(f"OK ({backend.upper()}, {device_count} devices)")
        return backend
    except Exception as e:
        print(f"ERROR ({e})")
        return None

def check_command(command):
    print(f"Checking command '{command}'...", end=" ")
    path = shutil.which(command)
    if path:
        print(f"OK ({path})")
        return True
    else:
        print("MISSING")
        return False

def main():
    print("=== Dmipy-JAX Environment Check ===\n")
    
    # 1. Python Version
    py_ver = sys.version.split()[0]
    print(f"Python version: {py_ver}")
    if sys.version_info < (3, 9):
        print("WARNING: Python 3.10+ is recommended.")
    print("")

    # 2. Package Managers
    check_command("uv")
    print("")

    # 3. Core Dependencies
    missing_packages = []
    
    # Core JAX
    jax_ok, _ = check_package("jax")
    if not jax_ok:
        missing_packages.append("jax")
    
    check_package("jaxlib")
    
    # Kidger Stack
    stack_packages = ["equinox", "optimistix", "lineax", "diffrax", "jaxtyping", "optax"]
    for pkg in stack_packages:
        ok, _ = check_package(pkg)
        if not ok:
            missing_packages.append(pkg)

    print("")
    
    # 4. JAX Backend Check
    if jax_ok:
        backend = check_jax_backend()
        if backend == "cpu":
            print("NOTICE: JAX is running on CPU. For acceleration, ensure CUDA/ROCm is installed if available.")
    
    # 5. Summary
    print("\n=== Summary ===")
    if missing_packages:
        print("MISSING PACKAGES found. Please install them using uv:")
        print(f"  uv add {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        print("Environment looks good for Dmipy-JAX development!")
        sys.exit(0)

if __name__ == "__main__":
    main()
