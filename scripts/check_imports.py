import sys
print(f"Python: {sys.version}")
try:
    import jax
    print(f"JAX: {jax.__version__}")
    import equinox
    print(f"Equinox: {equinox.__version__}")
    import optimistix
    print(f"Optimistix: {optimistix.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
