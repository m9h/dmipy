import sys
print(sys.path)
try:
    import jax
    print("JAX version:", jax.__version__)
    import equinox
    print("Equinox imported")
    import dmipy_jax.core.acquisition as acq
    print("Acquisition imported")
    from dmipy_jax.core.multimodal import JointModel
    print("JointModel imported")
except ImportError as e:
    print("ImportError:", e)
except Exception as e:
    print("Exception:", e)
print("Done")
