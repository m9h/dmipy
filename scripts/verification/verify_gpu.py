import jax
import os

print("JAX Version:", jax.__version__)
print("\nJAX Devices:")
try:
    devices = jax.devices()
    for d in devices:
        print(f"  - {d}")
        print(f"    Platform: {d.platform}")
        print(f"    Device Type: {d.device_kind}")
except Exception as e:
    print(f"Error getting devices: {e}")

print("\nEnvironment Variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

print("\nDefault Backend:", jax.default_backend())
