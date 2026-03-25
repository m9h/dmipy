
import numpy as np
import disimpy.simulations as simulations
import inspect

grad_array = np.zeros((3, 60), dtype=np.float64)
grad_array[0, :] = 0.04

print(f"My Numpy: {np.__version__}")
# Disimpy imports numpy?
import disimpy
print("SOURCE CODE:")
print(inspect.getsource(simulations.simulation)[:4000])
# Disimpy doesn't expose numpy directly usually, but let's check substrate
from disimpy import substrates
# substrate uses np
print(f"Disimpy substrate np: {substrates.np.__version__}")
print(f"Same numpy? {np is substrates.np}")

gradient = grad_array

is_int_float = isinstance(gradient, (int, float, np.floating, np.integer))
is_1d = (isinstance(gradient, np.ndarray) and gradient.ndim == 1 and len(gradient) == 3)
is_2d = (isinstance(gradient, np.ndarray) and gradient.ndim == 2 and gradient.shape[0] == 3)

print((
        not is_int_float
        and not is_1d
        and not is_2d
    ))
    
print(f"Is 2D? {is_2d}")
print(f"Is Instance? {isinstance(gradient, np.ndarray)}")
print(f"Ndim? {gradient.ndim}")
print(f"Shape? {gradient.shape}")

# Try to call simulation with dummy?
substrate = substrates.free()
try:
    simulations.simulation(1, 1e-9, gradient, 1e-3, substrate)
except Exception as e:
    print(f"Caught error: {e}")
