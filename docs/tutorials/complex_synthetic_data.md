# Creating Complex Synthetic Datasets

For testing and benchmarking Microstructure Imaging algorithms, simple homogeneous voxel simulations are often insufficient. Dmipy-JAX allows you to generate powerful synthetic datasets with known ground truth properties, including:
- Complex multi-compartment compositions (e.g., NODDI, SANDI).
- Spatially varying microstructure parameters (Phantoms).
- Realistic spatial noise distributions (Gaussian Random Fields).

This tutorial walks through creating a realistic "phantom" dataset.

## 1. Defining the Microstructure Model

First, we define the physics model we want to simulate. Dmipy's `MultiCompartmentModel` allows composing arbitrary models.

```python
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.cylinder_models import C1Stick
from dmipy.signal_models.gaussian_models import G2Zeppelin, G1Ball

# Define NODDI components
stick = C1Stick()      # Intra-neurite
zeppelin = G2Zeppelin() # Extra-neurite
ball = G1Ball()        # CSF

# Combine models
noddi = MultiCompartmentModel(models=[stick, zeppelin, ball])

# Apply Constraints (Standard NODDI)
# 1. Link Zepp_par = Stick_par
noddi.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
# 2. Tortuosity constraint: Zepp_perp = f(Zepp_par, f_stick)
noddi.set_tortuous_parameter(
    'G2Zeppelin_1_lambda_perp', 
    'G2Zeppelin_1_lambda_par', 
    'partial_volume_0', 
    'partial_volume_1'
)
```

## 2. Generating Spatially Varying Parameters

Instead of a single voxel, we create "Parameter Maps" where each parameter is a 3D numpy array.

```python
import numpy as np

dimensions = (20, 20, 20)
x = np.linspace(0, 1, dimensions[0])
y = np.linspace(0, 1, dimensions[1])
z = np.linspace(0, 1, dimensions[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Generate varying volume fractions
# Stick (intra) fraction increases along X
f_stick = 0.3 + 0.4 * X 
# CSF fraction increases along Y
f_csf = 0.2 * Y

# Ensure sum <= 1
f_zeppelin = 1.0 - f_stick - f_csf

# Define orientations (e.g., rotating along Z)
mu = np.zeros(dimensions + (3,))
theta = Z * np.pi
mu[..., 0] = np.cos(theta)
mu[..., 1] = np.sin(theta)

# Build parameter dictionary
parameters = {
    'partial_volume_0': f_stick,
    'partial_volume_1': f_zeppelin,
    'partial_volume_2': f_csf,
    'C1Stick_1_lambda_par': 1.7e-9, # Constant
    'G1Ball_1_lambda_iso': 3.0e-9,  # Constant
    'C1Stick_1_mu': mu,
    'G2Zeppelin_1_mu': mu           # Aligned
}
```

## 3. Simulating Signal

Use the `simulate_signal` method with your acquisition scheme. Dmipy handles the broadcasting over voxels automatically.

```python
from dmipy.data.synthetic import get_3shell_acquisition_scheme

scheme = get_3shell_acquisition_scheme()
signal = noddi.simulate_signal(scheme, parameters)
# signal shape: (20, 20, 20, N_measurements)
```

## 4. Adding Realistic Spatial Noise

Simple white noise doesn't capture realistic MRI artifacts like coil sensitivity profiles or tissue heterogeneity. We can use **Gaussian Random Fields (GRF)** to add spatially correlated noise.

### Spatially Varying SNR Map
Simulate coil sensitivity by varying SNR across the image.

```python
import scipy.ndimage

def generate_grf(shape, fwhm=5.0):
    noise = np.random.normal(0, 1, shape)
    sigma = fwhm / 2.355
    smooth = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
    smooth -= smooth.mean()
    smooth /= smooth.std()
    return smooth

# Base SNR 30, varying by +/- 10
snr_map = 30 + 10 * generate_grf(dimensions, fwhm=10.0)
snr_map = np.maximum(snr_map, 5.0) # Avoid negative SNR
```

### Adding Rician Noise
Add noise based on the local SNR map.

```python
sigma_map = 1.0 / snr_map
sigma_map = sigma_map[..., None] # Broadcast to measurements

noise_r = np.random.normal(0, 1, signal.shape) * sigma_map
noise_i = np.random.normal(0, 1, signal.shape) * sigma_map

noisy_signal = np.sqrt((signal + noise_r)**2 + noise_i**2)
```

## 5. Complete Example

For a ready-to-run script combining these techniques, check `examples/simulate_noddi_sandi.py` and `examples/spatial_noise_demo.py` in the repository.
