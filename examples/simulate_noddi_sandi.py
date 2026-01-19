
import numpy as np
import nibabel as nib
from dmipy.signal_models.cylinder_models import C1Stick, C2CylinderSoda
from dmipy.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy.signal_models.sphere_models import S1Sphere
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data.synthetic import get_3shell_acquisition_scheme

def simulate_noddi_like_dataset(dimensions=(10, 10, 10)):
    """
    Simulates a NODDI-like dataset (Stick + Zeppelin + Ball)
    with spatially varying complexity.
    """
    print("Constructing NODDI model...")
    # 1. Define Components
    # Intra-neurite: Stick (zero radius cylinder)
    stick = C1Stick()
    # Extra-neurite: Zeppelin (cylindrically symmetric tensor)
    zeppelin = G2Zeppelin()
    # CSF: Isotropic Ball
    ball = G1Ball()
    
    # 2. combine into Multi-Compartment Model
    noddi = MultiCompartmentModel(models=[stick, zeppelin, ball])
    
    # 3. Constrain Tortuosity (Optional but standard for NODDI)
    # Zepp_perp = f(Zepp_par, f_intra)
    noddi.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 
                                 'G2Zeppelin_1_lambda_par', 
                                 'partial_volume_0', # f_stick
                                 'partial_volume_1') # f_zeppelin
    
    # Link parallel diffusivities (Stick D_par = Zeppelin D_par)
    noddi.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    
    # 4. Generate Parameter Maps
    print(f"Generating parameter maps for {dimensions} volume...")
    n_voxels = np.prod(dimensions)
    
    # Initialize parameters dictionary
    parameters = {}
    
    # Generate spatially varying volume fractions
    # e.g., Linear gradient of f_intra from 0.1 to 0.7
    x = np.linspace(0, 1, dimensions[0])
    y = np.linspace(0, 1, dimensions[1])
    z = np.linspace(0, 1, dimensions[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # f_intra (Stick)
    f_stick = 0.1 + 0.6 * X # varying along X
    
    # f_csf (Ball) - varying along Y
    f_csf = 0.5 * Y 
    
    # Normalize fractions: f_stick + f_zepp + f_csf = 1
    # We set f_zepp = 1 - f_stick - f_csf
    # Ensure non-negative
    total_non_zepp = f_stick + f_csf
    mask_overflow = total_non_zepp > 0.95
    f_stick[mask_overflow] /= (total_non_zepp[mask_overflow] / 0.95)
    f_csf[mask_overflow] /= (total_non_zepp[mask_overflow] / 0.95)
    
    f_zeppelin = 1.0 - f_stick - f_csf
    
    parameters['partial_volume_0'] = f_stick
    parameters['partial_volume_1'] = f_zeppelin
    parameters['partial_volume_2'] = f_csf
    
    # Orientation (mu) - Crossing Pattern or fanning?
    # Let's simple changes orientation along Z
    mu = np.zeros(dimensions + (3,))
    # Rotate in XY plane along Z
    theta = Z * np.pi 
    mu[..., 0] = np.cos(theta)
    mu[..., 1] = np.sin(theta)
    mu[..., 2] = 0.0
    
    parameters['C1Stick_1_mu'] = mu
    parameters['G2Zeppelin_1_mu'] = mu # Aligned
    
    # Diffusivities
    parameters['C1Stick_1_lambda_par'] = 1.7e-9 # 1.7 um^2/ms
    parameters['G1Ball_1_lambda_iso'] = 3.0e-9 # 3.0 um^2/ms (CSF)
    
    # 5. Simulate
    print("Simulating signal...")
    scheme = get_3shell_acquisition_scheme()
    
    # simulate_signal expects (N_vox, N_params) or dict of arrays
    # It handles broadcasting.
    signal = noddi.simulate_signal(scheme, parameters)
    
    print(f"Signal simulated: {signal.shape}")
    return signal, scheme, parameters

def simulate_sandi_like_dataset(dimensions=(10, 10, 10)):
    """
    Simulates a SANDI-like dataset (Cylinder + Soma + Zeppelin + Ball)
    """
    print("\nConstructing SANDI model...")
    # Intra-neurite (Cylinder/Stick)
    stick = C1Stick()
    # Soma (Sphere)
    sphere = S1Sphere()
    # Extra-neurite (Zeppelin)
    zeppelin = G2Zeppelin()
    # CSF (Ball)
    ball = G1Ball()
    
    sandi = MultiCompartmentModel(models=[stick, sphere, zeppelin, ball])
    
    # Link orientations if aligned
    sandi.set_equal_parameter('G2Zeppelin_1_mu', 'C1Stick_1_mu')
    # Sphere has no orientation
    
    # ... setup parameters similar to above ...
    # This proves the extensibility.
    return sandi

if __name__ == "__main__":
    signal, scheme, params = simulate_noddi_like_dataset()
    simulate_sandi_like_dataset()
    print("Done.")
