import mdt_adapter
import numpy as np
import json
import collections

# Define a rich protocol
# 3 shells: b=1000, 2000, 3000
# 32 directions each + b0s
def create_protocol():
    bval_shells = [1000, 2000, 3000]
    n_dirs = 32
    
    bvals = [0]
    bvecs = [[0, 0, 0]]
    
    # Simple fibonacci sphere or random points
    # For now, just random points on sphere
    np.random.seed(42)
    for b in bval_shells:
        for _ in range(n_dirs):
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            bvals.append(b * 1e6) # SI units s/m^2
            bvecs.append(v)
            
    return np.array(bvals), np.array(bvecs)

def create_full_protocol():
    bval_shells = [1000, 2000, 3000]
    n_dirs = 32
    
    bvals = [0]
    bvecs = [[0, 0, 0]]
    # Timing parameters (SI units)
    # Typical Connectome/HCP like?
    # Delta = 43.1 ms, delta = 10.6 ms, TE = 89.5 ms
    Delta_val = 0.0431
    delta_val = 0.0106
    TE_val = 0.0895
    
    Delta = [Delta_val]
    delta = [delta_val]
    TE = [TE_val]
    
    np.random.seed(42)
    for b in bval_shells:
        for _ in range(n_dirs):
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            bvals.append(b * 1e6) 
            bvecs.append(v)
            Delta.append(Delta_val)
            delta.append(delta_val)
            TE.append(TE_val)
            
    return np.array(bvals), np.array(bvecs), {'Delta': np.array(Delta), 'delta': np.array(delta), 'TE': np.array(TE)}

bvals, bvecs, protocol_params = create_full_protocol()

# List of models to test
# We focus on models we can map to dmipy-jax or that are requested
models_to_test = [
    # Top priority
    ('Stick', 'BallStick_r1', {'S0.s0': 1.0, 'w_stick0.w': 1.0, 'Stick0.theta': 0.0, 'Stick0.phi': 0.0}), # Pure stick using BallStick
    ('Ball', 'BallStick_r1', {'S0.s0': 1.0, 'w_stick0.w': 0.0, 'Stick0.theta': 0.0, 'Stick0.phi': 0.0, 'Ball.d': 2e-9}), # Pure Ball using BallStick (if params allow w=0)
    ('Zeppelin', 'Tensor', {'S0.s0': 1.0, 'Tensor.d': 2e-9, 'Tensor.dperp0': 0.5e-9, 'Tensor.dperp1': 0.5e-9, 'Tensor.theta': 0.0, 'Tensor.phi': 0.0, 'Tensor.psi': 0.0}),
    ('Tensor', 'Tensor', {'S0.s0': 1.0, 'Tensor.d': 2e-9, 'Tensor.dperp0': 0.5e-9, 'Tensor.dperp1': 0.8e-9, 'Tensor.theta': 0.5, 'Tensor.phi': 0.5, 'Tensor.psi': 0.2}),
    ('NODDI', 'NODDI', {'S0.s0': 1.0, 'w_ic.w': 0.5, 'NODDI_IC.theta': 0.0, 'NODDI_IC.phi': 0.0, 'NODDI_IC.kappa': 2.0, 'w_ec.w': 0.5}),
    ('Kurtosis', 'Kurtosis', {'S0.s0': 1.0, 'KurtosisTensor.d': 2e-9, 'KurtosisTensor.dperp0': 0.5e-9, 'KurtosisTensor.dperp1': 0.5e-9, 'KurtosisTensor.theta': 0.0, 'KurtosisTensor.phi': 0.0, 'KurtosisTensor.psi': 0.0, 'KurtosisTensor.W_0000': 0.1}), # Simplified params
    # Add CHARMED, IVIM etc as we confirm parameters
]

results = {}

print("Starting Benchmark Generation...")

# Specific logic for "Ball using BallStick" - MDT might complain about missing params if we don't supply Ball.d
# We need to be careful with param names.
# For BallStick_r1: params are ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi']. Wait, where is Ball.d?
# In MDT 1.2.7 BallStick might assume fixed diffusivity or have it in S0?
# Let's check param lists from introspection output again.
# BallStick_r1: ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi']
# It seems BallStick_r1 does NOT have a variable diffusivity for the ball/stick part? That's strange.
# Maybe it's fixed? Or maybe listed implicitly?
# Ah, Tensor has 'Tensor.d'.

# Let's rebuild the list based on introspection output:
# BallStick_r1: ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi'] 
# -> This looks like it lacks diffusivities! Maybe it uses defaults 1.7e-9?

# CHARMED_r1: ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', ... 'CHARMEDRestricted0.d']
# This has explicit diffusivities.

# Let's try running a subset first.

models_map = {
    'Tensor': {
        'mdt_name': 'Tensor',
        'params': {
            'S0.s0': 1.0,
            'Tensor.d': 1.7e-9, 'Tensor.dperp0': 0.2e-9, 'Tensor.dperp1': 0.2e-9,
            'Tensor.theta': 0.0, 'Tensor.phi': 0.0, 'Tensor.psi': 0.0
        }
    },
    'NODDI': {
        'mdt_name': 'NODDI',
        'params': {
            'S0.s0': 1.0, 
            'w_ic.w': 0.6, 
            'w_ec.w': 0.4, # MDT usually enforces sum=1? Or w_ic + w_ec + w_csf = 1?
            # Introspection: ['S0.s0', 'w_ic.w', 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa', 'w_ec.w']
            # Only w_ic and w_ec listed. Implies CSF is remainder? Or w_ic and w_ic are relative?
            # MDT usually is w_ic volume fraction. 
            'NODDI_IC.theta': 0.0, 'NODDI_IC.phi': 0.0, 'NODDI_IC.kappa': 1.0
        }
    },
    'CHARMED': {
        'mdt_name': 'CHARMED_r1',
        'params': {
            'S0.s0': 1.0,
            'Tensor.d': 1.7e-9, 'Tensor.dperp0': 0.2e-9, 'Tensor.dperp1': 0.2e-9, # Hindered part (Tensor)
            'Tensor.theta': 0.0, 'Tensor.phi': 0.0, 'Tensor.psi': 0.0,
            'w_res0.w': 0.5,
            'CHARMEDRestricted0.d': 1.0e-9,
            'CHARMEDRestricted0.theta': 0.5, 'CHARMEDRestricted0.phi': 0.5
        }
    },
     'IVIM': {
        'mdt_name': 'IVIM',
        'params': {
            'S0.s0': 1.0,
            'Perfusion.d': 10e-9,
            'w_diffusion.w': 0.9,
            'Diffusion.d': 1e-9
        }
    },
    'BallStick': {
        'mdt_name': 'BallStick_r1',
        'params': {
            'S0.s0': 1.0,
            'w_stick0.w': 0.5,
            'Stick0.theta': 0.0, 'Stick0.phi': 0.0
            # Stick0 diffusivity is likely fixed or defaults to 1.7e-9 if not listed
        }
    },
    'Kurtosis': {
        'mdt_name': 'Kurtosis',
        'params': {
             'S0.s0': 1.0, 
             'KurtosisTensor.d': 2e-9, 
             'KurtosisTensor.dperp0': 0.5e-9, 
             'KurtosisTensor.dperp1': 0.5e-9, 
             'KurtosisTensor.theta': 0.0, 'KurtosisTensor.phi': 0.0, 'KurtosisTensor.psi': 0.0, 
             'KurtosisTensor.W_0000': 0.1,
             'KurtosisTensor.W_1000': 0.0, 'KurtosisTensor.W_1100': 0.0, 'KurtosisTensor.W_2000': 0.0,
             # Fill zeros for simplified kurtosis
             'KurtosisTensor.W_1110': 0.0, 'KurtosisTensor.W_1111': 0.1,
             'KurtosisTensor.W_2100': 0.0, 'KurtosisTensor.W_2110': 0.0, 'KurtosisTensor.W_2111': 0.0,
             'KurtosisTensor.W_2200': 0.0, 'KurtosisTensor.W_2210': 0.0, 'KurtosisTensor.W_2211': 0.0,
             'KurtosisTensor.W_2220': 0.0, 'KurtosisTensor.W_2221': 0.0, 'KurtosisTensor.W_2222': 0.1
        }
    },
    'ActiveAx': {
        'mdt_name': 'ActiveAx',
        'params': {
            'S0.s0': 1.0,
            'w_ic.w': 0.6,
            'w_ec.w': 0.4,
            'CylinderGPD.theta': 0.0, 'CylinderGPD.phi': 0.0,
            'CylinderGPD.R': 4e-6 # 4 microns
        }
    }
}

for name, config in models_map.items():
    print(f"Simulating {name} ({config['mdt_name']})...")
    try:
        model = mdt_adapter.get_mdt_model(config['mdt_name'])
        params = config['params']
        
        # Simulate
        signal = mdt_adapter.simulate_signal(model, bvals, bvecs, params, protocol_params=protocol_params)
        
        results[name] = {
            'mdt_model': config['mdt_name'],
            'params': params,
            'bvals': bvals.tolist(),
            'bvecs': bvecs.tolist(),
            'protocol_params': {k: v.tolist() for k,v in protocol_params.items()},
            'signal': signal.tolist()
        }
        print(f"  Success! Signal mean: {np.mean(signal)}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()

# Save results
with open('mdt_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved benchmarks for {len(results)} models to mdt_benchmarks.json")
