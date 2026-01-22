import collections
import collections.abc
# Monkeypatch for Python 3.10+ where collections.Mapping is removed
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping

import mdt
import numpy as np
import tempfile
import os

def get_mdt_model(name: str):
    """
    Factory function to instantiate MDT models.
    
    Args:
        name: Name of the model (Stick, Ball, Zeppelin, NODDI)
        
    Returns:
        MDT model instance
    """
    if name in mdt.get_models_list():
        return mdt.get_model(name)()
        
    # Fallbacks for simplified names or components
    if name == 'Stick':
        # Use Tensor model for Stick (d_perp = 0)
        return mdt.get_model('Tensor')()

    elif name == 'Ball':
        # Ball is usually available or use Tensor with d_para = d_perp
        try:
             return mdt.get_component('composite_models', 'Ball')()
        except:
             # Fallback to Tensor if Ball is missing
             return mdt.get_model('Tensor')()
    
    elif name == 'Zeppelin':
        # Zeppelin is a Tensor
        return mdt.get_model('Tensor')()
        
    else:
        raise ValueError(f"Unknown model name: {name}")

def simulate_signal(model, bvals, bvecs, params, protocol_params=None):
    """
    Simulates signal.
    
    Args:
        model: MDT model instance
        bvals: (N,) array in s/m^2 (SI)
        bvecs: (N, 3) array
        params: dict of model params
        protocol_params: dict of optional protocol arrays (Delta, delta, TE) in SI units.
    """
    
    bval_path = None
    bvec_path = None
    # Dictionary to hold paths for protocol creation
    protocol_kwargs = {}
    temp_files = [] # Keep track to close/remove later

    try:
        # Create temp files for bvals/bvecs
        f_bval = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bval')
        temp_files.append(f_bval.name)
        f_bvec = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bvec')
        temp_files.append(f_bvec.name)

        # Write bvals (convert s/m^2 to s/mm^2)
        bvals_si = bvals.flatten()
        bvals_smm2 = bvals_si / 1e6
        np.savetxt(f_bval.name, bvals_smm2[None], fmt='%g', delimiter=' ')
        protocol_kwargs['bvals'] = f_bval.name
        
        # Write bvecs (3, N) - MDT expects FSL style usually?
        # Check previous implementation: np.savetxt(f_bvec, bvecs.T, ...)
        np.savetxt(f_bvec.name, bvecs.T, fmt='%g', delimiter=' ')
        protocol_kwargs['bvecs'] = f_bvec.name

        # Handle other protocol parameters (Delta, delta, TE)
        # MDT create_protocol kwargs are passed to protocol.with_updates()
        # which likely expects values/arrays, not filenames.
        if protocol_params:
            for key, val in protocol_params.items():
                protocol_kwargs[key] = val

        protocol = mdt.create_protocol(**protocol_kwargs)
        
        print(f"DEBUG Protocol Created: {protocol}")
        if hasattr(protocol, 'length'):
             print(f"DEBUG Protocol Length: {protocol.length}")
        if hasattr(protocol, 'get_b_values'):
             print(f"DEBUG Protocol B-values: {protocol.get_b_values()}")
        
        # Cleanup
        for p in temp_files:
            if os.path.exists(p):
                os.remove(p)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to create protocol with mdt.create_protocol: {e}")
    
    # 2. Prepare parameters
    # MDT models often expect a specific parameter vector or dictionary ordering.
    # We might need to map 'params' dict to the model's parameter list.
    
    # Let's see what the model expects
    required_params = model.get_free_param_names()
    
    # Construct input values in the correct order
    # Note: MDT might be voxel-based, so input shape might need to be (X, Y, Z, N_params)
    # We'll assume a single voxel for now: (1, 1, 1, N_params)
    
    param_values = []
    for p_name in required_params:
        if p_name in params:
            param_values.append(params[p_name])
        else:
            # Handle defaults or composite names (e.g., 'Stick.theta')
            # For simplicity, try exact match first
             raise ValueError(f"Missing parameter: {p_name}. Required: {required_params}")
             
    input_data = np.array(param_values).reshape(1, 1, 1, -1)
    
    # 3. compute output
    # Use mdt.simulations.simulate_signals
    # Signature: (model, protocol, parameters)
    # Model should be the object.
    # Parameters should be the input data array (X, Y, Z, N_params)
    
    try:
        from mdt import simulations
        # simulate_signals returns a dictionary of results usually, or array
        result = simulations.simulate_signals(model, protocol, input_data)
        
        # result might be an array or dict. 
        # If array, shape (X, Y, Z, N_vols)
        if hasattr(result, 'flatten'):
             return result.flatten()
        else:
             return result # Should check structure
             
    except Exception as e:
        raise RuntimeError(f"Signal simulation failed during call to simulate_signals: {e}")

