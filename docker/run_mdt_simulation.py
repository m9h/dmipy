import argparse
import numpy as np
import collections
import mdt_adapter
import sys

def main():
    parser = argparse.ArgumentParser(description='Run MDT simulation from NPZ input.')
    parser.add_argument('--input', type=str, required=True, help='Path to input NPZ file.')
    parser.add_argument('--output', type=str, required=True, help='Path to output NPZ file.')
    args = parser.parse_args()

    # Load input data
    try:
        data = np.load(args.input, allow_pickle=True)
        # Extract required fields
        model_name = str(data['model_name'])
        bvals = data['bvals'] # Expecting s/m^2
        bvecs = data['bvecs']
        param_dict = data['params'].item() # Dictionary of parameter arrays or scalars
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # Convert parameter arrays to list of dicts or handle vectorized input if adapter supports it.
    # The current adapter in mdt_adapter.py:simulate_signal seems to handle a single voxel input 
    # (1, 1, 1, N_params) derived from 'params' dict.
    # However, for efficiency, we likely want to loop over many samples if we passed arrays.
    
    # Check if parameters are arrays
    num_samples = 1
    first_key = list(param_dict.keys())[0]
    if hasattr(param_dict[first_key], '__len__') and not isinstance(param_dict[first_key], (str, bytes)):
         num_samples = len(param_dict[first_key])
    
    print(f"Running simulation for model: {model_name}, Samples: {num_samples}")

    # Instantiate model
    try:
        model = mdt_adapter.get_mdt_model(model_name)
    except Exception as e:
        print(f"Error instantiating model {model_name}: {e}")
        sys.exit(1)

    signals = []

    # Iterate and simulate
    # This is a bit inefficient for Python loop, but MDT's simulate_signals might need (X,Y,Z) anyway
    # mdt_adapter.simulate_signal handles the protocol creation per call, which is expensive if loop.
    # We should probably modify adapter or just accept overhead for validation (it's 10-100k samples, might take a few mins).
    # Since this is a validation script, we can optimize later if needed.
    
    # NOTE: mdt_adapter.simulate_signal creates temp files for every call. This will be very slow for 100k samples.
    # Optimizing: Create protocol ONCE outside the loop if bvals/bvecs are constant.
    
    # Reuse protocol logic from adapter manually here to avoid overhead? 
    # Or update mdt_adapter to accept pre-built protocol? 
    # For now, let's try to use the adapter as is, but maybe we can batch if MDT supports (N_voxels).
    # MDT usually supports volume input.
    
    # Restructure params for volume-based simulation in MDT
    # MDT expects input_data of shape (X, Y, Z, N_params)
    # We can fake a volume: (num_samples, 1, 1, N_params)
    
    # 1. Inspect model parameters requirement
    required_params = model.get_free_param_names()
    
    # 2. Build input array (N, 1, 1, P)
    input_volume = np.zeros((num_samples, 1, 1, len(required_params)))
    
    for i, p_name in enumerate(required_params):
        if p_name not in param_dict:
             print(f"Error: Missing parameter {p_name}")
             print(f"Required parameters: {required_params}")
             sys.exit(1)
        val = param_dict[p_name]
        # if scalar, broadcast
        if np.isscalar(val) or np.ndim(val)==0:
             input_volume[:, 0, 0, i] = val
        else:
             input_volume[:, 0, 0, i] = val

    # 3. Create protocol ONCE
    # We need to replicate protocol creation from mdt_adapter, but adapted to reuse.
    # Actually, let's just use mdt directly here for the protocol to ensure efficiency.
    import mdt
    import tempfile
    import os
    
    # Create temp files
    f_bval = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bval')
    f_bvec = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bvec')
    
    try:
        # Write bvals (convert s/m^2 to s/mm^2)
        bvals_si = bvals.flatten()
        bvals_smm2 = bvals_si / 1e6
        np.savetxt(f_bval.name, bvals_smm2[None], fmt='%g', delimiter=' ')
        
        # Write bvecs
        np.savetxt(f_bvec.name, bvecs.T, fmt='%g', delimiter=' ')
        
        protocol = mdt.create_protocol(bvals=f_bval.name, bvecs=f_bvec.name)
    finally:
        f_bval.close()
        f_bvec.close()
        os.unlink(f_bval.name)
        os.unlink(f_bvec.name)

    # 4. Simulate All
    from mdt import simulations
    
    print("Starting batch simulation...")
    # simulate_signals(model, protocol, input_data)
    # input_data: (X, Y, Z, N_params)
    # Returns: (X, Y, Z, N_vols)
    
    try:
        result_volume = simulations.simulate_signals(model, protocol, input_volume)
        # result_volume shape should be (N, 1, 1, N_measurements)
        
        # Squeeze to (N, N_measurements)
        signals = result_volume.reshape(num_samples, -1)
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save output
    np.savez(args.output, signals=signals)
    print(f"Saved signals to {args.output}")

if __name__ == '__main__':
    main()
