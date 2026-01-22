import numpy as np
import mdt
import mdt_adapter

def run_verification():
    print("Starting MDT Oracle Verification...")
    
    # Define Protocol
    # b-values in s/m^2 (MDT usually works in SI or standard units, check documentation if fails)
    # Standard diffusion: b=0, 1000, 2000, 3000 s/mm^2 => 0, 1e9, 2e9, 3e9 s/m^2
    
    bvals = np.array([0, 1e9, 2e9, 3e9]) 
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    import inspect
    try:
        from mdt import simulations
        print(f"Sig simulate_signals: {inspect.signature(simulations.simulate_signals)}")
    except Exception as e:
        print(f"Inspect simulate_signals failed: {e}")

    # Test Stick Model
    print("\nTesting Stick Model (via Tensor)...")
    try:
        stick = mdt_adapter.get_mdt_model('Stick')
        print(f"Model instantiated: {stick}")
        print(f"dir(stick): {dir(stick)}")
        
        # Verify required params
        required = stick.get_free_param_names()
        print(f"Required parameters: {required}")
        
        # Standard Tensor params are typically: 'Tensor.d', 'Tensor.phi', 'Tensor.theta'
        # Or eigenvalues 'Tensor.l1', 'Tensor.l2', 'Tensor.l3'.
        # Let's define params based on what we see. 
        # For now, we'll try to simulate with some reasonable defaults if we knew the names.
        # But since we don't know exact names yet (introspection will reveal them), 
        # we'll construct a param dict dynamically or hardcode for 'Tensor'.
        
        params = {}
        if 'Tensor.d' in required: # d_parallel
             params['Tensor.d'] = 2e-9
        
        # Stick has 0 perpendicular diffusivity
        if 'Tensor.dperp0' in required:
             params['Tensor.dperp0'] = 0.0
        if 'Tensor.dperp1' in required:
             params['Tensor.dperp1'] = 0.0
             
        # S0
        if 'S0.s0' in required:
             params['S0.s0'] = 1.0

        # Orientation
        if 'Tensor.theta' in required:
            params['Tensor.theta'] = 0.0
        if 'Tensor.phi' in required:
            params['Tensor.phi'] = 0.0
        if 'Tensor.psi' in required:
            params['Tensor.psi'] = 0.0
            
        print(f"Using Params: {params}")
        
        signal = mdt_adapter.simulate_signal(stick, bvals, bvecs, params)
        print(f"Synthesized Signal (Stick): {signal}")
        
    except Exception as e:
        print(f"Stick Protocol Analysis Failed: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()

    # Test Ball Model
    print("\nTesting Ball Model...")
    try:
        ball = mdt_adapter.get_mdt_model('Ball')
        print(f"Model instantiated: {ball}")
        required = ball.get_free_param_names()
        print(f"Required parameters: {required}")
        
        params_ball = {}
        # Ball maps to Tensor with d_perp = d_par usually if using Tensor fallback
        # Or Ball specific params.
        # Logic in mdt_adapter: try get_component('composite_models', 'Ball') else Tensor.
        
        if 'S0.s0' in required:
            params_ball['S0.s0'] = 1.0
            
        if 'Ball.d' in required:
             params_ball['Ball.d'] = 2e-9
        elif 'Tensor.d' in required:
             # If it fell back to Tensor
             params_ball['Tensor.d'] = 2e-9
             params_ball['Tensor.dperp0'] = 2e-9
             params_ball['Tensor.dperp1'] = 2e-9
             params_ball['Tensor.theta'] = 0.0
             params_ball['Tensor.phi'] = 0.0
             params_ball['Tensor.psi'] = 0.0
             
        signal_ball = mdt_adapter.simulate_signal(ball, bvals, bvecs, params_ball)
        print(f"Synthesized Signal (Ball): {signal_ball}")

    except Exception as e:
        print(f"Ball Protocol Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    run_verification()
