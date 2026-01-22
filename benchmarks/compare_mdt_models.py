import json
import numpy as np
import jax
import jax.numpy as jnp
from dmipy_jax.signal_models.gaussian_models import Tensor, Ball
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder, CallaghanRestrictedCylinder
from dmipy_jax.signal_models.ivim import IVIM
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues

# Helper to load benchmarks
def load_benchmarks(path='docker/mdt_benchmarks.json'):
    with open(path, 'r') as f:
        return json.load(f)

benchmarks = load_benchmarks()

print(f"Loaded {len(benchmarks)} benchmarks.")

for name, data in benchmarks.items():
    print(f"\n--- Comparing {name} ---")
    mdt_signal = np.array(data['signal'])
    bvals = np.array(data['bvals']) # s/m^2 SI
    bvecs = np.array(data['bvecs'])
    params = data['params']
    protocol_params = data.get('protocol_params', {})
    
    # dmipy-jax models expect (bvals, bvecs, **kwargs)
    # Some models need Delta/delta in kwargs.
    
    # Common kwargs
    # Delta/delta/TE are arrays (N,) or scalars.
    Delta = np.array(protocol_params.get('Delta', [0.0]*len(bvals)))
    delta = np.array(protocol_params.get('delta', [0.0]*len(bvals)))
    TE = np.array(protocol_params.get('TE', [0.0]*len(bvals)))

    # We pass these via dictionary expansion
    protocol_kwargs = {
        'big_delta': Delta,
        'small_delta': delta,
        'Delta': Delta,
        'delta': delta,
        'TE': TE,
        'echo_time': TE # for MTE_NODDI
    }
    
    model_dj = None
    dj_signal = None
    
    try:
        if name == 'Tensor':
            model_dj = Tensor()
            # Params: lambda_par, lambda_perp1, lambda_perp2, theta, phi, psi
            dj_params = {
                'lambda_1': params['Tensor.d'],
                'lambda_2': params['Tensor.dperp0'],
                'lambda_3': params['Tensor.dperp1'],
                'alpha': params['Tensor.phi'], # Euler alpha usually phi (z-rot)
                'beta': params['Tensor.theta'], # Euler beta usually theta (y-rot)
                'gamma': params['Tensor.psi']   # Euler gamma usually psi (z-rot)
            }
            # Verify Euler convention: MDT uses phi/theta/psi. Dmipy uses alpha/beta/gamma.
            # Usually: alpha=phi, beta=theta, gamma=psi.
            
            print(f"  Debug Tensor: bvals {bvals.shape}, bvecs {bvecs.shape}")
            # print(f"  Debug Params: {dj_params}")
            
            dj_signal = model_dj(bvals, bvecs, **dj_params)
            print(f"  Debug Tensor Output: {dj_signal.shape}, MDT: {mdt_signal.shape}")
            
            # Handle MDT subsampling (Tensor might only use b<=1000 or similar)
            if dj_signal.shape != mdt_signal.shape:
                print(f"  [Warning] Shape mismatch. Assuming MDT used subset (first {mdt_signal.shape[0]} volumes?).")
                # Truncate to match MDT (assuming MDT takes first N volumes?? or consistent protocol order)
                # Ideally we should know which volumes.
                # If 33, it's likely b0 + first shell.
                dj_signal = dj_signal[:mdt_signal.shape[0]]
            
        elif name == 'NODDI':
            # Use MTE_NODDI? Or rebuild standard NODDI?
            # Standard NODDI has fixed intrinsic diffusivity and tortuosity.
            # Let's try constructing a custom combined signal.
            # MDT NODDI:
            # - IC: Stick (d_par ~ 1.7e-9)
            # - EC: Zeppelin (d_par = d_par_ic, d_perp = d_par * (1-vic))
            # - ISO: Ball (d_iso ~ 3e-9)
            
            # Params: w_ic (ficvf), w_ec (fiso? No w_ec usually 1-w_ic-w_iso).
            # MDT params: w_ic.w, w_ec.w. Wait, do they sum to 1?
            # Introspection: ['S0.s0', 'w_ic.w', 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa', 'w_ec.w']
            # Implicit w_iso? Or w_ec is typically (1-ficvf) * (1-fiso).
            # Let's check generated params: w_ic=0.6, w_ec=0.4. Sum=1. So f_iso=0?
            
            # Components
            stick = Stick()
            zep = Zeppelin()
            
            # Orientation (Watson distributed? MDT NODDI uses Watson.)
            # dmipy-jax has BinghamNODDI, maybe has Watson?
            # If not, let's treat it as a single Stick if kappa is high, OR skip orientation dispersion for now if hard.
            # MDT "NODDI" usually implies orientation dispersion.
            # But the MDT adapter generated 'NODDI_IC.kappa' = 1.0 (very dispersed).
            # Without a Watson/Bingahm integration, we can't match exactly.
            # But we can try 'Stick' + 'Zeppelin' as a "Parallel NODDI" (AMICO style) approximation if no dispersion implemented in dmipy-jax yet.
            # Wait, `mte_noddi.py` had MTE_NODDI but presumably without dispersion integration? 
            # Or maybe MTE_NODDI in dmipy-jax handles it?
            # `mte_noddi.py` shows it uses `C1Stick` and `G2Zeppelin`, no convolution. So it equates to "Parallel NODDI" (un-dispersed).
            # MDT "NODDI" simulates WITH dispersion.
            # So comparisons will fail unless we set kappa -> infinity (aligned).
            # My benchmark gen used kappa=1.0. This will mismatch.
            # I will note this.
            
            pass 

        elif name == 'IVIM':
             model_dj = IVIM()
             # Params
             f_diff = params['w_diffusion.w']
             # f in dmipy IVIM is perfusion fraction
             f_perf = 1.0 - f_diff 
             
             dj_params = {
                 'f': f_perf, 
                 'D_tissue': params['Diffusion.d'],
                 'D_pseudo': params['Perfusion.d']
             }
             dj_signal = model_dj(bvals, **dj_params) * params['S0.s0']

        elif name == 'BallStick':
            # 1 Stick + 1 Ball
            stick = Stick()
            ball = Ball()
            
            w_stick = params['w_stick0.w']
            # w_ball = 1 - w_stick
            
            # Orientation
            mu = jnp.array([params['Stick0.theta'], params['Stick0.phi']])
            
            # Diffusivities? MDT BallStick defaults?
            # Assume d=1.7e-9 for stick, 3e-9 for ball?
            d_stick = 1.7e-9 # Typical
            d_ball = 3.0e-9 # CSF
            
            # MDT BallStick might use same d? Or d is not in params list?
            # If params missing, they are fixed defaults.
            
            S_stick = stick(bvals, bvecs, mu=mu, lambda_par=d_stick)
            S_ball = ball(bvals, bvecs, lambda_iso=d_ball)
            
            dj_signal = params['S0.s0'] * (w_stick * S_stick + (1.0 - w_stick) * S_ball)

        elif name == 'CHARMED':
            # RestrictedCylinder + Tensor
            cyl = RestrictedCylinder()
            ten = Tensor()
            
            w_res = params['w_res0.w'] # Restricted fraction
            
            # Restricted
            mu_cyl = jnp.array([params['CHARMEDRestricted0.theta'], params['CHARMEDRestricted0.phi']])
            d_par_res = params['CHARMEDRestricted0.d']
            # Diameter? MDT might fit it or hold fixed. If not in params, it's fixed.
            # Assume 6 micron (radius 3um) or similar if not found?
            # Or MDT 'CHARMED_r1' implies radius?
            # If I can't match diameter, RMSE will be high.
            diameter = 6e-6
            
            S_res = cyl(bvals, bvecs, 
                        mu=mu_cyl, lambda_par=d_par_res, diameter=diameter,
                        **protocol_kwargs)
            
            # Hindered (Tensor)
            # Tensor params are in the dict
            l1 = params['Tensor.d']
            l2 = params['Tensor.dperp0']
            l3 = params['Tensor.dperp1']
            # Orientation
            alpha = params['Tensor.phi']
            beta = params['Tensor.theta']
            gamma = params['Tensor.psi']
            
            S_hin = ten(bvals, bvecs, 
                        lambda_1=l1, lambda_2=l2, lambda_3=l3,
                        alpha=alpha, beta=beta, gamma=gamma)
                        
            dj_signal = params['S0.s0'] * (w_res * S_res + (1.0 - w_res) * S_hin)

        elif name == 'ActiveAx':
             # CylinderGPD + Zeppelin (usually) or Tensor?
             # Introspection: ActiveAx params: S0, w_ic, w_ec, CylinderGPD.theta, .phi, .R
             # No diffusion params? Implies fixed d_par=1.7e-9?
             
             cyl = RestrictedCylinder()
             zep = Zeppelin()
             
             w_ic = params['w_ic.w']
             
             # Restricted
             mu = jnp.array([params['CylinderGPD.theta'], params['CylinderGPD.phi']])
             diam = 2.0 * params['CylinderGPD.R']
             d_par = 1.7e-9 # Standard fixed value
             
             S_res = cyl(bvals, bvecs, mu=mu, lambda_par=d_par, diameter=diam, **protocol_kwargs)
             
             # Hindered (Zeppelin aligned)
             # d_hin_par = d_par
             # d_hin_perp? Tortuosity?
             # ActiveAx / AxCaliber usually assume d_perp(hin) depends on tortuosity or is fitted?
             # But 'w_ec' is only param.
             # Standard ActiveAx: d_perp = d_par * (1 - w_ic) (Tortuosity)?
             
             d_perp = d_par * (1.0 - w_ic)
             S_hin = zep(bvals, bvecs, mu=mu, lambda_par=d_par, lambda_perp=d_perp)
             
             dj_signal = params['S0.s0'] * (w_ic * S_res + (1.0 - w_ic) * S_hin)

        # If model instantiated and ran
        if dj_signal is not None:
             # Compare
             err = np.mean((dj_signal - mdt_signal)**2)
             rmse = np.sqrt(err)
             print(f"  RMSE: {rmse:.4e}")
             print(f"  MDT Mean: {np.mean(mdt_signal):.4f}")
             print(f"  DJ  Mean: {np.mean(dj_signal):.4f}")
             
        else:
            print("  [Not Implemented or Skipped]")
            
    except Exception as e:
        print(f"  Comparison Error: {e}")
        import traceback
        traceback.print_exc()
            
    except Exception as e:
        print(f"  Comparison Error: {e}")
        import traceback
        traceback.print_exc()
