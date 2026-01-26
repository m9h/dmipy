
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import nibabel as nib
from wand_loader import WandLoader
from dmipy_jax.models.joint import JointInversionModel, JointInversionParameters
from dmipy_jax.models.epg import JAXEPG

# Set Default Device
# jax.config.update("jax_platform_name", "cpu") # Use GPU if available

def run_joint_fit():
    print("--- WAND Joint Inversion (mcDESPOT + qMT) ---")
    
    # 1. Load Data
    loader = WandLoader(base_dir="data/wand/sub-00395/ses-02")
    data_dict, proto = loader.load_mcdespot()
    
    spgr = data_dict['spgr'] # (X,Y,Z,8)
    ssfp = data_dict['ssfp'] # (X,Y,Z,16)
    qmt  = data_dict['qmt']  # (X,Y,Z,Nq)
    
    print(f"Data Loaded: SPGR {spgr.shape}, SSFP {ssfp.shape}, qMT {qmt.shape}")
    
    # 2. Extract ROI (Center Slice, White Matter-ish)
    # Mid-slice
    cx, cy, cz = spgr.shape[0]//2, spgr.shape[1]//2, spgr.shape[2]//2
    
    # Take a 3x3x3 block average to reduce noise for test
    roi_slice = (slice(cx-1, cx+2), slice(cy-1, cy+2), slice(cz-1, cz+2))
    
    s_spgr_roi = np.mean(spgr[roi_slice], axis=(0,1,2))
    s_ssfp_roi = np.mean(ssfp[roi_slice], axis=(0,1,2))
    s_qmt_roi  = np.mean(qmt[roi_slice], axis=(0,1,2))
    
    y_target = jnp.concatenate([s_spgr_roi, s_ssfp_roi, s_qmt_roi])
    print(f"ROI Signal (Example): {y_target[:5]} ... {y_target[-5:]}")
    
    # 3. Define Model and Loss
    model = JointInversionModel()
    
    # Parameters to Optimise (Packed Vector)
    # [f_mw, f_mm, T1_mw, T1_ie, T2_mw, T2_ie, k_mw, S0]
    # Scaling: T1/T2 in ms (or s?), S0 in au.
    # JointInversionModel expects T1/T2 in ms for Water, ms for MM?
    # qMT expects s^-1 inputs?
    # JointInversionModel implementation:
    #   R1_mw = 1 / (params.T1_mw * 1e-3) -> Input in ms
    #   R1_m = 1 / (params.T1_mm * 1e-3) -> Input in ms
    # So Inputs are in ms.
    
    def unpack(p):
        # Sigmoid constraints or exponential for positivity
        # Use simple absolute value or exp for now?
        # Better: use optax with bounds or parameter transformation.
        # Here we use transform: p is unconstrained.
        
        f_mw = jax.nn.sigmoid(p[0]) * 0.4 # Bound 0-0.4
        f_mm = jax.nn.sigmoid(p[1]) * 0.3 # Bound 0-0.3
        
        T1_mw = 100.0 + jax.nn.softplus(p[2]) * 800.0 # 100 - 900 ms
        T1_ie = 500.0 + jax.nn.softplus(p[3]) * 2000.0 # 500 - 2500 ms
        
        T2_mw = 5.0 + jax.nn.softplus(p[4]) * 40.0 # 5 - 45 ms
        T2_ie = 20.0 + jax.nn.softplus(p[5]) * 200.0 # 20 - 220 ms
        
        k_mw = jax.nn.softplus(p[6]) * 50.0 # 0 - 50 Hz
        
        S0 = jax.nn.softplus(p[7]) * 10000.0 # 0 - 10000 (Arbitrary scale)
        
        return JointInversionParameters(
            f_mw_water=f_mw,
            f_mm_total=f_mm,
            T1_mw=T1_mw, T1_ie=T1_ie,
            T2_mw=T2_mw, T2_ie=T2_ie,
            T1_mm=1000.0, # Fixed
            T2_mm=0.010,  # Fixed 10us in ms = 0.010 ms
            k_m_w=k_mw,
            S0=S0
        )
        
    def loss_fn(p):
        params = unpack(p)
        y_pred = model(params, proto)
        # MSE
        return jnp.mean((y_pred - y_target)**2)
        
    # 4. Optimizer Loop
    # Initialize
    # f_mw=0.15, f_mm=0.10, T1_mw=400, T1_ie=1000, T2_mw=20, T2_ie=80, k=10, S0=mean
    p0 = jnp.array([
        0.0, 0.0, # Sigmoid(0) = 0.5 -> 0.2, 0.15
        0.0, 0.0, # Softplus(0) ~ 0.69
        0.0, 0.0,
        0.0,
        jnp.log(jnp.max(y_target)/10000.0) # S0 init guess
    ])
    
    print("Starting Optimization...")
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(p0)
    
    @jax.jit
    def step(p, opt_st):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_st = optimizer.update(grads, opt_st)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss
        
    for i in range(500):
        p0, opt_state, loss_val = step(p0, opt_state)
        if i % 50 == 0:
            print(f"Iter {i}: Loss {loss_val:.2f}")
            
    # Report
    final_params = unpack(p0)
    print("\n--- Fit Results ---")
    print(f"MWF (Water): {final_params.f_mw_water:.3f}")
    print(f"MPF (Total): {final_params.f_mm_total:.3f}")
    print(f"T1 MW: {final_params.T1_mw:.1f} ms, T1 IE: {final_params.T1_ie:.1f} ms")
    print(f"T2 MW: {final_params.T2_mw:.1f} ms, T2 IE: {final_params.T2_ie:.1f} ms")
    print(f"Exchange k_mw: {final_params.k_m_w:.1f} Hz")
    print(f"S0: {final_params.S0:.1f}")

if __name__ == "__main__":
    run_joint_fit()
