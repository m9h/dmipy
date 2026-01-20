
"""
Demo: Visualizing BigMac ODFs with FURY.

This script demonstrates how to:
1. Load BigMac data using the new `load_bigmac_mri`.
2. Compute simple ODFs (using DIPY CSD as a placeholder recon).
3. Visualize the ODFs using FURY's actor.odf_slicer.

Requirements:
    - fury
    - dipy
"""

import numpy as np
import matplotlib.pyplot as plt
from dmipy_jax.io.datasets import load_bigmac_mri
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.data import get_sphere
from fury import window, actor, ui

def main():
    print("Loading BigMac Data...")
    try:
        # Load small ROI
        sl = (slice(100, 110), slice(100, 110), slice(100, 101))
        data_dict = load_bigmac_mri(voxel_slice=sl)
        
        dwi = np.array(data_dict['dwi'])
        scheme = data_dict['scheme']
        bvals = np.array(scheme.bvalues)
        bvecs = np.array(scheme.gradient_directions)
        
        # Convert Scheme to DIPY GradientTable
        from dipy.core.gradients import gradient_table
        # bvals in SI? DIPY expects s/mm^2 usually for auto_response?
        # Check units. If bvals > 10000 -> SI.
        if np.max(bvals) > 10000:
            bvals_dipy = bvals / 1e6
        else:
            bvals_dipy = bvals
            
        gtab = gradient_table(bvals_dipy, bvecs)
        
        # Mask
        mask = np.array(data_dict['mask']) if data_dict['mask'] is not None else np.ones(dwi.shape[:-1])
        
    except FileNotFoundError:
        print("BigMac data not found.")
        return

    print("Reconstructing ODFs (CSD)...")
    # Auto response function from valid data
    # We need a shell. 
    # BigMac has multiple shells. Let's pick b=~4000 or similar if available, or just use all.
    # auto_response checks for single shell usually.
    
    # Placeholder: Just assume we have data and run simple CSD.
    # Real pipeline would be more robust.
    
    response, ratio = auto_response_ssst(gtab, dwi, roi_radii=1, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit = csd_model.fit(dwi, mask=mask)
    
    odf = csd_fit.odf(get_sphere('symmetric724'))
    
    print("Visualizing with FURY...")
    
    # ODF Slicer
    # ODFs are (X, Y, Z, N_dirs)
    scene = window.Scene()
    
    # Add ODF actor
    odf_actor = actor.odf_slicer(odf, sphere=get_sphere('symmetric724'), scale=0.5, colormap='plasma')
    scene.add(odf_actor)
    
    # Add T1 slice if available
    if data_dict['T1'] is not None:
        t1 = np.array(data_dict['T1'])
        t1_actor = actor.slicer(t1)
        t1_actor.display(z=0)
        scene.add(t1_actor)
        
    print("Interactive window opening. Close to finish.")
    window.show(scene)
    
    # Save snapshot
    window.record(scene, out_path='bigmac_fury_odf.png', size=(600, 600))
    print("Saved bigmac_fury_odf.png")

if __name__ == "__main__":
    main()
