import numpy as np
import meshio
import os
import sys
import jax.numpy as jnp

# Add local path to find dmipy_jax
sys.path.append(os.getcwd())

from dmipy_jax.io.sci_head_loader import load_sci_head_mesh

def demo_kernel_flow_v2():
    print("Kernel Flow 2.0: Loading SCI Head Model (Full Topology)...")
    
    # 1. Load Data
    mesh_data = load_sci_head_mesh('data/SCI_headmodel/extracted/HeadMesh.mat')
    points = np.array(mesh_data['points'])
    cells = [('tetra', np.array(mesh_data['cells']['tetra']))]
    tissues = np.array(mesh_data['cell_data']['tissue']) 

    # CENTER THE MESH (Crucial for alignment and camera)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    print(f"Mesh Centered. New Bounds: {np.min(points, axis=0)} to {np.max(points, axis=0)}")

    # 2. Setup Optodes (Neuronavigation)
    # CRITICAL: Restrict search to SCALP nodes (Tissue 1) to prevents sensors "sinking" into the brain.
    print("Extracting Scalp Surface nodes for sensor placement...")
    scalp_cell_indices = np.where(tissues == 1)[0]
    scalp_tetras = cells[0][1][scalp_cell_indices] # Access the numpy array of tetras
    scalp_node_indices = np.unique(scalp_tetras)
    # Mapping global index -> local scalp index not needed if we carry global indices
    
    # Source: Frontal Scalp (Max Y among Scalp Nodes)
    # We search only within scalp_node_indices
    scalp_points = points[scalp_node_indices]
    
    # Find index relative to 'scalp_points' array
    local_src_idx = np.argmax(scalp_points[:, 1]) 
    # Map back to global 'points' index
    src_idx = scalp_node_indices[local_src_idx]
    src_pos = points[src_idx]
    
    # Detector: 30mm Lateral (+X) on Scalp
    target_pos = src_pos + np.array([30.0, 0.0, 0.0])
    
    # Search only Scalp Nodes
    dists = np.linalg.norm(scalp_points - target_pos, axis=1)
    local_det_idx = np.argmin(dists)
    det_idx = scalp_node_indices[local_det_idx]
    det_pos = points[det_idx]
    
    print(f"Source (Scalp): {src_pos}")
    print(f"Detector (Scalp): {det_pos} (Dist: {np.linalg.norm(src_pos - det_pos):.2f} mm)")
    
    # 3. Physics Parameters
    D = 0.33  # mm
    v = 0.214 # mm/ps
    mu_a = 0.01 # mm^-1
    mu_eff = np.sqrt(3 * mu_a * (1/(3*D)) * mu_a) # Approx sqrt(mu_a/D) ? No, sqrt(3*mu_a*mu_s')
    # Use standard mu_eff ~ 0.17 for visualization decay
    mu_eff_vis = 0.15 
    
    # 4. Compute Static Anatomy Fields (Banana)
    print("Computing Static Sensitivity (Banana)...")
    dist_src = np.linalg.norm(points - src_pos, axis=1) + 1e-3
    dist_det = np.linalg.norm(points - det_pos, axis=1) + 1e-3
    
    # CW fields ~ 1/r * exp(-mu r)
    phi_src_cw = (1.0 / dist_src) * np.exp(-mu_eff_vis * dist_src)
    phi_det_cw = (1.0 / dist_det) * np.exp(-mu_eff_vis * dist_det)
    
    sensitivity = phi_src_cw * phi_det_cw
    # Normalize and Log
    sens_norm = sensitivity / np.max(sensitivity)
    log_sensitivity = np.log10(np.clip(sens_norm, 1e-9, 1.0))
    
    # Export 1: kf_anatomy_centered.xdmf (Context)
    # Includes Tissues (Cell Data) and Sensitivity (Point Data)
    print("Exporting kf_anatomy_centered.xdmf...")
    mesh_static = meshio.Mesh(
        points,
        cells,
        point_data={"log_sensitivity": log_sensitivity},
        cell_data={"tissue": [tissues]}
    )
    mesh_static.write("kf_anatomy_centered.xdmf")
    
    # Export 2: kf_optodes_centered.xdmf (Hardware)
    print("Exporting kf_optodes_centered.xdmf...")
    optode_points = np.array([src_pos, det_pos])
    # Define as vertices
    mesh_optodes = meshio.Mesh(
        optode_points,
        [("vertex", np.array([[0], [1]]))],
        point_data={"type": [1, 2]}
    )
    mesh_optodes.write("kf_optodes_centered.xdmf")
    
    # 5. Compute Dynamic Pulse
    print("Computing Dynamic Pulse...")
    t_start = 50.0
    t_end = 2000.0
    steps = 20
    time_points = np.linspace(t_start, t_end, steps)
    
    filename_pulse = "kf_pulse_centered.xdmf"
    with meshio.xdmf.TimeSeriesWriter(filename_pulse) as writer:
        writer.write_points_cells(points, cells)
        
        for k, t in enumerate(time_points):
            if k % 5 == 0:
                print(f"Step {k}/{steps}: t={t:.1f} ps")
                
            # Green's Function Pulse
            sigma = np.sqrt(4 * D * v * t)
            amplitude = (4 * np.pi * D * v * t)**(-1.5)
            spatial_term = np.exp( - (dist_src**2) / (4 * D * v * t) )
            absorption_term = np.exp( - mu_a * v * t )
            
            fluence = amplitude * spatial_term * absorption_term
            
            # Normalize step-wise for visualization "glow"
            max_f = np.max(fluence)
            if max_f > 0:
                fluence_norm = fluence / max_f
            else:
                fluence_norm = fluence
            
            log_fluence = np.log10(np.clip(fluence_norm, 1e-6, 1.0))
            
            writer.write_data(t, point_data={"log_photon_density": log_fluence})
            
    print("Kernel Flow 2.0 Simulation Complete.")
    print("Generated: kf_anatomy_centered.xdmf, kf_pulse_centered.xdmf, kf_optodes_centered.xdmf")

if __name__ == "__main__":
    demo_kernel_flow_v2()
