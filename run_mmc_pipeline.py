import numpy as np
import json
import os
import sys
import subprocess
import meshio

# Add local path to find dmipy_jax
sys.path.append(os.getcwd())
from dmipy_jax.io.sci_head_loader import load_sci_head_mesh

def run_mmc_pipeline():
    print("MMC Pipeline: Initialization...")
    
    # ---------------------------------------------------------
    # 1. Load and Center Mesh (Must match previous logic)
    # ---------------------------------------------------------
    mesh_data = load_sci_head_mesh('data/SCI_headmodel/extracted/HeadMesh.mat')
    points = np.array(mesh_data['points'])
    cells = np.array(mesh_data['cells']['tetra']) # (N_cells, 4)
    tissues = np.array(mesh_data['cell_data']['tissue']).flatten() # (N_cells,)

    # CENTER THE MESH
    centroid = np.mean(points, axis=0)
    points = points - centroid
    print(f"Mesh Centered. Bounds: {np.min(points, axis=0)} to {np.max(points, axis=0)}")
    
    # Setup Source (Scalp Search) for Config
    scalp_indices = np.where(tissues == 1)[0]
    scalp_tetras = cells[scalp_indices]
    scalp_nodes = np.unique(scalp_tetras)
    scalp_points = points[scalp_nodes]
    
    # Find Anterior Source (Central Optode)
    local_src_idx = np.argmax(scalp_points[:, 1])
    src_pos = scalp_points[local_src_idx]
    print(f"Central Source: {src_pos}")

    # ---------------------------------------------------------
    # 1b. Define Kernel Flow Array (Hexagonal Rings)
    # ---------------------------------------------------------
    # Central Source + Detectors at various radii
    # Define ideal tangent plane positions relative to source
    # Rings: 3mm (6 dets), 5mm (6 dets), 10mm (6 dets), 25mm (6 dets)
    radii = [3.0, 5.0, 10.0, 25.0] 
    angles = np.radians(np.arange(0, 360, 60)) # 60 degree increments
    
    array_candidates = [src_pos] # Index 0 is Source
    
    # Simple Tangent Plane approximation (Normal ~ Y axis)
    # Basis vectors on tangent plane
    u = np.array([1.0, 0.0, 0.0]) # Lateral
    v = np.array([0.0, 0.0, 1.0]) # Superior-Inferior (approx Z)
    
    for r in radii:
        for theta in angles:
            # Ideal position on tangent plane
            pos_ideal = src_pos + r * (np.cos(theta)*u + np.sin(theta)*v)
            array_candidates.append(pos_ideal)
            
    # Snap Candidates to Surface (Neuronavigation)
    array_hardware = []
    array_indices = []
    for cand in array_candidates:
        dists = np.linalg.norm(scalp_points - cand, axis=1)
        nearest = np.argmin(dists)
        array_hardware.append(scalp_points[nearest])
        array_indices.append(scalp_nodes[nearest]) # Global Index
        
    array_hardware = np.array(array_hardware)
    print(f"Generated Kernel Flow Array: {len(array_hardware)} optodes.")
    
    # Export Hardware XDMF
    print("Exporting kf_array_hardware.xdmf...")
    mesh_hw = meshio.Mesh(
        array_hardware,
        [("vertex", np.arange(len(array_hardware)).reshape(-1, 1))],
        point_data={"type": [1] + [2]*(len(array_hardware)-1)} # 1=Source, 2=Detector
    )
    mesh_hw.write("kf_array_hardware.xdmf")

    # ---------------------------------------------------------
    # 2. Export .node and .elem (TetGen/MMC Format)
    # ---------------------------------------------------------
    # MMC requires explicit file inputs for robustness
    node_file = "sci_head.node"
    elem_file = "sci_head.elem"
    
    # 1. Write .node file
    # Format: <#nodes> <dim> <#attr> <#boundary_markers>
    #         <index> <x> <y> <z>
    print(f"Exporting: {node_file} ...")
    with open(node_file, 'w') as f:
        f.write(f"{len(points)} 3 0 0\n")
        # Optimization: use numpy savetxt?
        # Standard TetGen is 1-based index? MMC handles both, usually 1-based.
        # Let's use 1-based indices in the file for safety.
        node_indices = np.arange(1, len(points)+1).reshape(-1, 1)
        node_data = np.hstack([node_indices, points])
        np.savetxt(f, node_data, fmt="%d %.6f %.6f %.6f")

    # 2. Write .elem file
    # Format: <#elems> <nodes_per_elem> <#attr>
    #         <index> <n1> <n2> <n3> <n4> <label>
    print(f"Exporting: {elem_file} ...")
    with open(elem_file, 'w') as f:
        f.write(f"{len(cells)} 4 0\n") # 0 attributes in header? Label is usually 5th col?
        # Actually TetGen .elem is: index n1 n2 n3 n4 [attr]
        # MMC uses the 5th column as Tissue ID (Label).
        
        elem_indices = np.arange(1, len(cells)+1).reshape(-1, 1)
        # Cells are 0-based coming from meshio. Convert to 1-based.
        cells_1based = cells + 1
        tissues_col = tissues.reshape(-1, 1)
        
        elem_data = np.hstack([elem_indices, cells_1based, tissues_col])
        np.savetxt(f, elem_data, fmt="%d %d %d %d %d %d")
        
    # ---------------------------------------------------------
    # 3. Generate Config JSON (Referencing Mesh)
    # ---------------------------------------------------------
    # Find containing element for InitElem (Required by MMC)
    # Source is at src_pos (vertex). Find any cell containing this vertex.
    # src_idx is global index (0-based)
    # cells is (N, 4) 0-based
    # Find row where cells == local_src_idx (wait, src_idx is global)
    # We need the global index of the source node
    src_node_idx = scalp_nodes[local_src_idx]
    
    # Find any tetra containing src_node_idx
    # np.where returns (row_indices, col_indices)
    incident_cells = np.where(cells == src_node_idx)[0]
    if len(incident_cells) == 0:
        print("Error: Source node not found in any cell?")
        return
        
    # Pick the first one. 
    # MMC 1-based index
    init_elem = int(incident_cells[0]) + 1
    print(f"InitElem: {init_elem} (Cell Index {incident_cells[0]})")

    # Use Absolute Path for MeshID to avoid CWD issues
    # TetGen/MMC appends .node/.elem, so we provide the base path without extension
    mesh_base_abs = os.path.abspath("sci_head")
    print(f"Mesh Base Path: {mesh_base_abs}")

    jmesh = {
        "Mesh": {
            "MeshID": mesh_base_abs,
            "ElemFormat": "tet4",
            "InitElem": init_elem
        },
        "Session": {
            "ID": "sci_head_mmc",
            "Photons": 1e7
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5e-9,
            "Dt": 5e-9
        },
        "Optode": {
            "Source": {
                "Type": "pencil",
                "Pos": src_pos.tolist(),
                "Dir": [0.0, -1.0, 0.0]
            }
        },
        "Domain": {
            "Media": [
                {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},        # 0: Air
                {"mua": 0.015, "mus": 10.0, "g": 0.9, "n": 1.4},     # 1: Skin
                {"mua": 0.01, "mus": 15.0, "g": 0.9, "n": 1.4},      # 2: Skull
                {"mua": 0.002, "mus": 0.1, "g": 0.9, "n": 1.33},     # 3: CSF
                {"mua": 0.02, "mus": 20.0, "g": 0.9, "n": 1.4},      # 4: GM
                {"mua": 0.018, "mus": 20.0, "g": 0.9, "n": 1.4},     # 5: WM
                {"mua": 0.018, "mus": 20.0, "g": 0.9, "n": 1.4},     # 6+
                {"mua": 0.018, "mus": 20.0, "g": 0.9, "n": 1.4}
            ]
        }
    }
    
    json_path = "sci_head_mmc.json"
    print(f"Exporting MMC Config: {json_path} ...")
    with open(json_path, 'w') as f:
        json.dump(jmesh, f, indent=4)
        
    # ---------------------------------------------------------
    # 3. Execute MMC
    # ---------------------------------------------------------
    mmc_bin = "/home/mhough/dev/debian/mmc/bin/mmc"
    cmd = [mmc_bin, "-f", json_path, "-n", "10000000", "-b", "1"] 
    # MMC options: -n photons, -b boundary (1=ref, 0=abs)
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        # Run and capture output
        ret = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("MMC Success!")
        # print(ret.stdout) 
    except subprocess.CalledProcessError as e:
        print("MMC Failed!")
        print(e.stderr)
        return

    # ---------------------------------------------------------
    # 4. Parse Output & Convert to XDMF
    # ---------------------------------------------------------
    # Expect output file. Default is usually input filename + .dat or .mch
    # We specified "OutputFormat": "json" ?? MMC might ignore.
    # MMC usually outputs 'sci_head_mmc.dat' (fluence at nodes)
    
    out_file = "sci_head_mmc.dat" 
    if not os.path.exists(out_file):
        # check for .mch
        if os.path.exists("sci_head_mmc.mch"):
            out_file = "sci_head_mmc.mch"
        else:
            print("Output file not found. Listing dir:")
            print(os.listdir('.'))
            return

    print(f"Loading Output: {out_file}")
    # Load binary float data (usually 4-byte float per node)
    fluence = np.fromfile(out_file, dtype=np.float32)
    
    print(f"Fluence Shape: {fluence.shape}, Mesh Points: {points.shape}")
    
    # Reshape if time-resolved?
    # If we simulated 1 time gate (CW), it's (N_nodes,)
    
    # Log transform
    log_fluence = np.log10(np.clip(fluence, 1e-10, None))
    # Replace -inf with small value
    log_fluence[np.isneginf(log_fluence)] = -10.0
    
    # Normalize for Vis
    log_fluence -= np.max(log_fluence)
    
    # Write XDMF
    mesh_out = meshio.Mesh(
        points,
        [("tetra", cells)],
        point_data={"log_fluence_mmc": log_fluence},
        cell_data={"tissue": [tissues]}
    )
    mesh_out.write("kf_mmc_centered.xdmf")
    print("Generated: kf_mmc_centered.xdmf")

if __name__ == "__main__":
    run_mmc_pipeline()
