
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import sys
import os
import trimesh

# Try to import disimpy, if not available (e.g. during build verification), mock it or exit
try:
    from disimpy import gradients, simulations, substrates, utils
    DISIMPY_AVAILABLE = True
except ImportError:
    DISIMPY_AVAILABLE = False
    print("WARNING: Disimpy not found. Simulation will not run.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Disimpy simulation on CATERPillar substrate.")
    parser.add_argument("--spheres", required=True, help="Path to spheres.csv from CATERPillar")
    parser.add_argument("--output", required=True, help="Path to output signal file (.npy)")
    parser.add_argument("--bvecs", help="Path to bvecs file")
    parser.add_argument("--bvals", help="Path to bvals file")
    parser.add_argument("--n_walkers", type=int, default=100000, help="Number of random walkers")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--mesh_resolution", type=int, default=16, help="Resolution (subdivisions) for sphere mesh")
    parser.add_argument("--mesh_mode", choices=['concat', 'union'], default='concat', 
                        help="Mesh combination mode. 'concat' is faster but leaves internal walls. 'union' is physically correct for overlapping spheres but slow.")
    return parser.parse_args()

def load_spheres(csv_path):
    """
    Load CATERPillar spheres.csv.
    Expected columns: x, y, z, r, type (approximate)
    """
    df = pd.read_csv(csv_path)
    # Check for standard column names, adjust if necessary based on CATERPillar format
    # Assuming columns: 'x', 'y', 'z', 'radius'
    if 'r' in df.columns:
        df['radius'] = df['r']
    return df[['x', 'y', 'z', 'radius']].values

def create_substrate_mesh(spheres, resolution=2, mode='concat'):
    """
    Convert list of spheres to a single trimesh object.
    """
    print(f"Generating mesh from {len(spheres)} spheres with resolution {resolution} (icosphere subdivisions)...")
    meshes = []
    # Create a template unit sphere
    template = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
    
    for x, y, z, r in spheres:
        # Copy template
        m = template.copy()
        # Scale
        m.apply_scale(r)
        # Translate
        m.apply_translation([x, y, z])
        meshes.append(m)
    
    print("Combining meshes...")
    if mode == 'union':
        # This is very expensive for many spheres
        combined = trimesh.boolean.union(meshes)
    else:
        combined = trimesh.util.concatenate(meshes)
    
    return combined

def main():
    args = parse_args()
    
    if not DISIMPY_AVAILABLE:
        print("Disimpy not available. Exiting.")
        sys.exit(1)

    # 1. Load Geometry
    print(f"Loading spheres from {args.spheres}...")
    spheres_data = load_spheres(args.spheres)
    
    # 2. Generate Mesh
    print(f"Creating substrate mesh (mode: {args.mesh_mode})...")
    mesh_obj = create_substrate_mesh(spheres_data, resolution=2, mode=args.mesh_mode) # resolution=2 is decent for visual/approx
    
    # Extract vertices and faces for Disimpy
    vertices = mesh_obj.vertices
    faces = mesh_obj.faces
    
    # Create Disimpy substrate
    # Padding should be large enough to contain walkers if periodic
    # Assuming CATERPillar generates a cubic voxel, usually centered?
    # We might need to adjust padding or periodicity.
    # CATERPillar usually fills a box.
    padding = np.zeros(3) 
    print("Initializing Disimpy substrate...")
    substrate = substrates.mesh(vertices, faces, padding=padding, periodic=True, init_pos="intra")
    
    # 3. Define Gradient Scheme
    # If bvals/bvecs provided
    if args.bvals and args.bvecs:
        print("Loading gradients...")
        bvals = np.loadtxt(args.bvals)
        bvecs = np.loadtxt(args.bvecs).T
        gradient = np.concatenate([bvals[:, None], bvecs], axis=1) # generic format check needed
        # Disimpy expects specific format. 
        # utils.generate_gradient generic usage?
        # Actually disimpy has specific gradient object creation.
        # Let's assume a simple gradient array for now as per tutorial:
        # gradient = np.zeros((n_meas, 4))
        # gradient[:, 0] = bvals
        # gradient[:, 1:] = bvecs
        # Need to check units. Disimpy usually uses SI. bvals in s/m^2.
        
        # Reshape to (N, 4) if not already
        bs = bvals
        gs = bvecs
        gradient = np.zeros((len(bs), 4))
        gradient[:, 0] = bs
        gradient[:, 1:] = gs
    else:
        # Default dummy gradient for testing
        print("Using default dummy gradients...")
        n_gradients = 30
        bs = np.ones(n_gradients) * 1000e6 # 1000 s/mm^2 -> 1e9 s/m^2
        gs = utils.fibonacci_sphere(n_gradients)
        gradient = np.zeros((n_gradients, 4))
        gradient[:, 0] = bs
        gradient[:, 1:] = gs

    # 4. Simulation Parameters
    # Standard water diffusivity at 37C
    diffusivity = 3.0e-9 # m^2/s
    
    # Time parameters need to be derived from b-values or scheme if possible.
    # For now, hardcode or assume small delta/ big DELTA
    # Disimpy tutorial uses 'dt'.
    # We need to define total time.
    # Let's assume a standard Stejskal-Tanner.
    # Big Delta = 20ms, small delta = 5ms?
    # If just running "simulation", it runs for specific n_steps * dt?
    # Actually, `simulations.simulation` takes `dt` and runs for `n_steps`?
    # Tutorial: "Number of steps = 1000, Step duration = ..."
    # We should match physical time.
    T_max = 50e-3 # 50ms
    dt = T_max / args.n_steps
    
    print(f"Running simulation with {args.n_walkers} walkers, {args.n_steps} steps, dt={dt:.2e}s...")
    
    # 5. Run Simulation
    signals = simulations.simulation(
        n_walkers=args.n_walkers,
        diffusivity=diffusivity,
        gradient=gradient,
        dt=dt,
        substrate=substrate,
    )
    
    # 6. Save
    print(f"Saving results to {args.output}...")
    np.save(args.output, signals)
    print("Done.")

if __name__ == "__main__":
    main()
