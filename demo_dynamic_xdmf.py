import meshio
import numpy as np
from dmipy_jax.io.sci_head_loader import load_sci_head_mesh
import time

def generate_dynamic_demo(steps=20):
    """
    Generates a synthetic 'dynamic conductivity wave' on the SCI Head Model.
    Saves the result as 'sci_head_dynamic.xdmf' with 20 time steps.
    """
    
    # 1. Load the 15M element mesh (this takes memory!)
    print("Loading SCI Head Mesh...")
    # Using our loader to get points (we only need points to generate the wave)
    # But meshio writer needs cells too. 
    # To be efficient, let's load via meshio directly if possible to keep it in meshio format
    # OR use our loader data and reconstruct meshio object.
    
    # Direct meshio read of VTK is fastest for just pass-through
    vtk_path = "data/SCI_headmodel/extracted/HeadMesh.vtk"
    
    try:
        mesh = meshio.read(vtk_path)
    except Exception as e:
        print(f"Failed to read VTK: {e}")
        return

    print(f"Mesh loaded. Points: {len(mesh.points)}")
    
    # 2. Setup Time Series Writer
    # We will write 'sci_head_dynamic.xdmf' and 'sci_head_dynamic.h5'
    xdmf_filename = "sci_head_dynamic.xdmf"
    
    print(f"Starting specific writer for {xdmf_filename}...")
    
    # Create the writer context
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        
        # Write geometry and topology ONCE
        writer.write_points_cells(mesh.points, mesh.cells)
        
        # Center of the head (approx)
        center = np.mean(mesh.points, axis=0)
        
        # 3. Time Loop
        for t in range(steps):
            print(f"Generating Step {t+1}/{steps}...")
            
            # Create a simple travelling wave: sin(k*r - w*t)
            # r is distance from center
            # k = wave number, w = frequency
            
            dist = np.linalg.norm(mesh.points - center, axis=1)
            
            # Simulate a "Conductivity Perturbation" propagating outward
            # Base conductivity 1.0 + perturbation
            wave = np.sin(0.1 * dist - 0.5 * t)
            
            # Decay with distance to look more realistic
            amplitude = np.exp(-0.02 * dist)
            
            field = 1.0 + amplitude * wave
            
            # Write only the data for this time step
            # Note: We write it as POINT data for smooth visualization
            writer.write_data(
                t, 
                point_data={"conductivity_wave": field}
            )
            
    print(f"Done! Saved {xdmf_filename}")

if __name__ == "__main__":
    generate_dynamic_demo(steps=10) # 10 steps for demo speed
