import meshio
import numpy as np

def analyze_vtk(file_path):
    print(f"Reading {file_path} with meshio...")
    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        print(f"Failed to read with meshio: {e}")
        return

    print("\n--- Mesh Analysis Report ---")
    print(f"Number of Points: {len(mesh.points)}")
    print(f"Point Dtype: {mesh.points.dtype}")
    print(f"Point Range: {mesh.points.min(axis=0)} to {mesh.points.max(axis=0)}")
    
    print("\n--- Cells ---")
    total_cells = 0
    for cell_block in mesh.cells:
        print(f"Type: {cell_block.type}, Count: {len(cell_block.data)}")
        total_cells += len(cell_block.data)
    print(f"Total Cells: {total_cells}")

    print("\n--- Point Data (Node Attributes) ---")
    if not mesh.point_data:
        print("None")
    else:
        for name, data in mesh.point_data.items():
            print(f"Name: '{name}', Shape: {data.shape}, Dtype: {data.dtype}")
            print(f"  Range: {data.min()} to {data.max()}")

    print("\n--- Cell Data (Element Attributes) ---")
    if not mesh.cell_data:
        print("None")
    else:
        for name, data_list in mesh.cell_data.items():
            # data_list corresponds to cell_blocks
            print(f"Name: '{name}'")
            for i, data in enumerate(data_list):
                print(f"  Block {i}: Shape: {data.shape}, Dtype: {data.dtype}")
                # Analyze labels
                if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating):
                     unique_vals = np.unique(data)
                     if len(unique_vals) < 20:
                         print(f"    Unique Values: {unique_vals}")
                     else:
                         print(f"    Range: {data.min()} to {data.max()}")

    print("\n--- Field Data (Global Attributes) ---")
    if not mesh.field_data:
        print("None")
    else:
        for name, data in mesh.field_data.items():
            print(f"Name: '{name}', Data: {data}")

if __name__ == "__main__":
    analyze_vtk("data/SCI_headmodel/extracted/HeadMesh.vtk")
