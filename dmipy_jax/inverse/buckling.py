
import os
import subprocess
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Optional, Union, Tuple

class BucklingSimulator:
    """
    Python wrapper for the C++ 'Brains' buckled growth simulator.
    
    Manages I/O with the binary:
    - Writes `growth_map.txt` (nodal growth values).
    - Runs `./Brains`.
    - Reads `stress_map.txt` (elemental stress tensors).
    """
    
    def __init__(self, binary_path: Union[str, Path], work_dir: Optional[Union[str, Path]] = None):
        self.binary_path = Path(binary_path).absolute()
        if work_dir is None:
            self.work_dir = self.binary_path.parent
        else:
            self.work_dir = Path(work_dir).absolute()
            
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Simulator binary not found at {self.binary_path}")
            
    def run_simulation(self, growth_map: np.ndarray) -> np.ndarray:
        """
        Runs the simulation with the given growth map.
        
        Args:
            growth_map: (N_nodes,) array of growth multipliers.
            
        Returns:
            stress_map: (N_elements, 3, 3) stress tensors.
        """
        # 1. Write growth map
        growth_file = self.work_dir / "growth_map.txt"
        np.savetxt(growth_file, growth_map, fmt='%.8f')
        
        # 2. Run binary
        # We assume the binary needs to be run from its directory to find .mesh files etc.
        # Capturing output to avoid clutter, but raising on error.
        try:
            # Cleanup previous output
            output_file = self.work_dir / "stress_map.txt"
            if output_file.exists():
                output_file.unlink()
                
            subprocess.run(
                [str(self.binary_path)], 
                cwd=str(self.work_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Simulation failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            
        # 3. Read stress map
        if not output_file.exists():
            raise FileNotFoundError("Simulation finished but stress_map.txt was not generated.")
            
        # Format: Txx Txy Txz Tyx Tyy Tyz Tzx Tzy Tzz
        stress_data = np.loadtxt(output_file)
        
        # Reshape to (Ne, 3, 3)
        # loadtxt returns (Ne, 9)
        stress_tensors = stress_data.reshape(-1, 3, 3)
        
        return stress_tensors

    def get_mesh_info(self) -> Tuple[int, int]:
        """
        Reads mesh file to determine N_nodes and N_elements.
        Helper to initialize growth map size.
        """
        mesh_file = self.work_dir / "brains.mesh" # Hardcoded in C++
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found at {mesh_file}")
            
        with open(mesh_file, 'r') as f:
            # Format: 'nn' then 'ne' then 'nf'
            # But line 1 is 'nn'.
            # Brains.cpp: filu >> inpa; int nn = atoi(inpa);
            # It seems the file starts directly with the number.
            
            # Let's inspect the file structure from Brains.cpp logic:
            # filu >> inpa; (gets nn)
            # then nn lines of nodes.
            # filu >> inpa; (gets ne)
            # then ne lines of tets.
            
            # Since reading token by token is annoying in python without regex or split,
            # Let's try to parse simply.
            
            content = f.read().split()
            nn = int(content[0])
            
            # Skip nn nodes (each node has x, y, z -> 3 numbers)
            # Pointer moves 1 + nn*3
            ptr = 1 + nn*4 # node index + x + y + z? 
            # Brains.cpp lines 52: filu >> inpa >> inpb >> inpc; Ut0[i] ...
            # Wait, line 47: filu >> inpa (nn)
            # Loop i=0 to nn: filu >> inpa >> inpb >> inpc. (3 floats)
            # So 1 token for count, then nn * 3 tokens.
            
            # Actually line 52 inside loop: filu >> inpa >> inpb >> inpc;
            # So yes, 3 numbers per node.
            
            ptr_ne = 1 + nn*3
            ne = int(content[ptr_ne])
            
            return nn, ne
