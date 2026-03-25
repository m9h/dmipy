import subprocess
import os
import h5py
import numpy as np

class KomaSimulator:
    def __init__(self, julia_binary="/home/mhough/.juliaup/bin/julia", script_path=None):
        self.julia_binary = julia_binary
        if script_path is None:
            # Default to adjacent file
            self.script_path = os.path.join(os.path.dirname(__file__), "run_koma.jl")
        else:
            self.script_path = script_path
            
    def generate_ground_truth(self, seq_path, output_path, D=1e-9):
        """
        Runs the Julia script to generate KomaMRI ground truth.
        """
        if not os.path.exists(self.julia_binary):
            raise FileNotFoundError(f"Julia binary not found at {self.julia_binary}")
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"Julia script not found at {self.script_path}")
            
        cmd = [
            self.julia_binary,
            self.script_path,
            "--seq_path", os.path.abspath(seq_path),
            "--output_path", os.path.abspath(output_path),
            "--diffusivity", str(D)
        ]
        
        print(f"Running KomaMRI wrapper: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("KomaMRI Failed:")
            print(result.stderr)
            raise RuntimeError("KomaMRI simulation failed.")
            
        print("KomaMRI Output:")
        print(result.stdout)
        
        return output_path

