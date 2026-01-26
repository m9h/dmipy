
import os
import subprocess
import tempfile
import shutil
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from dmipy_jax.simulation.sphere_sdf import MultiSphereSDF

class CATERPillarOracle:
    """
    Wrapper for the CATERPillar substrate generator.
    Handles configuration generation, execution, and parsing of sphere data.
    """
    def __init__(self, binary_path: str = "/home/mhough/dev/dmipy/vendor/CATERPillar/Caterpillar"):
        self.binary_path = binary_path
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"CATERPillar binary not found at {self.binary_path}. Please compile it.")

    def _write_config(self, config: Dict[str, Any], filepath: str):
        """Writes the configuration dictionary to a CATERPillar-compatible file."""
        with open(filepath, 'w') as f:
            for key, value in config.items():
                if key == "vox_sizes":
                    f.write("<vox_sizes>\n")
                    # Handle single value or list
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            f.write(f"{v}\n")
                    else:
                        f.write(f"{value}\n")
                    f.write("</vox_sizes>\n")
                else:
                    f.write(f"{key} {value}\n")
            f.write("<END>\n")

    def generate(self, config: Dict[str, Any], output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Runs CATERPillar with the given configuration.
        
        Args:
            config: Dictionary of parameters.
            output_dir: Directory to save outputs. If None, uses a temporary directory.
            
        Returns:
            pd.DataFrame: DataFrame containing sphere data (x, y, z, radius, type, id).
        """
        # Ensure output_dir exists or create temp
        temp_dir = False
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="caterpillar_run_")
            temp_dir = True
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        config_path = os.path.join(output_dir, "config.conf")
        
        # Enforce output directory in config
        config["data_directory"] = output_dir
        if "filename" not in config:
            config["filename"] = "simOutput"
            
        self._write_config(config, config_path)
        
        # Run binary
        cmd = [self.binary_path, config_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise RuntimeError(f"CATERPillar simulation failed: {e}")
            
        # Find output CSV
        # CATERPillar appends suffix like _rep00 or similar
        # Pattern: filename + suffix + "_spheres.csv"
        # We search for any csv ending in _spheres.csv in the dir
        csv_files = list(Path(output_dir).glob("*_spheres.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No sphere CSV output found in {output_dir}")
        
        # Load the latest one (or first)
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        
        # Clean up if temp
        if temp_dir:
            # We might want to keep it if debugging, but generally clean up
            # For now, let's keep it if something goes wrong? 
            # I'll just delete the config file but maybe we want to keep the CSV data in memory
            shutil.rmtree(output_dir)
            
        return df

    def get_sdf(self, df: pd.DataFrame) -> Callable[[Any], Any]:
        """
        Converts the sphere DataFrame into a JAX SDF function.
        """
        centers = df[['x', 'y', 'z']].values
        radii = df['radius'].values
        
        sdf_obj = MultiSphereSDF(centers, radii)
        return sdf_obj.get_sdf_func()

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Returns a default configuration for a quick test."""
        return {
            "repetitions": 1,
            "vox_sizes": [10],
            "data_directory": "/tmp",
            "filename": "test",
            "axons_without_myelin_icvf": 0.3,
            "axons_with_myelin_icvf": 0.0,
            "glial_pop1_icvf_soma": 0.05,
            "glial_pop1_icvf_branches": 0.1,
            "glial_pop2_icvf_soma": 0.0,
            "glial_pop2_icvf_branches": 0.0,
            "blood_vessels_icvf": 0.0,
            "spheres_overlap_factor": 4,
            "beading_variation": 0.0,
            "beading_variation_std": 0.0,
            "tortuous": 1,
            "alpha": 4,
            "beta": 0.25,
            "regrow_thr": 20,
            "min_rad": 0.2,
            "std_dev": 0.1,
            "ondulation_factor": 5,
            "beading_period": 10,
            "can_shrink": 1,
            "c2": 0.5,
            "nbr_threads": 1,
            "nbr_axons_populations": 1,
            "crossing_fibers_type": 0,
            "mean_glial_process_length": 15,
            "std_glial_process_length": 5
        }
