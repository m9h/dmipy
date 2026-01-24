import os
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import shutil
from pathlib import Path

class AmicoOracle:
    """
    AMICO Oracle wrapper using Docker.
    
    This class manages the interaction with the AMICO software running inside a Docker container.
    It handles:
    - Data marshalling (writing in-memory arrays to temporary NIfTI files)
    - Container execution (mounting volumes, running the internal script)
    - Result retrieval (reading output NIfTI files back into memory)
    """
    
    def __init__(self, docker_image="amico_oracle:latest", build_if_missing=True):
        self.docker_image = docker_image
        self.amico_dir = Path(__file__).parent / "AMICO"
        
        if build_if_missing:
            self._check_and_build_image()
            
    def _check_and_build_image(self):
        """Check if Docker image exists, else build it."""
        try:
            # Check if image exists
            subprocess.run(
                ["docker", "inspect", self.docker_image], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print(f"Docker image {self.docker_image} not found. Building...")
            self.build_docker_image()
            
    def build_docker_image(self):
        """Build the AMICO Docker image."""
        if not (self.amico_dir / "Dockerfile").exists():
            raise FileNotFoundError(f"Dockerfile not found in {self.amico_dir}")
            
        subprocess.run(
            ["docker", "build", "-t", self.docker_image, "."],
            cwd=self.amico_dir,
            check=True
        )
        print(f"Docker image {self.docker_image} built successfully.")

    def fit(self, data, bvals, bvecs, mask=None, model="NODDI"):
        """
        Run AMICO fit on the provided data.
        
        Args:
            data (np.ndarray): 4D DWI data (or 3D if single volume).
            bvals (np.ndarray): 1D array of b-values.
            bvecs (np.ndarray): 2D array of b-vectors (N, 3).
            mask (np.ndarray, optional): 3D binary mask. If None, computes a simple non-zero mask.
            model (str): Model name (NODDI, ActiveAx, SANDI).
            
        Returns:
            dict: Dictionary containing fitted parameter maps.
        """
        # Ensure data is 4D
        if data.ndim == 3:
            data = data[..., np.newaxis]
            
        model = model.upper()
        
        # Create temporary directory for data exchange
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Prepare Inputs
            dwi_file = temp_path / "DWI.nii.gz"
            bvals_file = temp_path / "DWI.bval"
            bvecs_file = temp_path / "DWI.bvec"
            mask_file = temp_path / "nodif_brain_mask.nii.gz" # Standard name often helpful
            
            # Save NIfTI
            affine = np.eye(4) # Dummy affine, we don't care about geometry for oracle, just pixel values usually
            nib.save(nib.Nifti1Image(data, affine), dwi_file)
            
            # Save bvals/bvecs (FSL format)
            np.savetxt(bvals_file, bvals[None, :], fmt='%d')
            np.savetxt(bvecs_file, bvecs.T, fmt='%.6f')
            
            # Save Mask
            if mask is None:
                # Create simple mask of all voxels with signal > 0 in first volume or simple box
                mask = np.any(data > 0, axis=-1).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_file)
            
            # 2. Run Docker Container
            # We mount the temp dir to /data inside the container
            cmd = [
                "docker", "run", "--rm",
                "--gpus", "all", # Enable GPUs if available/needed (AMICO is mostly CPU/OpenBLAS but good to have)
                "-v", f"{temp_path}:/data",
                self.docker_image,
                "python", "/opt/AMICO/run_amico_internal.py",
                "--data", "/data/DWI.nii.gz",
                "--bvals", "/data/DWI.bval",
                "--bvecs", "/data/DWI.bvec",
                "--mask", "/data/nodif_brain_mask.nii.gz",
                "--model", model,
                "--output", "AMICO/RESULTS"
            ]
            
            print(f"Running AMICO {model} fit in Docker...")
            try:
                subprocess.run(cmd, check=True) # user=root by default in docker usually, avoiding permission issues in temp might need care
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"AMICO fit failed: {e}")
                
            # 3. Retrieve Results
            results_dir = temp_path / "AMICO" / "RESULTS" / model
            
            if not results_dir.exists():
                # Try fallback or check where AMICO saves
                # It typically saves in <Subject>/AMICO/RESULTS/<Model>
                # Our subject is the root /data
                pass 
                
            results = {}
            # List all nifti files in the output directory
            # typical outputs: OD.nii.gz, ICVF.nii.gz, ISOVF.nii.gz for NODDI
            for nifti_file in results_dir.glob("*.nii.gz"):
                key = nifti_file.name.replace(".nii.gz", "")
                img = nib.load(nifti_file)
                results[key] = img.get_fdata()
                
            return results

if __name__ == "__main__":
    # Simple self-test
    print("Initializing AMICO Oracle...")
    oracle = AmicoOracle()
    print("Oracle initialized.")
