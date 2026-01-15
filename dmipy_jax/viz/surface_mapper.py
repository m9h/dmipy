"""
Volume-to-Surface Mapper for Macaque Visualization.

This module provides tools to map NIfTI volumes to FreeSurfer surfaces using `mri_vol2surf`
and visualize them using PySurfer. It assumes the subject is 'BigMac' by default, 
tailored for Macaque MRI analysis.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_to_surface(volume_path, hemi, subject='BigMac', output_dir=None, subjects_dir=None):
    """
    Maps a volumetric NIfTI file to a FreeSurfer surface using mri_vol2surf.
    
    Args:
        volume_path (str): Path to the input NIfTI volume.
        hemi (str): Hemisphere ('lh' or 'rh').
        subject (str): Subject ID (default: 'BigMac').
        output_dir (str, optional): Directory to save the output .mgh file. 
                                    If None, uses the directory of volume_path.
        subjects_dir (str, optional): Path to FreeSurfer SUBJECTS_DIR. 
                                      If None, relies on environment variable.

    Returns:
        str: Path to the generated .mgh file.
    """
    volume_path = Path(volume_path).resolve()
    if not volume_path.exists():
        raise FileNotFoundError(f"Input volume not found: {volume_path}")

    if output_dir is None:
        output_dir = volume_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_mgh = output_dir / f"{volume_path.stem}.{hemi}.mgh"

    # Construct mri_vol2surf command
    cmd = [
        "mri_vol2surf",
        "--mov", str(volume_path),
        "--hemi", hemi,
        "--surf", "white",
        "--projfrac", "0.5",  # Sample halfway through cortical ribbon (Gray Matter)
        "--o", str(output_mgh)
    ]

    # Add subject args
    cmd.extend(["--regheader", subject])
    
    # Environment setup
    env = os.environ.copy()
    if subjects_dir:
        env["SUBJECTS_DIR"] = str(subjects_dir)

    logger.info(f"Running mri_vol2surf: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"mri_vol2surf failed:\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise RuntimeError(f"mri_vol2surf failed. See logs for details.")

    if not output_mgh.exists():
        raise RuntimeError(f"Output MGH file was not created: {output_mgh}")

    return str(output_mgh)


def generate_pysurfer_script(mgh_file, hemi, subject='BigMac', colormap='viridis', output_script=None):
    """
    Generates a Python script to visualize the mapped data using PySurfer.

    Args:
        mgh_file (str): Path to the .mgh overlay file.
        hemi (str): Hemisphere ('lh' or 'rh').
        subject (str): Subject ID (default: 'BigMac').
        colormap (str): Colormap name (e.g., 'viridis', 'magma').
        output_script (str, optional): Path to save the generated script.
                                       If None, creates a temp file.

    Returns:
        str: Path to the generated Python script.
    """
    script_content = f"""
import sys
from surfer import Brain
import matplotlib.pyplot as plt

def show_brain():
    subject = '{subject}'
    hemi = '{hemi}'
    surf = 'inflated'
    
    print(f"Loading {{subject}} - {{hemi}}...")
    try:
        brain = Brain(subject, hemi, surf, background="white")
    except Exception as e:
        print(f"Error initializing Brain: {{e}}")
        sys.exit(1)

    print(f"Adding overlay: {mgh_file}")
    brain.add_overlay('{mgh_file}', min=0, max=1, sign="pos", name="overlay", cmap="{colormap}")
    
    brain.show_view("lateral")
    print("Visualization ready. Close the window to exit.")
    brain.show_view("lateral")
    input("Press Enter to close...")

if __name__ == "__main__":
    show_brain()
"""
    if output_script is None:
        fd, output_script = tempfile.mkstemp(suffix=".py", prefix="pysurfer_viz_")
        os.close(fd)
    
    with open(output_script, "w") as f:
        f.write(script_content)
    
    return output_script


def visualize_on_surface(nifti_file, hemi='lh', subject='BigMac', metric_type='neurite', subjects_dir=None):
    """
    Orchestrates the mapping and visualization process.

    Args:
        nifti_file (str): Path to the NIfTI volume.
        hemi (str): Hemisphere ('lh' or 'rh').
        subject (str): Subject ID (default: 'BigMac').
        metric_type (str): Type of metric for colormap selection ('neurite' or 'soma').
        subjects_dir (str, optional): Path to FreeSurfer SUBJECTS_DIR.
    """
    # Select colormap based on metric type
    if metric_type == 'neurite':
        colormap = 'viridis'
    elif metric_type == 'soma':
        colormap = 'magma'
    else:
        colormap = 'viridis'
        logger.warning(f"Unknown metric_type '{metric_type}'. Defaulting to viridis.")

    logger.info(f"Step 1: Mapping {nifti_file} to {hemi} surface...")
    try:
        mgh_file = map_to_surface(
            volume_path=nifti_file,
            hemi=hemi,
            subject=subject,
            subjects_dir=subjects_dir
        )
    except Exception as e:
        logger.error(f"Mapping failed: {e}")
        return

    logger.info(f"Step 2: Generating PySurfer script...")
    script_path = generate_pysurfer_script(
        mgh_file=mgh_file,
        hemi=hemi,
        subject=subject,
        colormap=colormap
    )

    logger.info(f"Step 3: Launching PySurfer...")
    # Run the script in a separate process
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Visualization script failed.")
    finally:
        # Cleanup temporary script
        if os.path.exists(script_path):
            os.remove(script_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map NIfTI volume to Macaque surface.")
    parser.add_argument("file", help="Input NIfTI file")
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh"], help="Hemisphere")
    parser.add_argument("--subject", default="BigMac", help="Subject ID")
    parser.add_argument("--metric", default="neurite", choices=["neurite", "soma"], help="Metric type")
    
    args = parser.parse_args()
    
    visualize_on_surface(args.file, args.hemi, args.subject, args.metric)
