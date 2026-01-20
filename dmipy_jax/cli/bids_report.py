import argparse
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import nibabel as nib

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger("dmipy-report")

def find_bids_root(path: Path) -> Path:
    """Walks up from path to find dataset_description.json or checks current dir."""
    current = path.resolve()
    # Check if we are inside a BIDS dataset
    for _ in range(5): # Check up to 5 levels up
        if (current / "dataset_description.json").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    
    # If not found, just assume the input path is meant to be the root or contains sub-folders
    return path

def analyze_shells(bvals_path: Path) -> dict:
    """
    Analyzes a bval file to identify shells.
    Returns: dict {shell_bval: n_directions}
    """
    try:
        bvals = np.loadtxt(bvals_path)
    except Exception:
        return {}
        
    # Handle single volume case
    if bvals.ndim == 0 or bvals.size == 1:
        val = int(bvals) if bvals.size == 1 else 0
        return {val: 1}
        
    # Cluster b-values (round to nearest 50 to tolerant small variations)
    rounded = np.round(bvals / 50) * 50
    unique, counts = np.unique(rounded, return_counts=True)
    
    shells = {}
    for u, c in zip(unique, counts):
        shells[int(u)] = int(c)
        
    return shells

def generate_report(bids_dir: Path) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"BIDS Dataset Report")
    lines.append(f"Location: {bids_dir}")
    
    # Load description
    desc_path = bids_dir / "dataset_description.json"
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                desc = json.load(f)
            lines.append(f"Name: {desc.get('Name', 'Unknown')}")
            lines.append(f"BIDS Version: {desc.get('BIDSVersion', 'Unknown')}")
        except:
            lines.append("Name: (Error reading dataset_description.json)")
    else:
        lines.append("Warning: dataset_description.json not found (might not be valid BIDS)")
        
    lines.append("-" * 60)
    
    # Count subjects
    subjects = list(bids_dir.glob("sub-*"))
    lines.append(f"Total Subjects: {len(subjects)}")
    
    # Analyze Modalities (heuristic)
    modalities = set()
    dwi_files = list(bids_dir.glob("sub-*/dwi/*_dwi.nii.gz")) + \
                list(bids_dir.glob("sub-*/ses-*/dwi/*_dwi.nii.gz"))
    anat_files = list(bids_dir.glob("sub-*/anat/*.nii.gz")) + \
                 list(bids_dir.glob("sub-*/ses-*/anat/*.nii.gz"))
                 
    if dwi_files: modalities.add("dwi")
    if anat_files: modalities.add("anat")
    
    lines.append(f"Modalities found: {', '.join(sorted(modalities))}")
    
    if "dwi" in modalities:
        lines.append("-" * 60)
        lines.append("DWI Summary")
        lines.append(f"Total DWI scans found: {len(dwi_files)}")
        
        # Analyze first DWI found as exemplar (common assumption in homogeneous cohorts)
        # Ideally we'd check checking for heterogeneity, but let's start simple.
        exemplar_dwi = dwi_files[0]
        lines.append(f"\nExemplar Scan: {exemplar_dwi.name}")
        
        # Load header for resolution
        try:
            img = nib.load(exemplar_dwi)
            hdr = img.header
            zooms = hdr.get_zooms()
            shape = img.shape
            lines.append(f"  Dimensions: {shape}")
            lines.append(f"  Voxel Size: {zooms[:3]} mm")
        except Exception as e:
            lines.append(f"  (Could not read NIfTI header: {e})")
            
        # Find corresponding bval
        # BIDS: replace _dwi.nii.gz with .bval
        if exemplar_dwi.name.endswith(".nii.gz"):
            bval_name = exemplar_dwi.name.replace(".nii.gz", ".bval")
        else:
             bval_name = exemplar_dwi.name.replace(".nii", ".bval")
             
        bval_path = exemplar_dwi.parent / bval_name
        
        if bval_path.exists():
            shells = analyze_shells(bval_path)
            lines.append("\n  Acquisition Shells (b-value : directions):")
            for b, count in sorted(shells.items()):
                lines.append(f"    b={b:<5} : {count}")
        else:
            lines.append("\n  (No .bval file found for exemplar)")

    lines.append("=" * 60)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate a summary report for a BIDS dataset.")
    parser.add_argument("bids_dir", type=str, help="Path to the BIDS dataset root directory.")
    parser.add_argument("--output", "-o", type=str, help="Optional output text file to save the report.")
    
    args = parser.parse_args()
    
    p = Path(args.bids_dir)
    if not p.exists():
        print(f"Error: Path {p} does not exist.")
        return
        
    root = find_bids_root(p)
    logger = setup_logging()
    logger.info(f"Analyzing {root}...")
    
    report = generate_report(root)
    
    print(report)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")

if __name__ == "__main__":
    main()
