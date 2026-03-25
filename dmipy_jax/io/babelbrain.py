"""
BabelBrain Dataset Loader (Zenodo DOI: 10.5281/zenodo.7894431).
Contains 5 subjects with MRI (T1, UTE/ZTE/PETRA) and CT data.
Used for validating Pseudo-CT generation.
"""

import os
import requests
import zipfile
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional

# Zenodo Record Info
ZENODO_Record = "7894431"
FILE_URL = "https://zenodo.org/record/7894431/files/BabelBrain_Data.zip?download=1"
FILE_NAME = "BabelBrain_Data.zip"

DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dmipy_jax", "babelbrain")

def download_babelbrain(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Downloads and extracts the BabelBrain dataset.
    Returns the path to the extracted directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, FILE_NAME)
    extract_dir = os.path.join(data_dir, "extracted")
    
    # Check if already extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"BabelBrain data found in {extract_dir}")
        return extract_dir
        
    # Download
    if not os.path.exists(zip_path):
        print(f"Downloading BabelBrain dataset from Zenodo ({ZENODO_Record})...")
        print("This is ~1-2GB, please wait...")
        
        headers = {'User-Agent': 'dmipy-jax/0.1 (research purpose)'}
        try:
            response = requests.get(FILE_URL, stream=True, headers=headers, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                print(f"Zenodo error {response.status_code}. Retrying in 5s...")
                import time; time.sleep(5)
                response = requests.get(FILE_URL, stream=True, headers=headers, timeout=60)
            
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            if os.path.exists(zip_path): os.remove(zip_path) # Cleanup partial
            raise e
        
    # Extract
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    
    return extract_dir


GITHUB_REPO_URL = "https://github.com/ProteusMRIgHIFU/BabelBrain/archive/refs/heads/main.zip"

def download_babelbrain_repo(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Fallback: Downloads the GitHub source repo."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "BabelBrain_Repo.zip")
    extract_dir = os.path.join(data_dir, "extracted_repo")
    
    if os.path.exists(extract_dir): return extract_dir
    
    print("Downloading BabelBrain GitHub Repo (Fallback)...")
    try:
        r = requests.get(GITHUB_REPO_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        return extract_dir
    except Exception as e:
        print(f"Repo download failed: {e}")
        return None

def load_subject(subject_id: str = "01", data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, str]:
    """
    Loads file paths for a specific subject.
    """
    try:
        root_dir = download_babelbrain(data_dir)
    except Exception as e:
        print(f"Zenodo download failed ({e}). Trying GitHub Repo Fallback...")
        root_dir = download_babelbrain_repo(data_dir)
        if not root_dir: raise e
        
    # Search logic (same as before)
    # BabelBrain repo structure usually: BabelBrain-main/OfflineBatchExamples/...
    # Zenodo structure: BabelBrain_Data/sub-01/...
    
    subject_files = {}
    sid = subject_id.replace("sub-", "")
    
    for root, dirs, files in os.walk(root_dir):
        # Zenodo style match
        if f"sub-{sid}" in os.path.basename(root):
            pass # Logic below will catch
        
        # GitHub Repo Example match
        # Often named "ExampleData" or located in "OfflineBatchExamples"
        # We'll just greedy match any NIfTI that looks right
        
        for f in files:
            f_lower = f.lower()
            full_path = os.path.join(root, f)
            
            # Simple heuristics
            if "t1" in f_lower and ".nii" in f_lower:
                subject_files['t1'] = full_path
            elif "ct" in f_lower and ".nii" in f_lower and "pseudo" not in f_lower:
                subject_files['ct'] = full_path
            elif ("ute" in f_lower or "zte" in f_lower or "petra" in f_lower) and ".nii" in f_lower:
                key = "petra" if "petra" in f_lower else ("zte" if "zte" in f_lower else "ute")
                subject_files[key] = full_path
                    
    return subject_files
                    
    if not subject_files:
         # Try simpler matching if directory names don't match sub-XX
         pass

    return subject_files

def get_pseudo_ct_data(subject_id="01"):
    """
    Helper to return nibabel images for T1 and CT (Ground Truth).
    """
    files = load_subject(subject_id)
    if 't1' not in files or 'ct' not in files:
        raise ValueError(f"Could not find paired T1/CT for subject {subject_id}. Found: {files.keys()}")
        
    t1_img = nib.load(files['t1'])
    ct_img = nib.load(files['ct'])
    
    # ZTE/PETRA if available
    ute_img = None
    if 'petra' in files: ute_img = nib.load(files['petra'])
    elif 'zte' in files: ute_img = nib.load(files['zte'])
    elif 'ute' in files: ute_img = nib.load(files['ute'])
    
    return t1_img, ct_img, ute_img
