import os
import datalad.api as dl

def download_data():
    # Path to the installed dataset
    # This assumes the script is run from project root or similar. 
    # Current location: dmipy/dmipy_jax/examples/clinical/download_conp_adolescent.py
    # Dataset location: dmipy/dmipy/data/conp-dataset/projects/AdolescentBrainDevelopment
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    dataset_path = os.path.join(project_root, "dmipy/data/conp-dataset/projects/AdolescentBrainDevelopment")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure you have run the installation steps.")
        return

    subject = "sub-10002"
    session = "ses-01"
    
    # Paths to download
    anat_path = os.path.join(dataset_path, subject, session, "anat")
    dwi_path = os.path.join(dataset_path, subject, session, "dwi")
    
    print(f"Downloading data for {subject}...")
    
    # Download Anat
    print(f"Getting {anat_path}...")
    dl.get(path=anat_path, dataset=dataset_path, recursive=True)
    
    # Download DWI
    print(f"Getting {dwi_path}...")
    dl.get(path=dwi_path, dataset=dataset_path, recursive=True)
    
    print("Download complete.")
    
    # Verify files
    for root, dirs, files in os.walk(dwi_path):
        for f in files:
            print(f"  - {f} ({os.path.getsize(os.path.join(root, f))} bytes)")

if __name__ == "__main__":
    download_data()
