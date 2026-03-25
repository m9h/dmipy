
import datalad.api as dl
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ds004024_downloader")

DATASET_URL = "https://github.com/OpenNeuroDatasets/ds004024.git"
DEFAULT_path = "data/ds004024"

def main():
    parser = argparse.ArgumentParser(description="Download ds004024 (Subject 01) via DataLad")
    parser.add_argument("--path", default=DEFAULT_path, help="Local path to install dataset")
    parser.add_argument("--subject", default="CON001", help="Subject ID to fetch (default: CON001)")
    args = parser.parse_args()
    
    path = os.path.abspath(args.path)
    
    # 1. Install/Clone Dataset
    if not os.path.exists(path):
        logger.info(f"Cloning ds004024 from {DATASET_URL} to {path}...")
        dl.install(path=path, source=DATASET_URL)
    else:
        logger.info(f"Dataset directory exists at {path}. Checking status...")
        # Verify it's a datalad dataset
        ds = dl.Dataset(path)
        if not ds.is_installed():
            logger.info("Directory exists but is not installed. Installing...")
            ds.install(source=DATASET_URL)
            
    ds = dl.Dataset(path)
    
    # 2. Get Data for Subject
    # Pattern: sub-CON001/
    logger.info(f"Fetching data for sub-{args.subject}...")
    
    # We need:
    # - DWI: sub-CON001/dwi/*
    # - EEG (for events): sub-CON001/eeg/* (or wherever events.tsv is)
    # - ANAT: sub-CON001/anat/*
    
    sub_path = f"sub-{args.subject}"
    
    # Structure found: ses-mri, ses-async14ms, ses-async4ms
    # MRI data in ses-mri
    # EEG likely in ses-async sessions
    
    paths_to_get = [
        os.path.join(path, sub_path, "ses-mri", "dwi"),
        os.path.join(path, sub_path, "ses-mri", "anat"),
        os.path.join(path, sub_path, "ses-async14ms", "eeg") # Example session
    ]
    
    # Also get dataset_description.json and participants.tsv for metadata
    paths_to_get.append(os.path.join(path, "dataset_description.json"))
    paths_to_get.append(os.path.join(path, "participants.tsv"))
    
    logger.info(f"Retrieving content for: {paths_to_get}")
    
    try:
        results = ds.get(path=paths_to_get)
        # DataLad returns a list of results. We check for failures.
        failures = [r for r in results if r['status'] in ('error', 'impossible')]
        if failures:
            logger.error(f"Some downloads failed: {failures}")
        else:
            logger.info("Download completed successfully.")
            
    except Exception as e:
        logger.error(f"DataLad get failed: {e}")
        # Sometimes 'eeg' folder might be named differently ('func' or custom).
        # We try to get the subject folder top-level if specific subfolders fail?
        # No, 'get sub-01' might comprise many gigabytes of raw eeg.
        # Let's inspect structure if failed.
        pass

if __name__ == "__main__":
    main()
