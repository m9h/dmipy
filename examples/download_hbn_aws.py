
import os
import argparse
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time


HBN_BUCKET = "fcp-indi"
HBN_MRI_PREFIX = "data/Projects/HBN/MRI/"

def get_s3_client():
    """Returns an S3 client configured for anonymous access."""
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))

def list_sites(client):
    """Lists available sites (e.g. Site-CBIC)."""
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=HBN_BUCKET, Prefix=HBN_MRI_PREFIX, Delimiter='/')
    sites = []
    for page in pages:
        if 'CommonPrefixes' in page:
            for p in page['CommonPrefixes']:
                # data/Projects/HBN/MRI/Site-X/
                if 'Site-' in p['Prefix']:
                    sites.append(p['Prefix'])
    return sites

def list_subjects_for_site(client, site_prefix):
    """Lists subjects within a site folder."""
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=HBN_BUCKET, Prefix=site_prefix, Delimiter='/')
    subjects = []
    for page in pages:
        if 'CommonPrefixes' in page:
            for p in page['CommonPrefixes']:
                 # .../sub-NDAR.../
                sub_id = p['Prefix'].rstrip('/').split('/')[-1]
                if sub_id.startswith('sub-'):
                    subjects.append({'id': sub_id, 'prefix': p['Prefix']})
    return subjects

def list_all_subjects(client):
    """Aggregates subjects from all sites."""
    sites = list_sites(client)
    all_subs = []
    print(f"Found sites: {[s.split('/')[-2] for s in sites]}")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(list_subjects_for_site, client, site): site for site in sites}
        for f in futures:
            all_subs.extend(f.result())
            
    # Sort by ID
    all_subs.sort(key=lambda x: x['id'])
    return all_subs

def download_file(client, bucket, key, local_path):
    """Downloads a single file from S3."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
             print(f"Downloading {key} -> {local_path}")
             client.download_file(bucket, key, local_path)
    except Exception as e:
        print(f"Error downloading {key}: {e}")

def download_subject_dwi(client, subject_info, output_dir):
    """
    Downloads DWI data for a subject from their specific prefix.
    """
    prefix = subject_info['prefix']
    sub_id = subject_info['id']
    
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=HBN_BUCKET, Prefix=prefix) # Recursive list under subject
    
    dwi_files = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Filter for DWI NIfTI, bval, bvec, json
                if '/dwi/' in key and (key.endswith('.nii.gz') or key.endswith('.bval') or key.endswith('.bvec') or key.endswith('.json')):
                    dwi_files.append(key)
                    
    if not dwi_files:
        print(f"No DWI data found for {sub_id}")
        return

    # print(f"Found {len(dwi_files)} DWI files for {sub_id}. Downloading...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for key in dwi_files:
            # key: data/Projects/HBN/MRI/Site-X/sub-Y/...
            # We want local: output_dir/sub-Y/...
            
            # Find part after sub-ID to keep structure?
            # actually we probably just want output_dir/sub-ID/dwi/file
            # but BIDS structure is usually sub-ID/dwi/
            
            # Extract relative path from subject directory
            # prefix is .../sub-Y/
            rel_path = key.replace(prefix, "") 
            local_path = os.path.join(output_dir, sub_id, rel_path)
            
            executor.submit(download_file, client, HBN_BUCKET, key, local_path)

def main():
    parser = argparse.ArgumentParser(description="Download HBN DWI data from AWS S3")
    parser.add_argument("--output_dir", default="data/hbn_aws", help="Directory to save data")
    parser.add_argument("--n_subjects", type=int, default=5, help="Number of subjects to download")
    parser.add_argument("--subjects", nargs="+", help="Specific subject IDs to download (e.g. sub-NDAR...)")
    args = parser.parse_args()
    
    client = get_s3_client()
    
    print("Fetching subject list from S3 sites...")
    all_subjects = list_all_subjects(client)
    print(f"Total subjects found: {len(all_subjects)}")
    
    if args.subjects:
        # Filter all_subjects for the requested IDs
        requested_set = set(args.subjects)
        subjects_to_process = [s for s in all_subjects if s['id'] in requested_set]
        print(f"Found {len(subjects_to_process)} of {len(requested_set)} requested subjects.")
        missing = requested_set - set(s['id'] for s in subjects_to_process)
        if missing:
            print(f"Warning: Could not find {len(missing)} requested subjects: {missing}")
    else:
        subjects_to_process = all_subjects[:args.n_subjects]
    
    
    print(f"Downloading data for {len(subjects_to_process)} subjects...")
    
    for sub in subjects_to_process:
        print(f"Processing {sub['id']}...")
        download_subject_dwi(client, sub, args.output_dir)
        
    print("Done.")

if __name__ == "__main__":
    main()
