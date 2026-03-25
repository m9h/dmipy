
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def list_bucket_contents():
    print("Listing contents of ds006557 on OpenNeuro S3...")
    s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    bucket = "openneuro.org"
    
    # Try generic prefix
    prefix = "ds006557/"
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    count = 0
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            # Only show first 50 files to avoid massive output, but prioritize dwi
            if "dwi" in key or count < 20:
                print(key)
            if "dwi" in key:
                 # If we find dwi, verify if it has data
                 pass
            count += 1
            if count > 200: # Stop after some basic check if no dwi found earlier
                break
                
    if count == 0:
        print("No objects found. Check bucket/prefix.")

if __name__ == "__main__":
    list_bucket_contents()
