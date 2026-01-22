
import os
from tcia_utils import nbia

# 1. Search for ProstateX Series
# Specifically looking for "ep2d_diff" or similar diffusion sequences
collection = "ProstateX"
print(f"Searching for Series in {collection}...")

# Get patients
patients = nbia.getPatient(collection=collection)
if not patients:
    print("No patients found. Check connection.")
    exit(1)

print(f"Debug: keys in patient 0: {patients[0].keys()}")
# Pick first patient
try:
    pid = patients[0]['PatientId'] # Try camelCase?
except:
    pid = patients[0]['PatientID'] # Fallback
print(f"Selected Patient: {pid}")

# Get Studies
studies = nbia.getStudy(patientId=pid, collection=collection)
if not studies:
    print("No Studies found.")
    exit(1)
study_uid = studies[0]['StudyInstanceUID']

# Get Series
series = nbia.getSeries(studyUid=study_uid, collection=collection)

# Filter for Diffusion
# Exclude calculated maps (BVAL, ADC, Ktrans, etc)
dwi_series = []
print("Available Series:")
for s in series:
    desc = s.get('SeriesDescription', '')
    print(f" - {desc} (UID: {s['SeriesInstanceUID']})")
    
    if "diff" in desc.lower() and "bval" not in desc.lower() and "adc" not in desc.lower() and "calc" not in desc.lower():
        dwi_series.append(s)

if not dwi_series:
    print("Warning: No raw diffusion series found. Trying looser filter.")
    dwi_series = [s for s in series if "diff" in s.get('SeriesDescription', '').lower()]

selected_series_uid = dwi_series[0]['SeriesInstanceUID']
desc = dwi_series[0]['SeriesDescription']
print(f"Selected for Download: {desc} (UID: {selected_series_uid})")

# Download
save_path = "data/ProstateX_Sample"
nbia.downloadSeries(series_data=[selected_series_uid], input_type="list", path=save_path)

print(f"Download complete to {save_path}")
