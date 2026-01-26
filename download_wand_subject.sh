#!/bin/bash
set -e

BASE_URL="https://gin.g-node.org/CUBRIC/WAND/raw/master"
SUBJECT="sub-00395"
SESSION="ses-02"
MODALITY="dwi"
OUTPUT_DIR="data/wand/${SUBJECT}/${SESSION}/${MODALITY}"

mkdir -p "${OUTPUT_DIR}"

# List of file prefixes
PREFIXES=(
    "${SUBJECT}_${SESSION}_acq-AxCaliber1_dir-AP_part-mag_dwi"
    "${SUBJECT}_${SESSION}_acq-AxCaliber2_dir-AP_part-mag_dwi"
    "${SUBJECT}_${SESSION}_acq-AxCaliber3_dir-AP_part-mag_dwi"
    "${SUBJECT}_${SESSION}_acq-AxCaliber4_dir-AP_part-mag_dwi"
    "${SUBJECT}_${SESSION}_acq-AxCaliberRef_dir-PA_part-mag_dwi"
    "${SUBJECT}_${SESSION}_acq-CHARMED_dir-AP_part-mag_dwi"
    "${SUBJECT}_${SESSION}_acq-CHARMED_dir-PA_part-mag_dwi"
)

EXTENSIONS=("nii.gz" "json" "bval" "bvec")

for PREFIX in "${PREFIXES[@]}"; do
    for EXT in "${EXTENSIONS[@]}"; do
        FILE="${PREFIX}.${EXT}"
        URL="${BASE_URL}/${SUBJECT}/${SESSION}/${MODALITY}/${FILE}"
        DEST="${OUTPUT_DIR}/${FILE}"
        
        echo "Downloading ${FILE}..."
        # Use -L to follow redirects (GIN uses them), -f to fail on 404
        # Remove existing symlink/file before downloading to avoid issues
        rm -f "${DEST}"
        curl -L -f -o "${DEST}" "${URL}"
    done
done

echo "Download complete."
