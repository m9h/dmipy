#!/bin/bash
set -e

BASE_URL="https://gin.g-node.org/CUBRIC/WAND/raw/master"
DERIV_DIR="data/wand/derivatives/eddy_qc/preprocessed/sub-00395"
OUTPUT_DIR="data/wand/sub-00395/ses-02/dwi_preproc"

mkdir -p "${OUTPUT_DIR}"

# Files to download
# 1. Eddy Corrected Data (Outlier Free) - This corresponds to ALL shells concatenated? 
#    Need to verify if this is AxCaliber ONLY or EVERYTHING. 
#    Usually 'sub-00395_eddy_corrected_data' implies a merged dataset.
#    Wait, raw data had separate NIfTIs for AxCaliber1-4 and CHARMED.
#    Edy usually merges them. I need to know the order to split them or use the merged bvals/bvecs.

FILES=(
    "sub-00395_eddy_corrected_data.eddy_outlier_free_data.nii.gz"
    "sub-00395_eddy_corrected_data.eddy_rotated_bvecs"
    "sub-00395_b0_brain_mask.nii.gz"
)

# I also need the bvals. Eddy output doesn't usually change bvals unless they are in the .eddy_command_txt or we have to construct them from the raw files.
# Usually we concatenate the raw bvals in the same order as `eddy` was run.
# I will download the `eddy_command_txt` to see the input order.

FILES+=("sub-00395_eddy_corrected_data.eddy_command_txt")

for FILE in "${FILES[@]}"; do
    URL="${BASE_URL}/derivatives/eddy_qc/preprocessed/sub-00395/${FILE}"
    DEST="${OUTPUT_DIR}/${FILE}"
    
    echo "Downloading ${FILE}..."
    rm -f "${DEST}"
    curl -L -f -o "${DEST}" "${URL}"
done

echo "Download complete."
