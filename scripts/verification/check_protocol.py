import json
import glob
import os

base_dir = "data/wand/sub-00395/ses-02/dwi"
files = sorted(glob.glob(os.path.join(base_dir, "*.json")))

print(f"{'Filename':<50} | {'Delta (ms)':<10} | {'TE (ms)':<10}")
print("-" * 80)

for f in files:
    with open(f, 'r') as fp:
        meta = json.load(fp)
        
    delta_big = meta.get('t_bdel', 'N/A')
    te = meta.get('EchoTime', 'N/A')
    if te != 'N/A':
        te = float(te) * 1000 # to ms
        
    fname = os.path.basename(f)
    print(f"{fname:<50} | {str(delta_big):<10} | {te:<10}")
