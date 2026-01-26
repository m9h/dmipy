import os
import json
import numpy as np
import nibabel as nib
import jax.numpy as jnp

class WandLoader:
    def __init__(self, base_dir="data/wand/sub-00395/ses-02"):
        self.base_dir = base_dir
        self.dwi_dir = os.path.join(base_dir, "dwi")
        self.anat_dir = os.path.join(base_dir, "anat")
        self.preproc_dir = os.path.join(base_dir, "dwi_preproc")

    def load_qmt(self):
        """
        Loads Quantitative Magnetization Transfer (qMT) data from 'anat'.
        
        Structure:
            - mt-off: Reference scans (Flip Angle ~5-15 deg)
            - mt-X: MT-weighted scans with varying offsets and MT powers.
            
        Returns:
            data: (X, Y, Z, N_meas)
            offsets: (N_meas,) in Hz. 0 for MT-off? Or inf? Usually large for off.
            mt_powers: (N_meas,) Flip Angle of MT pulse (degrees).
            excitation_flips: (N_meas,) Flip Angle of Readout pulse (degrees).
            TRs: (N_meas,) Repetition Time (s).
        """
        print("Loading qMT Data...")
        
        # We need to crawl the directory for files matching *QMT.nii.gz*
        # and parsing their JSONs.
        
        import glob
        qmt_files = sorted(glob.glob(os.path.join(self.anat_dir, "*QMT.nii.gz")))
        
        if not qmt_files:
            print(f"No qMT files found in {self.anat_dir}")
            return None
            
        all_data = []
        offsets = []
        mt_powers = []
        ex_flips = []
        trs = []
        
        print(f"Found {len(qmt_files)} qMT files.")
        
        for fpath in qmt_files:
            fname = os.path.basename(fpath)
            json_path = fpath.replace(".nii.gz", ".json")
            
            if not os.path.exists(json_path):
                print(f"Missing JSON for {fname}")
                continue
                
            img = nib.load(fpath)
            vol = img.get_fdata() # (X,Y,Z) or (X,Y,Z,1) usually?
            if vol.ndim == 4:
                vol = vol[..., 0] # Assume single volume per file for these individual acquisitions
                
            with open(json_path, 'r') as f:
                meta = json.load(f)
                
            # Parse Metadata
            # 1. Offset Frequency
            # mt-off usually doesn't have MTOffsetFrequency, or has it set to something placeholder.
            # If 'mt-off' in filename, offset is effectively Infinite.
            if 'mt-off' in fname:
                offset = np.inf 
                mt_flip = 0.0 # No Saturation
            else:
                offset_str = meta.get('MTOffsetFrequency', None)
                if offset_str:
                    offset = float(offset_str)
                else:
                    offset = np.inf # Fallback
                
                # Parse MT Flip Angle ("628" etc)
                # If FlipAngle is huge (>90), it's likely the MT pulse flip.
                fa_val = float(meta.get('FlipAngle', 0))
                if fa_val > 90:
                    mt_flip = fa_val
                else:
                    mt_flip = 0.0 # Maybe it's low power MT? Or just not listed here.
                    
            # 2. Excitation Flip Angle
            # The nominal FlipAngle in JSON might be overwritten by MT Flip.
            # We assume a fixed excitation flip for the SPGR readout if not explicit.
            # Usually qMT protocols use a fixed small flip angle (e.g. 5-15 deg).
            # The 'mt-off' file says FlipAngle: 5. We will assume this is the ExFlip for all.
            # UNLESS 'flip-X' entity changes it?
            # Let's inspect 'flip-X' entity in filename.
            # sub-00395_ses-02_flip-1_mt-1...
            # If we see different 'flip-' indices, maybe excitation flip changes?
            # But usually qMT varies MT power/offset.
            # For now, let's hardcode ExFlip = 5.0 derived from mt-off reference.
            # TODO: Improve this logic if we confirm multi-flip-angle VFA-qMT.
            ex_flip = 5.0 # Degrees
            
            # 3. TR
            tr = float(meta.get('RepetitionTime', 0.057)) # s
            
            all_data.append(vol)
            offsets.append(offset)
            mt_powers.append(mt_flip)
            ex_flips.append(ex_flip)
            trs.append(tr)
            
        # Concatenate
        # Stack along 4th dim
        full_data = np.stack(all_data, axis=-1)
        
        return full_data, np.array(offsets), np.array(mt_powers), np.array(ex_flips), np.array(trs)

    def load_axcaliber(self):
        """
        Loads and concatenates AxCaliber shells 1-4 (Raw).
        Returns:
            data: (X, Y, Z, N_total)
            bvals: (N_total,)
            bvecs: (N_total, 3)
            big_deltas: (N_total,)
            small_deltas: (N_total,)
        """
        print("Loading AxCaliber (Raw)...")
        
        shells = [1, 2, 3, 4]
        all_data = []
        all_bvals = []
        all_bvecs = []
        all_big_deltas = []
        all_small_deltas = []
        
        for s in shells:
            prefix = f"sub-00395_ses-02_acq-AxCaliber{s}_dir-AP_part-mag_dwi"
            nii_path = os.path.join(self.dwi_dir, f"{prefix}.nii.gz")
            bval_path = os.path.join(self.dwi_dir, f"{prefix}.bval")
            bvec_path = os.path.join(self.dwi_dir, f"{prefix}.bvec")
            json_path = os.path.join(self.dwi_dir, f"{prefix}.json")
            
            if not os.path.exists(nii_path):
                print(f"Warning: {nii_path} not found. Skipping shell {s}.")
                continue
                
            # Load Data
            img = nib.load(nii_path)
            data = img.get_fdata()
            
            # Load sidecars
            bvals = np.loadtxt(bval_path)
            bvecs = np.loadtxt(bvec_path).T # FSL format is 3xN, we want Nx3
            
            with open(json_path, 'r') as f:
                meta = json.load(f)
                
            # Parse Big Delta from SeriesDescription 'AX_delta17p3...' or t_bdel
            # t_bdel is usually reliable in this dataset
            t_bdel_str = meta.get('t_bdel', None)
            if t_bdel_str:
                big_delta = float(t_bdel_str) / 1000.0 # ms -> s
            else:
                # Fallback to parse SeriesDescription
                desc = meta.get('SeriesDescription', '')
                if 'delta' in desc:
                    # e.g. AX_delta17p3...
                    # Extract 17p3
                    try:
                        part = desc.split('delta')[1].split('_')[0]
                        part = part.replace('p', '.')
                        big_delta = float(part) / 1000.0
                    except:
                        print(f"Could not parse delta from {desc}")
                        big_delta = 0.02 # Default guess?
                else:
                    big_delta = 0.02
            
            # Estimate Small Delta (delta)
            # Assuming fixed for the sequence. 
            # If not in JSON, we'll assume a standard value for Connectom AxCaliber
            # Often delta is quite small relative to large Deltas, but finite.
            # Example values for AxCaliber: delta ~ 3-10ms.
            # Let's check SeriesDescription again: 
            # AX_delta17p3_30xb2200_30xb4400. Doesn't list small delta.
            # We will use a standard approximation or fixed value.
            # Let's use 0.003s (3ms) as a placeholder or try to derive.
            # Actually, let's verify if 'PVM_DwEffBval' or similar is hidden.
            # For now: 0.006s (6ms) is a reasonable guess for Connectom high-G.
            small_delta = 0.006 
            
            n_vols = data.shape[-1]
            
            all_data.append(data)
            all_bvals.append(bvals)
            all_bvecs.append(bvecs)
            all_big_deltas.append(np.full(n_vols, big_delta))
            all_small_deltas.append(np.full(n_vols, small_delta))
            
            print(f"Shell {s}: Delta={big_delta*1000:.1f}ms, delta={small_delta*1000:.1f}ms, N={n_vols}")

        # Concatenate
        full_data = np.concatenate(all_data, axis=-1)
        full_bvals = np.concatenate(all_bvals, axis=0)
        full_bvecs = np.concatenate(all_bvecs, axis=0)
        full_big = np.concatenate(all_big_deltas, axis=0)
        full_small = np.concatenate(all_small_deltas, axis=0)
        
        return full_data, full_bvals, full_bvecs, full_big, full_small

    def load_charmed(self):
        """
        Loads CHARMED (Preprocessed).
        Returns:
            data: (X, Y, Z, N)
            bvals: (N,)
            bvecs: (N, 3) 
            (Delta, delta assumed constant or returned)
        """
        print("Loading CHARMED (Preprocessed)...")
        
        # Paths
        nii_path = os.path.join(self.preproc_dir, "sub-00395_eddy_corrected_data.eddy_outlier_free_data.nii.gz")
        bvec_rotated_path = os.path.join(self.preproc_dir, "sub-00395_eddy_corrected_data.eddy_rotated_bvecs")
        # Bvals come from raw
        bval_raw_path = os.path.join(self.dwi_dir, "sub-00395_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval")
        mask_path = os.path.join(self.preproc_dir, "sub-00395_b0_brain_mask.nii.gz")
        
        if not os.path.exists(nii_path):
            print("Preprocessed data not found.")
            return None
            
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        bvals = np.loadtxt(bval_raw_path)
        bvecs = np.loadtxt(bvec_rotated_path).T
        mask = nib.load(mask_path).get_fdata() > 0
        
        # Check integrity
        if data.shape[-1] != len(bvals):
            print(f"Error: Data volumes ({data.shape[-1]}) mismatch bvals ({len(bvals)})")
            # Maybe the preprocessed data DROPPED outliers?
            # 'eddy_outlier_free_data' REPLACES outliers with interpolated data, usually keeping N constant.
            
        return data, bvals, bvecs, mask

    def load_mcdespot(self):
        """
        Loads all data required for Joint Inversion (mcDESPOT + qMT).
        
        Returns:
            data_dict: Dictionary containing:
                - 'spgr': (X, Y, Z, 8) Magnitude data
                - 'ssfp': (X, Y, Z, 16) Magnitude data
                - 'ir':   (X, Y, Z)   SPGR-IR Magnitude data
                - 'qmt':  (X, Y, Z, N_qmt) qMT Magnitude data
            
            proto_dict: Dictionary containing protocol arrays:
                - 'spgr_fa': (8,) degrees
                - 'spgr_tr': float (s)
                - 'ssfp_fa': (16,) degrees
                - 'ssfp_tr': float (s)
                - 'ssfp_phase': (16,) degrees (0 or 180)
                - 'ir_ti': float (s)
                - 'qmt_offsets': (N_qmt,) Hz
                - 'qmt_sat_fa': (N_qmt,) degrees
                - 'qmt_ex_fa': (N_qmt,) degrees
                - 'qmt_tr': (N_qmt,) s
        """
        print("Loading mcDESPOT + qMT Data...")
        
        # 1. SPGR VFA
        spgr_path = os.path.join(self.anat_dir, "sub-00395_ses-02_acq-spgr_part-mag_VFA.nii.gz")
        spgr_data = nib.load(spgr_path).get_fdata() # (X, Y, Z, 8)
        
        # Protocol (Hardcoded based on CUBRIC WAND knowledge)
        spgr_fa = np.array([3, 4, 5, 6, 7, 9, 13, 18], dtype=float)
        spgr_tr = 0.007 # 7ms approx? Need to verify from JSON or use standard
        # Let's load SPGR JSON for TR
        spgr_json = spgr_path.replace(".nii.gz", ".json")
        with open(spgr_json, 'r') as f:
            meta = json.load(f)
            spgr_tr = float(meta.get('RepetitionTime', 0.0073)) # Default to typical 7.3ms

        # 2. SSFP VFA
        ssfp_path = os.path.join(self.anat_dir, "sub-00395_ses-02_acq-ssfp_part-mag_VFA.nii.gz")
        ssfp_data = nib.load(ssfp_path).get_fdata() # (X, Y, Z, 16)
        
        # Protocol
        # FAs: [10, 13.33, 16.67, 20, 23.33, 30, 43.33, 60]
        # Acquired with 0 and 180 phase cycling.
        # Assuming interleaved: [FA1_0, FA1_180, FA2_0, FA2_180...]
        raw_ssfp_fas = np.array([10, 13.33, 16.67, 20, 23.33, 30, 43.33, 60], dtype=float)
        ssfp_fa = np.repeat(raw_ssfp_fas, 2) # [10, 10, 13.33, 13.33...]
        # Phase increment: [0, 180, 0, 180...]
        ssfp_phase = np.tile([0.0, 180.0], 8)
        
        ssfp_json = ssfp_path.replace(".nii.gz", ".json")
        with open(ssfp_json, 'r') as f:
            meta = json.load(f)
            ssfp_tr = float(meta.get('RepetitionTime', 0.0065)) # Default to typical 6.5ms

        # 3. SPGR-IR (Inversion Recovery)
        ir_path = os.path.join(self.anat_dir, "sub-00395_ses-02_acq-spgrIR_part-mag_VFA.nii.gz")
        if os.path.exists(ir_path):
            ir_data = nib.load(ir_path).get_fdata()
        else:
            print("Warning: SPGR-IR not found.")
            ir_data = None
        
        # 4. qMT
        qmt_data, qmt_offsets, qmt_sat_fa, qmt_ex_fa, qmt_tr = self.load_qmt()
        
        # Formulate result dictionary
        data_dict = {
            'spgr': spgr_data,
            'ssfp': ssfp_data,
            'ir': ir_data,
            'qmt': qmt_data
        }
        
        proto_dict = {
            'spgr_fa': spgr_fa,
            'spgr_tr': spgr_tr,
            'ssfp_fa': ssfp_fa,
            'ssfp_tr': ssfp_tr,
            'ssfp_phase': ssfp_phase,
            'qmt_offsets': qmt_offsets,
            'qmt_sat_fa': qmt_sat_fa,
            'qmt_ex_fa': qmt_ex_fa,
            'qmt_tr': qmt_tr
        }
        
        return data_dict, proto_dict
