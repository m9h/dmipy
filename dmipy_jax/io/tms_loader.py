import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class TMSLoader:
    """
    Loader for TMS-EEG latency data, designed for the WAND dataset context.
    
    Since TMS/EEG data often resides in non-BIDS formats (.mat, .dat) or derivative CSVs,
    this loader standardizes the interface to retrieve empirical latencies between brain regions.
    """
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Root directory containing data (e.g., 'data/wand').
        """
        self.base_path = base_path

    def load_latencies(self, subject_id: str, session: str = "ses-02") -> Dict[Tuple[str, str], float]:
        """
        Loads empirical conduction latencies (in ms) between stimulated and recorded regions.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-00395').
            session: Session identifier.
            
        Returns:
            Dictionary mapping (SourceRegion, TargetRegion) -> Latency_ms.
            Region names should match the atlas used in structural connectivity.
        """
        # Placeholder logic: In a real scenario, this would parse a .mat file 
        # (e.g., using scipy.io.loadmat) from the 'phenotype' or 'derivatives' folder.
        # WAND specific path structure often puts these in 'phenotype' or custom derivative folders.
        
        # We will look for a CSV first (easier interoperability), then fallback to mock if missing.
        potential_path = os.path.join(self.base_path, "derivatives", "tms_eeg", subject_id, f"{subject_id}_{session}_latencies.csv")
        
        if os.path.exists(potential_path):
            return self._load_from_csv(potential_path)
            
        # Fallback/Mock for Agent Development
        # Returns a small set of plausible latencies for testing the calibration agent.
        # Latencies typically 5-20 ms for inter-hemispheric or long-range tracts.
        print(f"Warning: TMS latency file not found at {potential_path}. Using mock data.")
        return self._get_mock_latencies()

    def _load_from_csv(self, file_path: str) -> Dict[Tuple[str, str], float]:
        """
        Expects CSV with columns: Source, Target, Latency_ms
        """
        df = pd.read_csv(file_path)
        latencies = {}
        for _, row in df.iterrows():
            key = (str(row['Source']), str(row['Target']))
            latencies[key] = float(row['Latency_ms'])
        return latencies

    def _get_mock_latencies(self) -> Dict[Tuple[str, str], float]:
        """
        Returns synthetic ground truth latencies for development.
        """
        return {
            ("L_M1", "R_M1"): 12.5,  # Inter-hemispheric motor
            ("L_DLPFC", "L_M1"): 8.2, # Frontal-motor
            ("R_DLPFC", "R_M1"): 8.4,
            ("L_V1", "L_M1"): 18.0,   # Visual-motor (long)
        }
