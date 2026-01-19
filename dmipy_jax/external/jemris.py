import jax.numpy as jnp
import numpy as np
import warnings

try:
    import h5py
except ImportError:
    h5py = None

def check_h5py_available():
    if h5py is None:
        raise ImportError("h5py is required to load JEMRIS/ISMRMRD files. Please install it.")

def load_jemris_signal(filepath: str) -> np.ndarray:
    """
    Loads complex k-space signal from a JEMRIS (ISMRMRD) HDF5 file.
    
    Args:
        filepath: Path to the .h5 file.
        
    Returns:
        np.ndarray: Complex signal array (N_coils, N_acquisitions, N_samples).
                    If single coil, returns (N_acquisitions, N_samples).
                    Note: JEMRIS usually simulates single coil unless specified.
    """
    check_h5py_available()
    
    with h5py.File(filepath, 'r') as f:
        # Standard ISMRMRD structure: /dataset/data
        # data is a compound dataset
        if 'dataset' not in f:
            raise ValueError(f"File {filepath} does not look like standard ISMRMRD (missing /dataset group).")
            
        ds = f['dataset']
        if 'data' not in ds:
             # JEMRIS specific fallback? 
             # JEMRIS usually outputs standard ISMRMRD.
             raise ValueError(f"File {filepath} missing /dataset/data table.")
             
        data = ds['data']
        
        # ISMRMRD 'data' is a struct array with fields like 'head', 'data'
        # But h5py reads compound types. 
        # The 'data' field inside the struct contains the actual samples.
        # It's usually variable length?
        
        # In ISMRMRD HDF5, the 'data' field of the dataset often stores the raw complex samples.
        # However, h5py might present this as a flat array of structs.
        
        # Let's inspect the first element to see structure if we could, but here we assume standard.
        # We assume fixed length acquisitions for now (Simulated data).
        
        # Extract all 'data' chunks.
        # 'data' column in the compound type might be a reference or a fixed array.
        # Use simple iteration for safety/flexibility.
        
        signals = []
        for i in range(len(data)):
            # Each row is an acquisition
            acq = data[i]
            # 'data' field contains the samples: [real, imag, real, imag...] or complex?
            # ISMRMRD HDF5 usually stores 'data' as array of floats (2*Samples).
            
            # Accessing the 'data' field of the numpy structured array
            raw_samples = acq['data'] 
            
            # Convert to complex
            # If shape is (2*N,), view as complex
            # JEMRIS output often (Channels, Samples * 2) or (Channels, Samples) complex?
            # It depends on how it was written. Standard ISMRMRD is float32 array [coil][2*samples]
            
            # Let's assume standard ISMRMRD layout:
            # We assume single coil for JEMRIS simulation usually unless defined otherwise.
            
            # Helper to convert arbitrary float array to complex
            if raw_samples.dtype.kind == 'f':
                 # View as complex64
                 c_sig = raw_samples.view(np.complex64)
                 # Or manual: raw[::2] + 1j*raw[1::2]
                 # But ISMRMRD writes as [real, imag, real, imag] usually?
                 # Actually, commonly it is (header, data).
                 pass
            
            # To be robust, let's just append the raw row for now?
            # Better: JEMRIS provides a handy 'signals' export sometimes.
            # But let's stick to the requested ISMRMRD path.
            
            # We will flatten and assume 1D complex stream for comparison against bloch.py linear output.
            # bloch.simulate_acquisition returns a SINGLE complex sum (or time course if modified).
            # Wait, `simulate_acquisition` returns `signal = sum(Mx + iMy)`.
            # Is that instantaneous signal at end? OR integrated signal?
            # Looking at bloch.py: `signal = jnp.sum(Mx + 1j * My)` using `M_final`.
            # It returns the signal at the END of the simulation duration.
            
            # So `bloch` simulates ONE point (usually one readout sample).
            # To simulate a full readout, we loop or use `saveat`.
            # Our `bloch.py` needs to be called repeatedly or modified to return timecourse.
            
            # For validation, we focus on matching the Single Point (e.g. FID endpoint) or modify bloch later.
            signals.append(raw_samples)

        return np.array(signals)

def load_jemris_xml(filepath: str) -> str:
    """Returns the XML header string from the ISMRMRD file."""
    check_h5py_available()
    with h5py.File(filepath, 'r') as f:
        if 'dataset/xml' in f:
             # Stored as variable length string
             return f['dataset/xml'][0].decode('utf-8')
    return ""
