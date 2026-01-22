import jax.numpy as jnp
import numpy as np
import os
from typing import Optional, Any

class SBIdMRIOracle:
    """
    Wrapper for pre-trained PyTorch SBI models (SBI_dMRI).
    Allows JAX-based benchmarking suite to evaluate PyTorch models.
    """
    posterior: Any
    device: str
    
    def __init__(self, model_path: str, device: str = "cpu"):
        try:
            import torch
            from sbi.utils import user_input_checks
        except ImportError:
            raise ImportError("PyTorch and SBI are required for SBIdMRIOracle. Please install 'torch' and 'sbi'.")
        
        self.device = device
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        print(f"Loading SBI oracle from {model_path}...")
        # Load the pickle
        # Usually sbi objects are pickled with torch.load or pickle.load
        # SBI_dMRI usually saves the posterior object directly.
        
        # map_location
        self.posterior = torch.load(model_path, map_location=device)
        self.posterior.to(device)
        self.posterior.eval() # Ensure eval mode
        
    def sample(self, signal: jnp.ndarray, n_samples: int = 1000) -> jnp.ndarray:
        """
        Draw samples from the posterior given a signal.
        Args:
            signal: (N_features,) Observation
        Returns:
            samples: (N_samples, N_params)
        """
        import torch
        
        # Convert JAX/Numpy to Torch
        sig_np = np.array(signal)
        sig_torch = torch.from_numpy(sig_np).float().to(self.device).reshape(1, -1)
        
        # Sample
        # posterior.sample expects (num_samples, x=observation)
        # Note: sbi API changed over versions. usually .sample((n,), x=...) or .sample((n,), condition=...)
        # or .sample(n, x=...)
        
        with torch.no_grad():
            samps = self.posterior.sample((n_samples,), x=sig_torch)
            
        # Convert back
        return jnp.array(samps.cpu().numpy().squeeze(0)) # Remove batch dim if present?
        # Output of sample is usually (n_samples, param_dim).
        
    def log_prob(self, theta: jnp.ndarray, signal: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate log probability.
        Args:
            theta: (N_batch, N_params) or (N_params,)
        """
        import torch
        
        # Convert
        sig_np = np.array(signal)
        theta_np = np.array(theta)
        
        if theta_np.ndim == 1:
            theta_np = theta_np[None, :]
            
        sig_torch = torch.from_numpy(sig_np).float().to(self.device).reshape(1, -1)
        # Repeat signal for batch of theta
        sig_torch = sig_torch.repeat(theta_np.shape[0], 1)
        
        theta_torch = torch.from_numpy(theta_np).float().to(self.device)
        
        with torch.no_grad():
            lp = self.posterior.log_prob(theta_torch, x=sig_torch)
            
        return jnp.array(lp.cpu().numpy())

