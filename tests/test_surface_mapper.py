
import sys
from unittest.mock import MagicMock

# Mock JAX, NumPy, JAXopt and other dependencies before importing dmipy_jax
sys.modules['jax'] = MagicMock()
sys.modules['jax.numpy'] = MagicMock()
sys.modules['jax.scipy'] = MagicMock()
sys.modules['jax.scipy.stats'] = MagicMock()
sys.modules['jax.scipy.linalg'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['jaxopt'] = MagicMock()

import os
import unittest
from unittest.mock import patch, MagicMock

# Now it should be safe to import
from dmipy_jax.viz.surface_mapper import map_to_surface, generate_pysurfer_script, visualize_on_surface

class TestSurfaceMapper(unittest.TestCase):
    @patch('dmipy_jax.viz.surface_mapper.subprocess.run')
    @patch('dmipy_jax.viz.surface_mapper.Path')
    def test_map_to_surface_command(self, mock_path, mock_subprocess):
        # Setup mocks
        mock_volume = MagicMock()
        mock_volume.resolve.return_value = mock_volume
        mock_volume.exists.return_value = True
        mock_volume.parent = MagicMock()
        mock_volume.stem = "test_vol"
        mock_volume.__str__.return_value = "/path/to/volume.nii.gz"
        
        mock_path.return_value = mock_volume
        
        # Call function
        output = map_to_surface("/path/to/volume.nii.gz", "lh", subject="BigMac")
        
        # Verify subprocess call
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        
        expected_cmd_start = ["mri_vol2surf", "--mov", "/path/to/volume.nii.gz", "--hemi", "lh"]
        
        # Check integrity of command construction
        self.assertEqual(cmd[:5], expected_cmd_start)
        self.assertIn("--surf", cmd)
        self.assertIn("white", cmd)
        self.assertIn("--projfrac", cmd)
        self.assertIn("0.5", cmd)
        self.assertIn("--regheader", cmd)
        self.assertIn("BigMac", cmd)

    def test_generate_pysurfer_script(self):
        # Generate script
        script_path = generate_pysurfer_script("overlay.mgh", "lh", subject="BigMac")
        
        # Read content
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Verify content
        self.assertIn("from surfer import Brain", content)
        self.assertIn("subject = 'BigMac'", content)
        self.assertIn("brain.add_overlay('overlay.mgh'", content)
        
        # Cleanup
        os.remove(script_path)

if __name__ == '__main__':
    unittest.main()
