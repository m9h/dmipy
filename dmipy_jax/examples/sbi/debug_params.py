import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.examples.sbi.train_noddi import get_noddi_model

model = get_noddi_model()
print("Model Parameter Names:")
print(model.parameter_names)
