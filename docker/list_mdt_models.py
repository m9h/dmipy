import mdt

import collections
import collections.abc

# Monkeypatch for Python 3.10+
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping

# Initialize MDT (listing models requires initialization)
# user settings should already be initialized in Dockerfile
# mdt.init.init_mdt() # This might be redundant or fail if interactive

print("MDT Version:", mdt.__version__)

models = mdt.get_models_list()
print(f"\nFound {len(models)} registered models:")
for m_name in models:
    try:
        # Some models might need specific instantiation
        model_cls = mdt.get_model(m_name)
        # Instantiate
        model = model_cls()
        params = model.get_free_param_names()
        print(f"- {m_name}: {params}")
    except Exception as e:
        print(f"- {m_name}: [Error instantiating: {e}]")

print("\nListing Composite/Component Models (if discoverable):")
# Try to find standard components often used
components = ['Stick', 'Ball', 'Zeppelin', 'RestrictedCylinder', 'TortuousZeppelin']
for c in components:
    try:
        # Check if they are available as get_model or get_component
        # Note: MDT 1.2.7 might not expose them directly as 'models' if they are just building blocks
        pass 
    except:
        pass
