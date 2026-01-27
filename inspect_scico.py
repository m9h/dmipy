
import scico
import scico.denoise
import sys

print(f"SCICO Version: {scico.__version__}")
print("\n--- scico.denoise content ---")
print(dir(scico.denoise))

print("\n--- BM4D check ---")
if hasattr(scico.denoise, 'bm4d'):
    print("scico.denoise.bm4d exists.")
    print(scico.denoise.bm4d.__doc__)
else:
    print("scico.denoise.bm4d NOT found.")
    
try:
    from scico.denoise import BM4D
    print("Imported BM4D class/function successfully.")
except ImportError:
    print("Could not direct import BM4D.")
