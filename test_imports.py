
import sys
print("Import test starting...")
sys.stdout.flush()

try:
    from dmipy_jax.io.mesh import create_sphere_mesh
    print("Imported io.mesh")
except ImportError as e:
    print("Failed io.mesh:", e)

try:
    from dmipy_jax.simulation.mesh_sim import MatrixFormalismSimulator
    print("Imported mesh_sim")
except ImportError as e:
    print("Failed mesh_sim:", e)

try:
    from dmipy_jax.signal_models.sphere_models import SphereCallaghan
    print("Imported sphere_models")
except ImportError as e:
    print("Failed sphere_models:", e)

print("Done.")
