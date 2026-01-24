
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import unxt
import equinox as eqx
from dmipy_jax.inverse.amico import AMICOSolver

# Dummy Model
def dummy_stick(params, acquisition):
    bvals = acquisition['bvals']
    D = params['diffusivity']
    return jnp.exp(-bvals * D)

class TestAMICOStrict:
    def test_adjoint_property(self):
        """
        Verify <Ax, y> = <x, A.H y> for the dictionary operator.
        """
        acquisition = {'bvals': jnp.array([0., 1000., 2000.])}
        dict_params = {'diffusivity': jnp.array([1e-3, 2e-3])}

        solver = AMICOSolver.create(dummy_stick, acquisition, dict_params)
        A = solver.dict_operator

        print(f"DEBUG: A.shape = {A.shape}, type = {type(A.shape)}")

        # Random vectors
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        # Scico LinearOperator shape is (output_shape, input_shape) if it preserves structure?
        # Debug output showed ((3,), (2,)).
        # We need input shape for x, output shape for y.
        # A.shape[1] is input shape tuple. A.shape[0] is output shape tuple.
        x = jax.random.normal(k1, A.shape[1])
        y = jax.random.normal(k2, A.shape[0])

        # Forward and Adjoint
        Ax = A(x)
        Aty = A.adj(y)

        dot1 = jnp.vdot(Ax, y)
        dot2 = jnp.vdot(x, Aty)

        print(f"Dot1: {dot1}, Dot2: {dot2}, Diff: {jnp.abs(dot1 - dot2)}")
        assert jnp.allclose(dot1, dot2, atol=1e-4)

    def test_unit_sandwich(self):
        """
        Verify that passing quantities works and returns quantities.
        """
        # Units
        u_bval = unxt.unit("s/mm^2")
        u_diff = unxt.unit("mm^2/s")

        bvals_raw = jnp.array([0., 1000., 2000.])
        bvals = unxt.Quantity(bvals_raw, u_bval)
        acquisition = {'bvals': bvals}

        def model_with_units(params, acq):
            b = acq['bvals']
            d = params['diffusivity']
            # b*d is dimensionless quantity.
            # JAX exp doesn't handle Quantity. Strip it.
            arg = b * d
            if hasattr(arg, 'unit'):
                 # Ensure it's dimensionless
                 u_dim = unxt.unit("m") / unxt.unit("m")
                 arg = arg.ustrip(u_dim)
            return jnp.exp(-arg)

        dict_params = {
            'diffusivity': unxt.Quantity(jnp.array([1e-3, 2e-3]), u_diff)
        }

        solver = AMICOSolver.create(model_with_units, acquisition, dict_params)

        data_raw = jnp.array([1.0, 0.36, 0.13])
        u_sig = unxt.unit("V")
        data = unxt.Quantity(data_raw * 10.0, u_sig)

        weights = solver.fit(data, lambda_reg=0.0)

        assert hasattr(weights, 'unit')
        u_dim = unxt.unit("m") / unxt.unit("m")
        w_mag = weights.ustrip(u_dim)
        assert jnp.allclose(w_mag[0], 10.0, atol=1.0)

    def test_jit_safety(self):
        """
        Verify that solver can be JIT compiled.
        """
        acquisition = {'bvals': jnp.array([0., 1000.])}
        dict_params = {'diffusivity': jnp.array([1e-3, 2e-3])}
        solver = AMICOSolver.create(dummy_stick, acquisition, dict_params)

        data = jnp.array([1.0, 0.5])

        @jax.jit
        def run_fit(d):
            return solver.fit(d)

        w = run_fit(data)
        assert w.shape == (2,)

if __name__ == "__main__":
    t = TestAMICOStrict()
    t.test_adjoint_property()
    t.test_unit_sandwich()
    t.test_jit_safety()
    print("Strict tests passed.")
