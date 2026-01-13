
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from dmipy.core.jax_modeling_framework import JAXMultiCompartmentModel
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

class TestJAXMultiCompartmentModel(unittest.TestCase):
    def test_instantiation(self):
        # Dummy model class simulating a JAX-compatible sub-model
        class DummyModel:
            def __call__(self, acquisition_data, **kwargs):
                return jnp.ones_like(acquisition_data['bvalues'])

        model = JAXMultiCompartmentModel([DummyModel()])
        self.assertIsInstance(model, JAXMultiCompartmentModel)

    def test_prediction_skeleton(self):
        # Create a dummy acquisition scheme
        bvalues = np.array([0, 1000, 2000])
        gradient_directions = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        delta = 0.01
        Delta = 0.03
        scheme = acquisition_scheme_from_bvalues(bvalues, gradient_directions, delta, Delta)

        # Dummy model
        class DummyModel:
             pass

        model = JAXMultiCompartmentModel([DummyModel()])

        # Calling __call__ currently runs _predict which returns zeros in the skeleton
        # parameters would be empty dict
        params = {}
        signal = model(scheme, params)

        self.assertEqual(signal.shape, bvalues.shape)
        # JAX arrays usually compare true with numpy arrays, but let's be safe
        np.testing.assert_array_equal(signal, np.zeros_like(bvalues))

if __name__ == '__main__':
    unittest.main()
