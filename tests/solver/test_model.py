from unittest import TestCase

import rbf_network as rbf
import solver as sol
import tensorflow as tf
import numpy as np


class TestModel(TestCase):
    def test_try_model_workflow(self):
        model = sol.Model()
        model.add_rbf(1, rbf_name='gaussian', center=[0, 0], parameters=[1])

        with tf.Session() as s:
            var = tf.Variable(np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=rbf.type.as_numpy_dtype))
            s.run(tf.initialize_all_variables())

            model.compile()
            print(tf.map_fn(lambda x: model.network.y(x), var))

    def test_compile(self):
        model = sol.Model()
        model.add_rbf(1, rbf_name='gaussian', center=[1, 0.4], parameters=[1])
        model.add_rbf(2.5, rbf_name='gaussian', center=[1.2, 2.3], parameters=[0.5])
        model.compile()

        with tf.Session() as s:
            s.run(tf.initialize_all_variables())

            np.testing.assert_almost_equal(model.weights.eval().tolist(), [1, 2.5])
            np.testing.assert_almost_equal(model.centers.eval().tolist(), [[1, 0.4], [1.2, 2.3]])
            np.testing.assert_almost_equal(model.parameters.eval().tolist(), [[1], [0.5]])

            self.assertFalse(model.network is None)
            np.testing.assert_almost_equal(model.network.weights.eval().tolist(), [1, 2.5])
            self.assertEqual(len(model.network._functions), 2)

            rbf1 = next(iter(model.network))
            np.testing.assert_almost_equal(rbf1.center.eval().tolist(), [1, 0.4])
            np.testing.assert_almost_equal(rbf1.parameters.eval().tolist(), [1])
