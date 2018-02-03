from unittest import TestCase

from tests.utils import duration

import tensorflow as tf
import math
import utils
import rbf_network as rbfn


class TestGaussianFunction(TestCase):
    delta = 1e-5

    def test_y(self):
        with tf.Session() as s:
            f = rbfn.Gaussian()

            # case 1d
            center = tf.Variable([1.5])  # [1.5], constant is ok
            parameters = tf.Variable([1.0])
            s.run(tf.global_variables_initializer())

            x = tf.placeholder(rbfn.type, shape=(1,))
            self.assertAlmostEqual(s.run(f.y(x, center, parameters), {x: [2.5]}), math.exp(-(2.5 - 1.5) ** 2 / (2 * 1 ** 2)),
                                   delta=self.delta)

            # case 2d
            center = tf.Variable([1.5, 2.5])
            parameters = tf.Variable([1.2])
            s.run(tf.global_variables_initializer())

            x = tf.placeholder(rbfn.type, shape=(2,))
            self.assertAlmostEqual(s.run(f.y(x, center, parameters), {x: [2.5, 2.3]}), math.exp(-((2.5 - 1.5) ** 2 + (2.3 - 2.5) ** 2)
                                                                            / (2 * 1.2 ** 2)), delta=self.delta)

    def test_y_gradient(self):
        with tf.Session() as s:
            f = rbfn.Gaussian()

            center = tf.Variable([1.5, 2.5], dtype=tf.float64)  # float32 e-8 precision
            parameters = tf.Variable(1.2, dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(tf.float64, shape=(2,))

            gradient = tf.gradients(f.y(x, center, parameters), [center, parameters])

            y = math.exp(-((2.5 - 1.5) ** 2 + (2.3 - 2.5) ** 2) / (2 * 1.2 ** 2))
            self.assertSequenceEqual(list(utils.flatten(s.run(gradient, {x: [2.5, 2.3]}))),
                                     [TestGaussianFunction.__dydc1(1.5, 2.5, 1.2, 2.5, 2.3),
                                      TestGaussianFunction.__dydc2(1.5, 2.5, 1.2, 2.5, 2.3),
                                      TestGaussianFunction.__dyda(1.5, 2.5, 1.2, 2.5, 2.3)])

    def test_y_gradient_performance(self):
        """
        to compare performance of the Automatic differentiation with the Automatic differentiation
        :return:
        """
        with tf.Session() as s:
            f = rbfn.Gaussian()

            center = tf.Variable([1.5, 2.5], dtype=tf.float64)
            parameters = tf.Variable(1.2, dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(tf.float64, shape=(2,))

            gradient = tf.gradients(f.y(x, center, parameters), [center, parameters])

            duration(lambda: s.run(gradient, {x: [2.5, 2.3]}))
            duration(lambda: s.run(gradient, {x: [0.8, 0.3]}))

            s.run(center.assign([1.4, 1.2]))
            duration(lambda: s.run(gradient, {x: [0.8, 0.3]}))

            duration(lambda: [TestGaussianFunction.__dydc1(1.5, 2.5, 1.2, 2.5, 2.3),
                                    TestGaussianFunction.__dydc2(1.5, 2.5, 1.2, 2.5, 2.3),
                                    TestGaussianFunction.__dyda(1.5, 2.5, 1.2, 2.5, 2.3)])

    def __y(c1, c2, a, x1, x2):
        return math.exp(-((x1 - c1) ** 2 + (x2 - c2) ** 2) / (2 * a ** 2))

    def __dydc1(c1, c2, a, x1, x2):
        return TestGaussianFunction.__y(c1, c2, a, x1, x2) * \
               -1 / (2 * a ** 2) * \
               2 * (x1 - c1) * \
               -1

    def __dydc2(c1, c2, a, x1, x2):
        return TestGaussianFunction.__y(c1, c2, a, x1, x2) * \
               -1 / (2 * a ** 2) * \
               2 * (x2 - c2) * \
               -1

    def __dyda(c1, c2, a, x1, x2):
        return TestGaussianFunction.__y(c1, c2, a, x1, x2) * \
               -((x1 - c1) ** 2 + (x2 - c2) ** 2) * \
               -2 / (2 * a ** 3)