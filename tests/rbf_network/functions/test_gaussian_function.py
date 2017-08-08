from unittest import TestCase

from tests.utils import utils

import tensorflow as tf
import math
import utils
import rbf_network as rbfn

class TestGaussianFunction(TestCase):
    delta = 1e-5

    def test_y(self):
        with tf.Session() as s:
            f = rbfn.GaussianFunction()

            # case 1
            f.center = tf.Variable([1.5])  # [1.5], constant is ok
            f.a = tf.Variable(1.0)
            s.run(tf.global_variables_initializer())

            x = tf.placeholder(rbfn.type)
            self.assertAlmostEqual(s.run(f.y(x), {x: [2.5]}), math.exp(-(2.5 - 1.5) ** 2 / (2 * 1 ** 2)),
                                   delta=self.delta)

            # case 2
            f.center = tf.Variable([1.5, 2.5])
            f.a = tf.Variable([1.2])
            s.run(tf.global_variables_initializer())

            x = tf.placeholder(rbfn.type)
            self.assertAlmostEqual(s.run(f.y(x), {x: [2.5, 2.3]}), math.exp(-((2.5 - 1.5) ** 2 + (2.3 - 2.5) ** 2)
                                                                            / (2 * 1.2 ** 2)), delta=self.delta)

    def test_y_gradient(self):
        with tf.Session() as s:
            f = rbfn.GaussianFunction()
            f.center = tf.Variable([1.5, 2.5], dtype=tf.float64)  # float32 e-8 precision
            f.a = tf.Variable(1.2, dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(tf.float64)

            gradient = tf.gradients(f.y(x), [f.center, f.a])

            y = math.exp(-((2.5 - 1.5) ** 2 + (2.3 - 2.5) ** 2) / (2 * 1.2 ** 2))
            self.assertSequenceEqual(list(utils.flatten(s.run(gradient, {x: [2.5, 2.3]}))),
                                     [TestGaussianFunction.dydc1(1.5, 2.5, 1.2, 2.5, 2.3),
                                      TestGaussianFunction.dydc2(1.5, 2.5, 1.2, 2.5, 2.3),
                                      TestGaussianFunction.dyda(1.5, 2.5, 1.2, 2.5, 2.3)])

    def test_y_gradient_performance(self):
        with tf.Session() as s:
            f = rbfn.GaussianFunction()
            f.center = tf.Variable([1.5, 2.5], dtype=tf.float64)
            f.a = tf.Variable(1.2, dtype=tf.float64)

            s.run(tf.global_variables_initializer())


            x = tf.placeholder(tf.float64)

            gradient = tf.gradients(f.y(x), [f.center, f.a])

            utils.duration(lambda: s.run(gradient, {x: [2.5, 2.3]}))
            utils.duration(lambda: s.run(gradient, {x: [0.8, 0.3]}))

            s.run(f.center.assign([1.4, 1.2]))
            utils.duration(lambda: s.run(gradient, {x: [0.8, 0.3]}))

            utils.duration(lambda: [TestGaussianFunction.dydc1(1.5, 2.5, 1.2, 2.5, 2.3),
                                    TestGaussianFunction.dydc2(1.5, 2.5, 1.2, 2.5, 2.3),
                                    TestGaussianFunction.dyda(1.5, 2.5, 1.2, 2.5, 2.3)])

    def y(c1, c2, a, x1, x2):
        return math.exp(-((x1 - c1) ** 2 + (x2 - c2) ** 2) / (2 * a ** 2))

    def dydc1(c1, c2, a, x1, x2):
        return TestGaussianFunction.y(c1, c2, a, x1, x2) * \
               -1 / (2 * a ** 2) * \
               2 * (x1 - c1) * \
               -1

    def dydc2(c1, c2, a, x1, x2):
        return TestGaussianFunction.y(c1, c2, a, x1, x2) * \
               -1 / (2 * a ** 2) * \
               2 * (x2 - c2) * \
               -1

    def dyda(c1, c2, a, x1, x2):
        return TestGaussianFunction.y(c1, c2, a, x1, x2) * \
               -((x1 - c1) ** 2 + (x2 - c2) ** 2) * \
               -2 / (2 * a**3)



