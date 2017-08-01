from unittest import TestCase

import tensorflow as tf
import math as math

from rbf_network.functions.gaussian_function import GaussianFunction
from rbf_network.network import Network
from rbf_network.consts import type


class TestNetwork(TestCase):
    def test_y(self):
        with tf.Session() as s:
            #1-D
            rbf1 = GaussianFunction()
            rbf1.center = tf.Variable([1.5], dtype=tf.float64)
            rbf1.a = tf.Variable(1.0, dtype=tf.float64)

            rbf2 = GaussianFunction()
            rbf2.center = tf.Variable([1.2], dtype=tf.float64)
            rbf2.a = tf.Variable(0.1, dtype=tf.float64)

            nn = Network([rbf1, rbf2])
            nn.weights = tf.Variable([1.0, 0.5], dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(dtype=tf.float64)
            self.assertEqual(s.run(nn.y(x), {x: [1.0]}),
                             1.0 * math.exp(-(1.0 - 1.5)**2 / (2 * 1.0**2)) +
                             0.5 * math.exp(-(1.0 - 1.2)**2 / (2 * 0.1**2)))

            # 2-D
            rbf1 = GaussianFunction()
            rbf1.center = tf.Variable([1.5, 2], dtype=tf.float64)
            rbf1.a = tf.Variable(1.0, dtype=tf.float64)

            rbf2 = GaussianFunction()
            rbf2.center = tf.Variable([1.2, 1.1], dtype=tf.float64)
            rbf2.a = tf.Variable(0.1, dtype=tf.float64)

            nn = Network([rbf1, rbf2])
            nn.weights = tf.Variable([1.0, 0.5], dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(dtype=tf.float64)
            self.assertEqual(s.run(nn.y(x), {x: [1.0, 2.0]}),
                             1.0 * math.exp(-((1.0 - 1.5) ** 2 + (2.0 - 2) ** 2) / (2 * 1.0 ** 2)) +
                             0.5 * math.exp(-((1.0 - 1.2) ** 2 + (2.0 - 1.1) ** 2) / (2 * 0.1 ** 2)))

