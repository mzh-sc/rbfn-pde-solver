from unittest import TestCase
from collections import Iterable

import tensorflow as tf
import math as math
import rbf_network as rbfn


class TestNetwork(TestCase):
    def test_y(self):
        with tf.Session() as s:
            #1-D
            rbf1 = rbfn.Gaussian()
            rbf1.center = tf.Variable([1.5], dtype=tf.float64)
            rbf1.parameters = tf.Variable(1.0, dtype=tf.float64)

            rbf2 = rbfn.Gaussian()
            rbf2.center = tf.Variable([1.2], dtype=tf.float64)
            rbf2.parameters = tf.Variable(0.1, dtype=tf.float64)

            nn = rbfn.Network([rbf1, rbf2])
            nn.weights = tf.Variable([1.0, 0.5], dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(dtype=tf.float64)
            self.assertEqual(s.run(nn.y(x), {x: [1.0]}),
                             1.0 * math.exp(-(1.0 - 1.5)**2 / (2 * 1.0**2)) +
                             0.5 * math.exp(-(1.0 - 1.2)**2 / (2 * 0.1**2)))

            # 2-D
            rbf1 = rbfn.Gaussian()
            rbf1.center = tf.Variable([1.5, 2], dtype=tf.float64)
            rbf1.parameters = tf.Variable(1.0, dtype=tf.float64)

            rbf2 = rbfn.Gaussian()
            rbf2.center = tf.Variable([1.2, 1.1], dtype=tf.float64)
            rbf2.parameters = tf.Variable(0.1, dtype=tf.float64)

            nn = rbfn.Network([rbf1, rbf2])
            nn.weights = tf.Variable([1.0, 0.5], dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(dtype=tf.float64)
            self.assertEqual(s.run(nn.y(x), {x: [1.0, 2.0]}),
                             1.0 * math.exp(-((1.0 - 1.5) ** 2 + (2.0 - 2) ** 2) / (2 * 1.0 ** 2)) +
                             0.5 * math.exp(-((1.0 - 1.2) ** 2 + (2.0 - 1.1) ** 2) / (2 * 0.1 ** 2)))


    def test_variables_aggregation(self):
        """
        The purpose is not to use flatter to get gradient vector, i.e. iso
            [[dw1, dw2,...], [rbf1.c1, rbf1.c2], rbf1.a, [rbf1.c1, rbf1.c2], rbf1.a,... ]
            to get
            [dw1, dw2,..., rbf1.c1, rbf1.c2, rbf1.a, rbf1.c1, rbf1.c2, rbf1.a,... ]
        To do that we have to include convertion in computation graph.

        At the beginning I tried to use
                gradient = tf.gradients(f.y(x), [tf.reshape(e, []) for e in tf.split(f.center, num_or_size_splits=2, axis=0)] + [f.a])
            but it returned None type exception (see https://stackoverflow.com/questions/44836859/why-does-tensorflow-reshape-tf-reshape-break-the-flow-of-gradients)
            as tensorflow does not have a graph to convert tensors  center, a to new tensor [c1, c2, a,...]
        :return:
        """
        with tf.Session() as s:
            #---------- working example without aggregation
            rbf1 = rbfn.Gaussian()
            rbf1.center = tf.Variable([1.5], dtype=tf.float64)
            rbf1.parameters = tf.Variable(0.1, dtype=tf.float64)

            rbf2 = rbfn.Gaussian()
            rbf2.center = tf.Variable([1.2], dtype=tf.float64)
            rbf2.parameters = tf.Variable(0.1, dtype=tf.float64)

            nn = rbfn.Network([rbf1, rbf2])
            nn.weights = tf.Variable([1.0, 0.5], dtype=tf.float64)

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(dtype=tf.float64)
            gradient = tf.gradients(nn.y(x), [nn.weights, rbf1.center, rbf1.parameters, rbf2.center, rbf2.parameters])
            res_gradient = s.run(gradient, {x: [1.0]});
            self.assertEqual(len(res_gradient), 5)
            self.assertEqual(len(res_gradient[0]), 2)
            self.assertEqual(len(res_gradient[1]), 1)
            self.assertTrue(not isinstance(res_gradient[2], Iterable))

            #---------- not working example with aggregation
            gradient = tf.gradients(nn.y(x), [tf.reshape(e, []) for e in tf.split(nn.weights, num_or_size_splits=2, axis=0)] +  #tf.Variable([1.0, 0.5], dtype=tf.float64) - shape (2,) to [(),()]
                                    [tf.reshape(rbf1.center, []), rbf1.parameters, tf.reshape(rbf2.center, []), rbf2.parameters]) #list of 6 tensor with () shape

            #implemented throw __enter__ __exit___
            with self.assertRaises(TypeError):
                    s.run(gradient, {x: [1.0]}) #TypeError: Fetch argument None has invalid type <class 'NoneType'>


            #----------  working example with aggregation
            weights = tf.Variable([1.0, 0.5], dtype=tf.float64)
            parameters = tf.Variable([1.5, 1.0, 1.2, 0.1], dtype=tf.float64)

            parameters_per_rbf = tf.reshape(parameters, [-1, 2]) #[[r1.c, r1.a], [r2.c, r2.a]]
            rbf1 = rbfn.Gaussian()
            rbf1.center = tf.reshape(parameters_per_rbf[0][0], [1]) #tf.Variable([1.5], dtype=tf.float64)
            rbf1.parameters = parameters_per_rbf[0][1] #tf.Variable(0.1, dtype=tf.float64)

            rbf2 = rbfn.Gaussian()
            rbf2.center = tf.reshape(parameters_per_rbf[1][0], [1])
            rbf2.parameters = parameters_per_rbf[1][1]

            nn = rbfn.Network([rbf1, rbf2])
            nn.weights = weights

            s.run(tf.global_variables_initializer())

            x = tf.placeholder(dtype=tf.float64)
            self.assertEqual(s.run(nn.y(x), {x: [1.0]}),
                             1.0 * math.exp(-(1.0 - 1.5)**2 / (2 * 1.0**2)) +
                             0.5 * math.exp(-(1.0 - 1.2)**2 / (2 * 0.1**2)))

            gradient = tf.gradients(nn.y(x), [weights, parameters])
            res_gradient = s.run(gradient, {x: [1.0]});
            self.assertEqual(len(res_gradient), 2)
            self.assertEqual(len(res_gradient[0]), 2)
            self.assertEqual(len(res_gradient[1]), 4)



