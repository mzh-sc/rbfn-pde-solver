from unittest import TestCase

from rbf_network import Gaussian
from tests.utils import duration
from utils import flatten

import tensorflow as tf
import numpy as np


class TestTensorflow(TestCase):
    def test_while_loop(self):
        def body(x):
            return x + 1

        def condition(x):
            return tf.reduce_sum(x) < 10

        x = tf.Variable(tf.constant(1, shape=[2, 2]))

        with tf.Session():
            tf.initialize_all_variables().run()
            result = tf.while_loop(condition, body, [x])
            print(result.eval())

    def test_inner_loop_and_hessian(self):
        with tf.Session() as s:
            var = tf.Variable(np.array([0, 1, 2, 3], dtype=np.float64))
            s.run(tf.initialize_all_variables())

            def inner_loop(x, p):
                return tf.reduce_sum(x * x * p)

            x = tf.placeholder(dtype=tf.float64, shape=(1,))
            hessian = tf.hessians(inner_loop(x, var), x)
            result = s.run(hessian, {x: [1.0]});

            print(result)

    def test_gaussian(self):
        with tf.Session() as s:
            x = tf.placeholder(dtype=tf.float64, shape=(1,))
            centers = tf.Variable(np.array([[0], [1]], dtype=np.float64))
            parameters = tf.Variable(np.array([[1], [2]], dtype=np.float64))
            s.run(tf.initialize_all_variables())

            g = Gaussian()
            print(s.run(tf.pow(x - centers, 2), feed_dict={x: [1.0]}))
            print(s.run(2 * parameters * parameters))


            print(s.run(tf.reduce_sum(tf.pow(x - centers, 2), axis=-1), feed_dict={x: [1.0]}))
            print(s.run(2 * tf.squeeze(parameters, axis=-1) * tf.squeeze(parameters, axis=-1)))


            print(s.run(tf.exp(-tf.reduce_sum(tf.pow(x - centers, 2), axis=-1) /
                        (2 * tf.squeeze(parameters, axis=-1) * tf.squeeze(parameters, axis=-1))), feed_dict={x: [1.0]}))

            nominators = -tf.reduce_sum(tf.pow(x - centers, 2), axis=-1)

            a = tf.squeeze(parameters, axis=-1)
            denominators_a = 2 * a * a

            y = tf.exp(nominators / denominators_a)
            print(s.run(tf.exp(nominators / denominators_a),
                        feed_dict={x: [1.0]}))

            # may be in the future not to use map_fn and send list of x
            print("More complex - list of xs")
            print(s.run(x - centers, feed_dict={x: [1.0]}))
            print(s.run(x - centers, feed_dict={x: [2.0]}))

            print(s.run(tf.reduce_sum(tf.pow(x - centers, 2), axis=-1), feed_dict={x: [1.0]}))
            print(s.run(tf.reduce_sum(tf.pow(x - centers, 2), axis=-1), feed_dict={x: [2.0]}))

            x = tf.placeholder(dtype=tf.float64, shape=(2, 1, 1))

            nominators = -tf.reduce_sum(tf.pow(x - centers, 2), axis=-1)

            print(s.run(tf.exp(nominators / denominators_a), feed_dict={x: [[[1.0]], [[2.0]]]}))

            #print(s.run(g.y(x, centers, parameters), feed_dict={x: [1.0]}))


    def test_try_concat(self):
        with tf.Session() as s:
            var1 = tf.Variable(np.array([[0, 0.1], [1, 1.1], [2, 2.1], [3, 3.1]], dtype=np.float64))
            s.run(tf.initialize_all_variables())

            print(s.run(var1[:, 0]))
            print(s.run(tf.concat((var1[:, 0], var1[:, 1]), axis=0)))

    def test_try_hessian(self):
        """
        Hessian calculations performance test
        1. Calculate hessian for all points at once
        2. Calculate Hessian in each point separately using map_fn

        The former approach takes more time as it seems that TF constructs graph for each point in this case and
        Hessian matrix is 10x10 size sparse matrix
        :return:
        """
        with tf.Session() as s:
            x = tf.placeholder(dtype=tf.float32, shape=(10, 1))
            f = lambda x: tf.pow(x, 3)

            feed = [[1.2], [2.2], [3.1], [0.2], [-3], [2], [5], [51], [0], [10]]

            x1 = tf.reshape(x, [10])
            # hessian function output has the shape (1, 10, 10)
            case1 = lambda: s.run(tf.diag_part(tf.squeeze(tf.hessians(f(x1), x1), {x: feed})))
            duration(case1, "for all points at once. First run.")
            res1 = duration(case1, "for all points at once. Second run.")

            # four times faster
            case2 = lambda: s.run(tf.map_fn(lambda e: tf.hessians(f(e), e)[0][0], x), {x: feed})
            duration(case2, "in each point separetely. First run.")
            res2 = duration(case2, "in each point separetely. First run.")

            expectedResult = [7.2, 13.2, 18.6, 1.2, -18, 12, 30, 306, 0, 60]
            np.testing.assert_almost_equal(list(flatten(res1)), expectedResult, decimal=1)
            np.testing.assert_almost_equal(list(flatten(res2)), expectedResult, decimal=1)

    def _getWhileTensor(self):
        """Creates and returns a tensor from a while context."""
        tensor1 = [] # python не разрешает создать переменную None и использовать ее внутри функции, поэтому использ. массив для хранения одной переменной

        def body(i):
            if not tensor1:
                tensor1.append(tf.constant(1, name='c1'))
            return tf.add(i, tensor1[0], name='add_')

        tf.while_loop(lambda i: i < 10, body, [0], name="w2")
        return tensor1[0]

    def testInvalidContextInWhile(self):
        j = tf.constant(0, name='c2')
        c0 = lambda i, j: j < (i + 1) * 10
        b0 = lambda i, j: (i, tf.add(j, 1, name='add2'))

        i = tf.constant(0, name='c1')
        c = lambda i: tf.less(i, 10)
        def b(e):
            tf.while_loop(c0, b0, [e, j], name='w2')
            return tf.add(e, 1, name='add1')

        r = tf.while_loop(c, b, [i], name='w1')
        with tf.Session() as s:
            print(s.run(r))
            print(i.eval())
            print(s.run(j))

        # Accessing  a  while loop tensor in a different while loop is illegal.
        while_tensor = self._getWhileTensor()

        # НЕЛЬЗЯ использовать тензор созданный внутри вложенного цикла как аргумент внешнего или в вне
        tf.while_loop(lambda i: i < 10, lambda i: while_tensor, [0], name='w3')

    def testValidWhileContext(self):
        # Accessing a tensor in a nested while is OK.
        def body(_):
            c = tf.constant(1)
            return tf.while_loop(lambda i: i < 10, lambda i: i + c, [0])

        with tf.Session() as s:
            print(s.run(tf.while_loop(lambda i: i < 5, body, [0])))

    def test_tf(self):
        x = tf.placeholder(dtype=tf.float64, shape=(None,2))
        op = lambda e: e[0] + e[1]
        with tf.Session() as s:
            print(s.run(op(x[:, 0] + x[:, 1],
                        feed_dict= { x: [[1, 2], [3, 4], [0, 0]] })))
