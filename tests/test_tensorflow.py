from unittest import TestCase
from tests.utils import duration
from utils import flatten

import tensorflow as tf
import numpy as np


class TestTensorflow(TestCase):
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

        The former approach takes more time as it seems that TF constructs graph for each point in this case.
        :return:
        """
        with tf.Session() as s:
            x = tf.placeholder(dtype=tf.float32, shape=(10, 1))
            f = lambda x: tf.pow(x, 3)

            feed = [[1.2], [2.2], [3.1], [0.2], [-3], [2], [5], [51], [0], [10]]

            x1 = tf.reshape(x, [10])
            case1 = lambda: s.run(tf.diag_part(tf.squeeze(tf.hessians(f(x1), x1), axis=0)), {x: feed})
            duration(case1, "for all points at once. First run.")
            res1 = duration(case1, "for all points at once. Second run.")

            # four times faster
            case2 = lambda: s.run(tf.map_fn(lambda e: tf.hessians(f(e), e)[0][0], x), {x: feed})
            duration(case2, "in each point separetely. First run.")
            res2 = duration(case2, "in each point separetely. First run.")

            self.assertSequenceEqual(list(flatten(res1)), list(flatten(res2)))
