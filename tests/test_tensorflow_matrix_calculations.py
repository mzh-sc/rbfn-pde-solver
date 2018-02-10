import tensorflow as tf
import unittest as ut
import numpy as np

from rbf_network import Gaussian


class TestTensorflowMatrixCalculations(ut.TestCase):
    def test_gaussian(self):
        with tf.Session() as s:
            x = tf.placeholder(dtype=tf.float64, shape=(2,))
            centers = tf.Variable(np.array([[0.0, 0.0], [1, 1]], dtype=np.float64))
            parameters = tf.Variable(np.array([[1], [2]], dtype=np.float64))
            s.run(tf.initialize_all_variables())

            g = Gaussian()
            print(s.run(tf.pow(x - centers, 2), feed_dict={x: [1.0, 1.0]}))
            print(s.run(2 * parameters * parameters))


            print(s.run(tf.reduce_sum(tf.pow(x - centers, 2), axis=-1), feed_dict={x: [1.0, 1.0]}))
            print(s.run(2 * tf.squeeze(parameters, axis=-1) * tf.squeeze(parameters, axis=-1)))

            nominators = -tf.reduce_sum(tf.pow(x - centers, 2), axis=-1)

            a = tf.squeeze(parameters, axis=-1)
            denominators_a = 2 * a * a

            print(s.run(tf.exp(nominators / denominators_a),
                        feed_dict={x: [1.0, 1.0]}))

            print(s.run(tf.exp(nominators / denominators_a),
                        feed_dict={x: [2.0, 2.0]}))

            # may be in the future not to use map_fn and send list of x
            print("More complex - list of xs")

            x = tf.placeholder(dtype=tf.float64, shape=(3, 1, 2))

            nominators = -tf.reduce_sum(tf.pow(x - centers, 2), axis=-1)

            print(s.run(tf.exp(nominators / denominators_a), feed_dict={x: [[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]]}))

            #print(s.run(g.y(x, centers, parameters), feed_dict={x: [1.0]}))

    def test_op(self):
        with tf.Session() as s:
            print(s.run(tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) -
                        tf.constant([1.0, 1.0]))) #[[ 0.  0.] [ 1.  1.] [ 2.  2.]]

            print(s.run(tf.constant([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]]) -
                        tf.constant([[1.0, 1.0], [1.0, 1.0]]))) #[[[ 0.  0.] [ 0.  0.]] [[ 1.  1.] [ 1.  1.]] [[ 2.  2.] [ 2.  2.]]]

            print(s.run(tf.squeeze(tf.expand_dims([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], 1), 1)))

            c = tf.constant([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]])
            print(s.run(c[:, 0, 0] + c[:, 0, 1])) #[ 2.  4.  6.]

            # VERY IMPORTANT EXAMPLE: how to return to points based operations.
            # 1. create list of tensors per each coordinate
            # 2. apply operation using list indexes
            c = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            f = lambda x: x[0] + x[1]
            print(s.run(f(tf.split(c, num_or_size_splits=2, axis=1)))) #[[ 2.] [ 4.] [ 6.]]

            # the same, but manual
            print(s.run(f([c[:, 0], c[:,1]]))) #[ 2.  4.  6.]

            print(s.run(tf.split(c, num_or_size_splits=tf.shape(c)[-1], axis=1)))

    def test_op1(self):
        print(isinstance([1, 2].count()))