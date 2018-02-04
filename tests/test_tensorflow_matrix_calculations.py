import tensorflow as tf
import unittest as ut
import numpy as np

from rbf_network import Gaussian


class TestTensorflowMatrixCalculations(ut.TestCase):
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

