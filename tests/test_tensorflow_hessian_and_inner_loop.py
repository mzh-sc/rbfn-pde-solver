import tensorflow as tf
import numpy as np
import unittest as ut


class TestTensorflowHessianAndInnerLoop(ut.TestCase):
    def test_try_inner_loop_and_hessian(self):
        with tf.Session() as s:
            params = tf.Variable(np.array([0, 1, 2, 3], dtype=np.float64))
            s.run(tf.initialize_all_variables())

            def inner_loop(x, params):
                # map_fn using inner loop. Currently it is not supported by Hessian()
                # mostprobably at the first iteration of map_fn it creates hessian tensors and assign them to glob_var
                # then it tries to reuse them
                return tf.reduce_sum(tf.map_fn(lambda param: x * x * param, params))

            x = tf.placeholder(dtype=tf.float64, shape=(1,))
            with self.assertRaises(ValueError):
                hessian = tf.hessians(inner_loop(x, params), x)
                result = s.run(hessian, {x: [1.0]});

                print(result)

    def test_try_get_hessian_via_gradients(self):
        x = tf.placeholder(dtype=tf.float64, shape=(2,))
        y =  0.2 * x[0] * x[0] * x[1] + x[1] * x[1] * x[0]

        with tf.Session() as s:
            print('hessian %s' % s.run(tf.hessians(y, x)[0], feed_dict={x: [2, 1]}))
            grad = tf.gradients(y, x)[0]
            dx1 = grad[0]
            dx2 = grad[1]
            print('gr(gr) %s' % s.run(tf.gradients(dx1, x)[0][0], feed_dict={x: [2, 1]}))
            print('gr(gr) %s' % s.run(tf.gradients(dx2, x)[0][1], feed_dict={x: [2, 1]}))

    def test_try_get_hessian_via_gradients_and_map_fn(self):
        x = tf.placeholder(dtype=tf.float64, shape=(2,))
        p = tf.get_variable(name='p', dtype=tf.float64,
                            initializer=tf.constant(1, dtype=tf.float64, shape=(1,)))
        y = tf.reduce_sum(tf.map_fn(lambda e: e * e * p * p, x))

        with tf.Session() as s:
            tf.initialize_all_variables()
            # not supported now. see https://github.com/tensorflow/tensorflow/issues/15219
            # print('hessian %s' % s.run(tf.hessians(y, p)[0], feed_dict={x: [2, 1]}))
            # print('hessian %s' % s.run(tf.hessians(y, x)[0], feed_dict={x: [2, 1]}))

            # not supported now. see https://github.com/tensorflow/tensorflow/issues/15219
            with self.assertRaisesRegex(TypeError, "Second-order gradient for while loops not supported."):
                grad = tf.gradients(y, x)[0]
                dx1 = grad[0]
                dx2 = grad[1]
                print('gr(gr) %s' % s.run(tf.gradients(dx1, x)[0][0], feed_dict={x: [2, 1]}))
                print('gr(gr) %s' % s.run(tf.gradients(dx2, x)[0][1], feed_dict={x: [2, 1]}))