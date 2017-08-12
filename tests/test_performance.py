from unittest import TestCase

import tensorflow as tf
import time


class TestPerformance(TestCase):
    def test_hessian(self):
        with tf.Session() as s:
            x = tf.placeholder(dtype=tf.float32, shape=(10, 1))
            f = lambda x: tf.pow(x, 3)

            x1 = tf.reshape(x, [10])
            t_org = time.perf_counter()
            res1 = s.run(tf.diag_part(tf.squeeze(tf.hessians(f(x1), x1), axis=0)), {x: [[1.2], [2.2], [3.1], [0.2], [-3], [2], [5], [51], [0], [10]]})
            t_end = time.perf_counter()

            print('Duration of {}'.format(t_end - t_org))

            t_org = time.perf_counter()
            #three times faster
            res2 = s.run(tf.map_fn(lambda e:
                                  tf.hessians(f(e), e)[0][0], x),
                        {x: [[1.2], [2.2], [3.1], [0.2], [-3], [2], [5], [51], [0], [10]]})
            t_end = time.perf_counter()
            print('Duration of {}'.format(t_end - t_org))
