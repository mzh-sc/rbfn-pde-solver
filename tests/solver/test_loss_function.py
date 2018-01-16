from unittest import TestCase

import solver as slv
import tensorflow as tf

class TestLossFunction(TestCase):
    def test_compile(self):
        class Expando(object):
            pass
        model = Expando()
        model.network = Expando()
        model.network.y = lambda x: x[0] + x[1]

        problem = slv.Problem()
        problem.add_constrain('c1',
                              lambda y, x: y(x),
                              lambda x: x[0] + x[1])
        problem.add_constrain('c2',
                              lambda y, x: y(x),
                              lambda x: x[0] + x[1] - 2)
        problem.compile()

        loss = slv.LossFunction(model=model, problem=problem)
        loss.set_control_points('c1', 1, [[0, 1], [1, 2]])
        loss.set_control_points('c2', 2, [[0, 1], [1, 2]])
        loss.compile()

        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            error = s.run(loss.error, feed_dict=loss.feed_dict)

        self.assertEqual((0 + 0 + 8 + 8) / 4, error)