from unittest import TestCase

import solver as slv
import tensorflow as tf
import numpy as np
import rbf_network as rbf

from solver import uniform_points_1d, uniform_points_2d


class TestOptimizer(TestCase):
    def test_try_solver_1d(self):
        """
        Model: 1 * rbf((0), 1) # weight * rbf(center, params)
        Approximated function: rbf((2), 1)

        Adjusting parameters - centers
        The idea is to see if solver can move model's center to the center of approximated function
        :return:
        """
        model = slv.Model()
        model.add_rbf(1, rbf_name='gaussian', center=[0], parameters=[1])
        model.compile()

        problem = slv.Problem()
        problem.add_constrain('equation',
                              lambda y, x: y(x),
                              lambda x: tf.exp(-tf.pow(x - 2, 2) / 2),
                              1)
        problem.compile()

        loss = slv.Loss(problem=problem, model=model)
        loss.set_constrain_weight('equation', 1)
        loss.compile()

        optimizer = slv.Optimizer(optimizer_name=slv.Optimizer.GRADIENT_DESCENT, learning_rate=0.2)
        optimizer.set_loss(loss.value)
        optimizer.add_optimized_variables(model.centers)
        optimizer.compile()

        feed_dict = {next(iter(loss.constrain_placeholders_dict.values())):
                         np.array(uniform_points_1d(-1.0, 3.0, 15), dtype=rbf.type.as_numpy_dtype())}
        with tf.Session() as s:
            s.run(tf.global_variables_initializer())

            print(s.run(loss.value, feed_dict=feed_dict))

            for i in range(100):
                s.run(optimizer.minimize, feed_dict=feed_dict)
                print(s.run(loss.value, feed_dict=feed_dict))

            print(model.weights.eval())
            print(model.centers.eval())
            print(model.parameters.eval())

    def test_try_solver_2d(self):
        """
        Model: 1 * rbf((0, 0), 1) # weight * rbf(center, params)
        Approximated function: rbf((2, 2), 1)

        Adjusting parameters - centers
        The idea is to see if solver can move model's center to the center of approximated function
        :return:
        """
        model = slv.Model()
        model.add_rbf(1, rbf_name='gaussian', center=[0, 0], parameters=[1])
        model.compile()

        problem = slv.Problem()
        problem.add_constrain('equation',
                              lambda y, x: y(x),
                              lambda x: tf.exp(-(tf.pow(x - 2, 2) + tf.pow(x - 2, 2)) / 2),
                              2)
        problem.compile()

        loss = slv.Loss(model=model, problem=problem)
        loss.set_constrain_weight('equation', 1)
        loss.compile()

        optimizer = slv.Optimizer(optimizer_name=slv.Optimizer.GRADIENT_DESCENT, learning_rate=0.2)
        optimizer.set_loss(loss.value)
        optimizer.add_optimized_variables(model.centers)
        optimizer.compile()

        feed_dict = {next(iter(loss.constrain_placeholders_dict.values())):
                         np.array(uniform_points_2d(-1, 3, 25, -1, 3, 25), dtype=rbf.type.as_numpy_dtype())}
        with tf.Session() as s:
            s.run(tf.global_variables_initializer())

            print(s.run(loss.value, feed_dict=feed_dict))

            for i in range(100):
                s.run(optimizer.minimize, feed_dict=feed_dict)
                print(s.run(loss.value, feed_dict=feed_dict))

            print(model.weights.eval())
            print(model.centers.eval())
            print(model.parameters.eval())
