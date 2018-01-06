from unittest import TestCase

import solver as slv
import tensorflow as tf
import numpy as np

from solver import uniform_points_1d, uniform_points_2d


class TestSolver(TestCase):
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

        problem = slv.Problem()

        problem.add_constrain('equation',
                              lambda y, x: y(x),
                              lambda x: tf.exp(-tf.pow(x - 2, 2) / 2))

        solver = slv.Solver(model=model, problem=problem)
        solver.set_control_points('equation', 1, uniform_points_1d(-1, 3, 15))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)

        with tf.Session() as s:
            model.compile()
            problem.compile()
            solver.compile(optimizer=optimizer, variables=[model.centers])

            s.run(tf.global_variables_initializer())

            res = s.run(solver.error, feed_dict=solver.feed_dict)
            print(res)

            for i in range(100):
                error = solver.fit()
                print(error)

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

        problem = slv.Problem()

        problem.add_constrain('equation',
                              lambda y, x: y(x),
                              lambda x: tf.exp(-(tf.pow(x - 2, 2) + tf.pow(x - 2, 2)) / 2))

        solver = slv.Solver(model=model, problem=problem)
        solver.set_control_points('equation', 1, uniform_points_2d(-1, 3, 25, -1, 3, 25))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)

        with tf.Session() as s:
            model.compile()
            problem.compile()
            solver.compile(optimizer=optimizer, variables=[model.centers])

            s.run(tf.global_variables_initializer())

            res = s.run(solver.error, feed_dict=solver.feed_dict)
            print(res)

            for i in range(100):
                error = solver.fit()
                print(error)

            print(model.weights.eval())
            print(model.centers.eval())
            print(model.parameters.eval())