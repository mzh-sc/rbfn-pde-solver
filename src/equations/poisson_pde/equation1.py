import math
import matplotlib.pyplot as plt
import tensorflow as tf
import solver as ps
import numpy as np
import solver.charts as charts

# model
rbfs_count = 5

with tf.Session() as s:
    model = ps.Model()
    for (w, c, a) in zip(np.ones(rbfs_count),
                         ps.random_points_2d(-0.1, 1.1, -0.1, 1.1, rbfs_count),
                         np.ones(rbfs_count)):
        model.add_rbf(w, 'gaussian', c, parameters=[a])
    model.compile()

    # problem
    problem = ps.Problem()

    def equation(y, x):
        h = tf.hessians(y(x), x)[0]
        return h[0][0] + h[1][1]


    # todo: [opt] don't use tf
    problem.add_constrain('equation',
                          equation,
                          lambda x: tf.sin(math.pi * x[0]) * tf.sin(math.pi * x[1]))
    problem.add_constrain('bc1',
                          lambda y, x: y(x),
                          lambda x: 0)
    problem.compile()

    # solver
    solver = ps.Solver(problem, model)
    solver.set_control_points('equation', 1,
                              ps.uniform_points_2d(0.1, 0.9, 2, 0.1, 0.9, 2))
    solver.set_control_points('bc1', 100,
                              ps.uniform_points_2d(0, 1, 2, 0, 0, 1) +
                              ps.uniform_points_2d(0, 1, 2, 1, 1, 1) +
                              ps.uniform_points_2d(0, 0, 1, 0, 1, 2) +
                              ps.uniform_points_2d(1, 1, 1, 0, 1, 2))


    solver.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
                   variables=[model.weights, model.centers, model.parameters],
                   metrics=['error'])


    fig = plt.figure()
    plt.ion()

    error_plot = charts.Error(fig, 121)
    nn_surface = charts.Surface(fig, 122,
                                x0=0, x1=1, x_count=10,
                                y0=0, y1=1, y_count=10,
                                function=model.network.y)
    while True:
        error = solver.fit()

        error_plot.add_error(error)
        error_plot.update()

        nn_surface.update()
        plt.draw()



