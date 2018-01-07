from datetime import datetime

import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

import solver as ps
import numpy as np
import solver.charts as charts

logdir = "C:/tf_logs/run-{}/".format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))

# model
rbfs_count = 5


# model creation
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

# todo: [opt] don't use tf. Precalculate them?
# the problem equations
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
                          ps.uniform_points_2d(0.1, 0.9, 6, 0.1, 0.9, 6))
solver.set_control_points('bc1', 100,
                          ps.uniform_points_2d(0, 1, 10, 0, 0, 1) +
                          ps.uniform_points_2d(0, 1, 10, 1, 1, 1) +
                          ps.uniform_points_2d(0, 0, 1, 0, 1, 10) +
                          ps.uniform_points_2d(1, 1, 1, 0, 1, 10))

solver.compile(optimizer=tf.train.GradientDescentOptimizer(0.1),
               variables=[model.weights, model.centers, model.parameters],
               metrics=['error'])

fig = plt.figure()
plt.ion()
plt.draw()

error_plot = charts.Error(fig, 121)
nn_surface = charts.Surface(fig, 122,
                            x0=0, x1=1, x_count=25,
                            y0=0, y1=1, y_count=25,
                            function=model.network.y)

# file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# file_writer.close()

with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    i = 0
    while True:
        error = solver.fit()
        error_plot.add_error(error)

        print(error)

        if i % 20 == 0:
            error_plot.update()

            nn_surface.update()
            plt.draw()
            plt.pause(0.005)

        i += 1

# ---- external ----
# see examples here https://bitbucket.org/andrewpeterson/amp/pull-requests/5/master/diff
# extOpt=ScipyOptimizerInterface(solver.error,method='l-BFGS-b',options={ 'maxiter': 200, 'ftol': 1.e-10, 'gtol': 1.e-10, 'factr': 1.e4})
# with tf.Session() as s:
#     s.run(tf.global_variables_initializer())
#     extOpt.minimize(s, feed_dict=solver.feed_dict, step_callback=lambda x: print(x))
#     print(s.run(solver.error, feed_dict=solver.feed_dict))
#
#     nn_surface.update()
#     plt.draw()
#     plt.pause(0.005)
# ---- external ----
