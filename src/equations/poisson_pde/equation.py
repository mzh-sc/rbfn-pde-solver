from datetime import datetime

from tensorflow.python.saved_model import tag_constants

from equations.poisson_pde.create_model1 import MODEL_NAME, DATA_DIR, TransientModel

import matplotlib.pyplot as plt
import tensorflow as tf
import solver as ps
import solver.charts as charts

logdir = "C:/tf_logs/run-{}/".format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))

with tf.Session() as s:
    # applicable when using saver without import_meta_graph
    # weights = tf.Variable(0.0, dtype=nn.type, validate_shape=False, name=ps.Model.WEIGHTS)
    # centers = tf.Variable(0.0, dtype=nn.type, validate_shape=False, name=ps.Model.CENTERS)
    # parameters = tf.Variable(0.0, dtype=nn.type, validate_shape=False, name=ps.Model.PARAMETERS)

    saver = tf.train.import_meta_graph('{}/{}.meta'.format(DATA_DIR, MODEL_NAME))
    saver.restore(s, tf.train.latest_checkpoint(DATA_DIR))

    tm = TransientModel.restore()

    loss = tm.op_loss
    model_y = tm.op_model_y

    # solver
    solver = ps.Solver(loss_function=loss)
    solver.compile(optimizer=tf.train.GradientDescentOptimizer(0.1),
                   variables=[tm.vr_weights, tm.vr_centers, tm.vr_parameters],
                   metrics=['error'])

    feed_dict = {}
    feed_dict[tm.pls_control_points[0]] = ps.uniform_points_2d(0.1, 0.9, 6, 0.1, 0.9, 6)
    feed_dict[tm.pls_control_points[1]] = ps.uniform_points_2d(0, 1, 10, 0, 0, 1) + \
                                ps.uniform_points_2d(0, 1, 10, 1, 1, 1) + \
                                ps.uniform_points_2d(0, 0, 1, 0, 1, 10) + \
                                ps.uniform_points_2d(1, 1, 1, 0, 1, 10)

    fig = plt.figure()
    plt.ion()
    plt.draw()

    error_plot = charts.Error(fig, 121)
    nn_surface = charts.Surface(fig, 122,
                                x0=0, x1=1, x_count=25,
                                y0=0, y1=1, y_count=25,
                                function=lambda x: s.run(model_y, feed_dict={tm.pl_x_of_y: x}))

    # file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    # file_writer.close()

    #s.run(tf.global_variables_initializer())

    i = 0
    while True:
        error = solver.fit(feed_dict=feed_dict)
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
