import matplotlib.pyplot as plt
import solver.charts as charts

from equations.poisson_pde.create_model_graph import *
from equations.solution import Solution


DATA_DIR = './data'
MODEL_NAME = 'model'
MODEL_PATH = '{}/{}'.format(DATA_DIR, MODEL_NAME)

if not ckeck_if_model_graph_exists(DATA_DIR, MODEL_NAME):
    create_model_graph(DATA_DIR, MODEL_NAME)

with tf.Session() as s:
    saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
    saver.restore(s, tf.train.latest_checkpoint(DATA_DIR))

    solution = Solution.load()

    feed_dict = {EQUATION_CONSTRAIN: ps.uniform_points_2d(0.1, 0.9, 6, 0.1, 0.9, 6),
                 BC1_CONSTRAIN: ps.uniform_points_2d(0, 1, 10, 0, 0, 1) + \
                                ps.uniform_points_2d(0, 1, 10, 1, 1, 1) + \
                                ps.uniform_points_2d(0, 0, 1, 0, 1, 10) + \
                                ps.uniform_points_2d(1, 1, 1, 0, 1, 10)}

    fig = plt.figure()
    plt.ion()
    plt.draw()

    error_plot = charts.Error(fig, 121)
    nn_surface = charts.Surface(fig, 122,
                                x0=0, x1=1, x_count=25,
                                y0=0, y1=1, y_count=25,
                                function=lambda x: solution.y(x))

    i = 1
    while True:
        error = solution.fit(feed_dict=feed_dict)
        error_plot.add_error(error)

        print(error)

        if i % 20 == 0:
            error_plot.update()

            nn_surface.update()
            plt.draw()
            plt.pause(0.005)

            saver.save(s, MODEL_PATH, global_step=i, write_meta_graph=False)
        i += 1