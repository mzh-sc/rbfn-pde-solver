import math
import tensorflow as tf
import solver as ps
import numpy as np
import rbf_network as nn

from equations.transient_model import TransientModel

MODEL_NAME = 'model1'
DATA_DIR = './data'


def create_model1():
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

    # loss
    loss = ps.LossFunction(problem, model)
    loss.set_control_points('equation', 1,
                            ps.uniform_points_2d(0.1, 0.9, 6, 0.1, 0.9, 6))
    loss.set_control_points('bc1', 100,
                            ps.uniform_points_2d(0, 1, 10, 0, 0, 1) +
                            ps.uniform_points_2d(0, 1, 10, 1, 1, 1) +
                            ps.uniform_points_2d(0, 0, 1, 0, 1, 10) +
                            ps.uniform_points_2d(1, 1, 1, 0, 1, 10))
    loss.compile()

    saver = tf.train.Saver()
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        x = tf.placeholder(dtype=nn.type, shape=(2,))
        y = tf.identity(model.network.y(x))
        TransientModel.save(TransientModel(
            vr_weights=model.weights,
            vr_parameters=model.parameters,
            vr_centers=model.centers,
            op_loss=loss.error,
            op_model_y=y,
            pl_x_of_y=x))

        saver.save(s, DATA_DIR + '/' + MODEL_NAME)


if __name__ == '__main__':
    create_model1()
