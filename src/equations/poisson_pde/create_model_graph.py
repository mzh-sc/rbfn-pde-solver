import math
import tensorflow as tf
import solver as ps
import numpy as np

from equations.solution import Solution
from datetime import datetime
from pathlib import Path


_graph_log_dir = "C:/tf_logs/run-{}".format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))

EQUATION_CONSTRAIN = 'equation'
BC1_CONSTRAIN = 'bc1'

def ckeck_if_model_graph_exists(model_dir, model_name):
    return Path(model_dir + '/' + model_name + '.meta').exists()

def create_model_graph(model_dir, model_name, write_graph_log=False):
    Path(model_dir).mkdir(exist_ok=True)

    # model
    rbfs_count = 16

    model = ps.Model()
    for (w, c, a) in zip(np.ones(rbfs_count),
                         ps.random_points_2d(-0.1, 1.1, -0.1, 1.1, rbfs_count),
                         np.ones(rbfs_count)):
        model.add_rbf(w, 'gaussian', c, parameters=[a])
    model.compile()


    # problem
    problem = ps.Problem()

    # tried inject hessian calculation before map_fn
    # def hessian(y, x):
    #     h = tf.hessians(y, x)[0]
    #     return h[0][0] + h[1][1]
    #
    # def equation(y, x):
    #     return y(x, lambda _y: hessian(_y, x))

    # what is expected
    # def equation(y, x):
    #     h = tf.hessians(y(x), x)[0]
    #     return h[0][0] + h[1][1]

    def equation(y, x):
        grad = tf.gradients(y(x), x)[0]
        dx1 = grad[0]
        dx2 = grad[1]
        return tf.gradients(dx1, x)[0][0] + tf.gradients(dx2, x)[0][1]

    # todo: [opt] don't use tf. Precalculate them?
    # the problem equations
    problem.add_constrain(EQUATION_CONSTRAIN,
                          equation,
                          lambda x: tf.sin(math.pi * x[0]) * tf.sin(math.pi * x[1]),
                          2)
    problem.add_constrain(BC1_CONSTRAIN,
                          lambda y, x: y(x),
                          lambda x: 0,
                          2)
    problem.compile()


    # loss
    loss = ps.Loss(problem, model)
    loss.set_constrain_weight(EQUATION_CONSTRAIN, 1)
    loss.set_constrain_weight(BC1_CONSTRAIN, 100)
    loss.compile()


    # optimizer
    optimizer = ps.Optimizer(ps.Optimizer.GRADIENT_DESCENT, learning_rate=0.1)
    optimizer.set_loss(loss.value)
    optimizer.add_optimized_variables(model.weights)
    optimizer.add_optimized_variables(model.centers)
    optimizer.add_optimized_variables(model.parameters)
    optimizer.compile()

    saver = tf.train.Saver()
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        Solution.save(Solution.create(model, loss, optimizer))
        saver.save(s, model_dir + '/' + model_name)

        if write_graph_log:
            file_writer = tf.summary.FileWriter(_graph_log_dir + '-' + model_name, tf.get_default_graph())
            file_writer.close()

if __name__ == '__main__':
    create_model_graph('./data', 'model', True)
