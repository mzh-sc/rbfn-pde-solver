import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import rbf_network as rbfn
import math
import time

from rbf_network.consts import type
from rbf_network.network import Network

dim = 2

a_x = 0
b_x = 1

a_y = 0
b_y = 1

X = np.linspace(a_x, b_x, 10, dtype=np.float32)
Y = np.linspace(a_y, b_y, 10, dtype=np.float32)
X, Y = np.meshgrid(X, Y)

#train points
train_points = np.stack((np.reshape(X, (-1)), np.reshape(Y, (-1))), axis=-1)

   # np.stack(
   # np.reshape(
   #     np.meshgrid(
   #          np.linspace(a_x, b_x, n_x),
   #          np.linspace(a_y, b_y, n_y)), (2, -1)), axis=-1) #if ``axis=-1`` it will be the last dimension

#neurons
n_neurons = 5
neuron_centers = 1.1 * np.random.rand(n_neurons, dim)
neuron_parameters = np.ones(n_neurons)
neuron_weights = np.ones(n_neurons)


with tf.Session() as s:
    t_org = time.perf_counter()

    weights = tf.Variable(neuron_weights, dtype=type)
    centers = tf.Variable(neuron_centers, dtype=type)
    parameters = tf.Variable(neuron_parameters, dtype=type)

    s.run(tf.global_variables_initializer())

    rbfs = []
    for i in range(n_neurons):
        rbf = rbfn.Gaussian()
        rbf.center = centers[i]
        rbf.a = parameters[i]
        rbfs.append(rbf)

    nn = Network(rbfs)
    nn.weights = weights


    xs = tf.placeholder(dtype=type, shape=(train_points.shape[0], dim))

    ys_exp = tf.constant([math.sin(math.pi * e[0]) * math.sin(math.pi * e[1])
                for e in train_points], dtype=type)

    def equation(y, x):
        h = tf.hessians(y(x), x)[0]
        return h[0][0] + h[1][1]

    # instead of reduce_sum use reduce_mean here. See
    # https://stackoverflow.com/questions/43145847/tensorflow-loss-minimization-is-increasing-loss
    error = tf.reduce_mean(tf.square(ys_exp -
        tf.map_fn(lambda x: equation(nn.y, x), xs)))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(error, var_list=[weights])

    #todo: tests

    fig = plt.figure()
    plt.ion()

    error_axes = fig.add_subplot(1, 2, 1)
    #error_axes.set_yscale('log')
    error_axes.grid(True)
    error_axes.set_ylabel('Error')

    surface_axes = fig.add_subplot(1, 2, 2, projection='3d')
    surface_axes.grid(True)
    surface_axes.set_ylabel('Solution')
    plt.draw()

    t_end = time.perf_counter()
    print('Duration of {}'.format(t_end - t_org))

    yerror = []
    # training loop
    for i in range(1000):
        t_org = time.perf_counter()

        s.run(train, {xs: train_points})

        curr_error = s.run(error, {xs: train_points})
        print("Error: {}".format(curr_error))

        #if i % 20 == 0:
        yerror.append(curr_error)
        error_axes.plot(range(len(yerror)), yerror, 'b-')

        surface_axes.cla()
        surface_axes.plot_surface(X, Y, s.run(
            tf.reshape(
                tf.map_fn(lambda x: nn.y(x),
                          tf.stack((tf.reshape(X.astype(np.float32), [-1]),
                                    tf.reshape(Y.astype(np.float32), [-1])), axis=-1), dtype=type), shape=[10, 10])),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False)

        plt.draw()

        t_end = time.perf_counter()
        print('Duration of {}'.format(t_end - t_org))

        plt.pause(0.005)

    # evaluate training accuracy
    curr_error = s.run(error, {xs: train_points})
    print("Error: {}".format(curr_error))

    fig.show()
    plt.waitforbuttonpress()