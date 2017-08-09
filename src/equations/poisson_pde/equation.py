import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import rbf_network as rbfn
import math as math

from rbf_network.consts import type
from rbf_network.network import Network

dim = 2

a_x = 0
b_x = 1

a_y = 0
b_y = 1

#train points
train_points = np.array([[x, y]
                         for x in np.linspace(a_x, b_x, 10)
                            for y in np.linspace(a_y, b_y, 10)], dtype=np.float32)

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
    weights = tf.Variable(neuron_weights, dtype=type)
    centers = tf.Variable(neuron_centers, dtype=type)
    parameters = tf.Variable(neuron_parameters, dtype=type)

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

    # instead of reduce_sum use reduce_mean here. See
    # https://stackoverflow.com/questions/43145847/tensorflow-loss-minimization-is-increasing-loss
    error = tf.reduce_mean(tf.square(ys_exp -
        tf.map_fn(lambda x: tf.reduce_sum(tf.hessians(nn.y(x), x)), xs)))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(error, var_list=[weights])

    #todo: tests
    #todo: matplot
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    fig.show()
    plt.draw()

    yerror = []

    s.run(tf.global_variables_initializer())
    # training loop
    for i in range(1000):
        s.run(train, {xs: train_points})

        #if i % 20 == 0:
        yerror.append(s.run(error, {xs: train_points}))
        ax.plot(range(len(yerror)), yerror)
        fig.show()
        plt.draw()

    # evaluate training accuracy
    curr_error = s.run(error, {xs: train_points})
    print("Error: {}".format(curr_error))

    fig.show()
    plt.waitforbuttonpress()