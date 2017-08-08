import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from rbf_network.consts import type
from rbf_network.network import Network

dim = 2
n_neurons = 5

#control points
n_x = 10
n_y = 10

a_x = 0
b_x = 1

a_y = 0
b_y = 1

X, Y = np.meshgrid(np.linspace(a_x, b_x, n_x),
    np.linspace(a_y, b_y, n_y))


weights = tf.Variable(tf.ones([nn_size], type))
centers = tf.Variable(tf.linspace(0.0, 1.0, nn_size), type)
parameters = tf.Variable(tf.ones([nn_size], type))

nn = Network(nn_size)
nn.weights = weights
for i, f in enumerate(nn):
    f.center = tf.slice(centers, [i * dim], [dim])
    f.parameters = tf.slice(parameters, [i], [dim])

x = tf.placeholder(type, [cp_count, dim], name='x')
y0 = tf.placeholder(type, [cp_count, 1], name='y0')

error = tf.reduce_sum(
    tf.map_fn(lambda i: tf.square(nn.y(x[i]) - y0[i]), tf.range(cp_count), dtype=type))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(error, var_list=weights)

# training data
# from 0 to 1
points = [i / (cp_count - 1) for i in range(cp_count)]
x_train = [[e] for e in points]
y_train = [[e] for e in points]

with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    plt.ion()
    fig, ax = plt.subplots(1, 1)
    fig.show()
    plt.draw()

    yerror = []
    # training loop
    for i in range(1000):
        s.run(train, {x: x_train, y0: y_train})

        if i % 20 == 0:
            yerror.append(s.run(error, {x: x_train, y0: y_train}))
            ax.plot(range(len(yerror)), yerror)
            fig.show()
            plt.draw()

    # evaluate training accuracy
    curr_error = s.run(error, {x: x_train, y0: y_train})
    print("Error: {}".format(curr_error))

    fig.show()
    plt.waitforbuttonpress()