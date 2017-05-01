import tensorflow as tf
import numpy as np

from rbf_network.consts import type
from rbf_network.functions.gaussian_function import GaussianFunction


class Network(object):
    def __init__(self, size):
        self._funcs = [GaussianFunction() for _ in range(size)]
        self.weights = None

    def y(self, x):
        return tf.reduce_sum(self.weights * [e.y(x)[0] for e in self])

    def __iter__(self):
        return iter(self._funcs)

if __name__ == '__main__':
    with tf.Session() as s:
        nn = Network(1, 1)
        nn.weights = [1]
        for f in nn:
            f.center = [3.2]
            f.parameters = [4.5]

        print(s.run(f.y(), {f.x: [2.3]}))
