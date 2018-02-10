import tensorflow as tf
from rbf_network.consts import type


class Network(object):
    def __init__(self, rbf):
        self._function = rbf.y
        self.weights = None
        self.centers = None
        self.parameters = None

    def y(self, x):
        with tf.name_scope("rbfn-value"):
            return tf.reduce_sum(self.weights * self._function(x, self.centers, self.parameters), axis=-1)
