import tensorflow as tf
from rbf_network.consts import type


class Network(object):
    def __init__(self, rbf):
        self._function = rbf.y
        self.weights = None
        self.centers = None
        self.parameters = None

    def y(self, x):
        x_tensor = None
        if isinstance(x, list):
            x_tensor = tf.concat(x, axis=1)
        else:
            x_tensor = x

        rank = x_tensor.shape.ndims
        if rank > 2:
            raise Exception("Rank should be either 1 or 2. Got - {}".format(rank))
        if rank == 2:
            x_tensor = tf.expand_dims(x_tensor, 1)
        with tf.name_scope("rbfn-value"):
            return tf.expand_dims( # to have the same output as TF math operations: [[1], [3]] -> [[f(1)], [f(3)]] iso [f(1), f(3)]
                tf.reduce_sum(self.weights * self._function(x_tensor, self.centers, self.parameters), axis=-1), axis=-1)
