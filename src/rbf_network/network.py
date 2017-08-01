import tensorflow as tf

class Network(object):
    def __init__(self, rbfs):
        self._functions = rbfs
        self.weights = None

    def y(self, x):
        return tf.reduce_sum(self.weights * [e.y(x) for e in self])

    def __iter__(self):
        return iter(self._functions)