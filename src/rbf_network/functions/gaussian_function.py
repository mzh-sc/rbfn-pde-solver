import tensorflow as tf

from abc import ABCMeta
from rbf_network.functions.basis_function import BasisFunction

class Gaussian(BasisFunction, metaclass=ABCMeta):
    name = 'gaussian'

    def __init__(self):
        BasisFunction.__init__(self)

        self.a = None

    def y(self, x):
        return tf.exp(-tf.reduce_sum(tf.pow(x - self.center, 2)) /
                      (2 * self.a * self.a))