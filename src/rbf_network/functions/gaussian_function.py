import tensorflow as tf

from abc import ABCMeta
from rbf_network.functions.basis_function import BasisFunction


class Gaussian(BasisFunction, metaclass=ABCMeta):
    name = 'gaussian'

    def __init__(self):
        BasisFunction.__init__(self)

    def y(self, x, centers, parameters):
        """
        see test_gaussian!
        :param x: (dim,) ex. [0.3, 2.5]
        :param center: (n, dim) ex. [[0.3, 2.5], [0.3, 2.5], [0.3, 2.5]]
        :param parameters: (n, 1) ex. [[1], [2], [3]
        :return: (n) [2, 3.3, 2.1]
        """
        nominators = -tf.reduce_sum(tf.pow(x - centers, 2), axis=-1)
        # tf.pow(x - centers, 2)[x:(dim), y(n, dim)] -> (n, dim)
        # tf.reduce_sum[(n, dim)] -> (n)

        a = tf.squeeze(parameters, axis=-1)  # [(n, 1)] -> (n)
        denominators_a = 2 * a * a  # (n)

        return tf.exp(nominators / denominators_a)  # (n)


if __name__ == '__main__':
    pass
