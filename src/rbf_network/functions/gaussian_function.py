import tensorflow as tf

from abc import ABCMeta
from rbf_network.functions.basis_function import BasisFunction

class Gaussian(BasisFunction, metaclass=ABCMeta):
    name = 'gaussian'

    def __init__(self):
        BasisFunction.__init__(self)

    def y(self, x):
        if x.shape != (self.dimention,):
            raise Exception("Unexpected shape {}. The current implementation can handle "
                            "(dim,) shape only".format(x.shape))

        #note: it is important to do squeezing for getting f1, otherwise we get [f1]
        #then tf.reduce_sum(self.weights * tf.stack([e.y(x) for e in self])) lead to
        #[w1, w2] * [[f1], [f2]] w1*f1 + w1*f2 +w2*f1 + w2*f2 iso w1*f1 + w2*f2
        return tf.exp(-tf.reduce_sum(tf.pow(x - self.center, 2)) /
                      (2 * tf.squeeze(self.parameters) * tf.squeeze(self.parameters)))

if __name__ == '__main__':
    pass