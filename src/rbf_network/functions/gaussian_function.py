import tensorflow as tf
import time

from abc import ABCMeta
from rbf_network.functions.basis_function import BasisFunction
from rbf_network.consts import type

from rbf_network.consts import type

class GaussianFunction(BasisFunction, metaclass=ABCMeta):
    def __init__(self):
        BasisFunction.__init__(self)

        self.a = None

    def y(self, x):
        return tf.exp(-tf.reduce_sum(tf.pow(x - self.center, 2)) /
                      (2 * self.a * self.a))


def track(func):
    t_org = time.perf_counter()
    res = func()
    print('time: {:f}'.format(time.perf_counter() - t_org))
    return res


if __name__ == '__main__':
    # tf.get_default_graph().finalize()
    with tf.Session() as s:
        f = GaussianFunction(tf.placeholder(type, [1]))
        # s.run(tf.global_variables_initializer())

        f.center = [3.2]
        f.parameters = [4.5]
        print(track(lambda: s.run(f.y(), {f.x: [2.3]})))
        print(track(lambda: s.run(f.y(), {f.x: [2.3]})))

        print('Gradient')
        grads = tf.gradients(f.y(), [f.center, f.parameters])
        print(track(lambda: s.run(grads, {f.x: [2.3]})))
        f.parameters = [4.1]
        print(track(lambda: s.run(grads, {f.x: [2.1]})))
