import tensorflow as tf
import rbf_network as nn

from solver.model import Model
from solver.problem import Problem


class Optimizer(object):
    GRADIENT_DESCENT = 'gradient_descent'

    __known_optimizers = {}

    def __new__(cls, *args, **kwargs):
        cls_obj = super(Optimizer, cls).__new__(cls)

        Optimizer.__known_optimizers[Optimizer.GRADIENT_DESCENT] = \
            lambda learning_rate: tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return cls_obj

    @property
    def minimize(self):
        return self.__minimize

    def __init__(self, optimizer_name, **optimizer_kwargs):
        if optimizer_name not in Optimizer.__known_optimizers:
            raise ValueError('Unknown Optimizer name - {}'.format(optimizer_name))

        self.__optimizer_name = optimizer_name
        self.__optimizer_kwargs = optimizer_kwargs

        self.__loss = None
        self.__variables = []

        self.__optimizer = None
        self.__minimize = None

    def set_loss(self, loss):
        self.__loss = loss

    def add_optimized_variables(self, variables):
        self.__variables.append(variables)

    def compile(self):
        with tf.name_scope("optimizer-compile-minimize"):
            optimizer = Optimizer.__known_optimizers[self.__optimizer_name](**self.__optimizer_kwargs)
            self.__minimize = optimizer.minimize(self.__loss, var_list=self.__variables)

if __name__ == '__main__':
    pass