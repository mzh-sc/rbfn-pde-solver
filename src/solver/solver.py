import tensorflow as tf
import rbf_network as nn

from solver.model import Model
from solver.problem import Problem


class Solver(object):
    __known_metrics = {
        'error': lambda self, feed_dict: tf.get_default_session().run(self.__loss_function, feed_dict=feed_dict)
    }

    def __init__(self, loss_function):
        self.__loss_function = loss_function
        self.__minimize = None
        self.__metric = None

    def compile(self, optimizer, variables, metrics=['error']):
        with tf.name_scope("solver-compile-minimize"):
            self.__minimize = optimizer.minimize(self.__loss_function, var_list=variables)
        self.__metric = metrics

    def fit(self, feed_dict=None):
        with tf.name_scope("minimize"):
            tf.get_default_session().run(self.__minimize, feed_dict=feed_dict)
        with tf.name_scope("metrics-value"):
            return tuple([Solver.__known_metrics[e](self, feed_dict) if e in Solver.__known_metrics
                          else None
                          for e in self.__metric])

if __name__ == '__main__':
    pass