from collections import namedtuple

import tensorflow as tf
import tf_utils as tf_ext
import solver as slv
import rbf_network as nn


class Solution(object):

    @staticmethod
    def load():
        solution = Solution()

        solution.__weights = tf_ext.get_variable(slv.Model.WEIGHTS)
        solution.__centers = tf_ext.get_variable(slv.Model.CENTERS)
        solution.__parameters = tf_ext.get_variable(slv.Model.PARAMETERS)

        solution.__loss = tf.get_collection('op_loss')[0]
        solution.__minimize = tf.get_collection('op_minimize')[0]
        solution.__constrain_placeholders = {c.name[c.name.index('pl_')+3:-2]: c for c in tf.get_collection('pl_constrain')}

        solution.__y = tf.get_collection('op_y')[0]
        solution.__x = tf.get_collection('pl_x')[0]

        return solution

    @staticmethod
    def save(solution):
        tf.add_to_collection('op_loss', solution.__loss)
        tf.add_to_collection('op_minimize', solution.__minimize)
        for pl in solution.__constrain_placeholders.values():
            tf.add_to_collection('pl_constrain', pl)

        tf.add_to_collection('op_y', solution.__y)
        tf.add_to_collection('pl_x', solution.__x)

    @staticmethod
    def create(model, loss, optimizer):
        solution = Solution()

        solution.__weights = model.weights
        solution.__centers = model.centers
        solution.__parameters = model.parameters

        solution.__loss = loss.value
        solution.__minimize = optimizer.minimize
        solution.__constrain_placeholders = list(loss.constrain_placeholders_dict)

        solution.__x = tf.placeholder(dtype=nn.type, shape=(2,))
        solution.__y = tf.identity(model.network.y(solution.__x))

    def __init__(self):
        self.__weights = None
        self.__centers = None
        self.__parameters = None

        self.__loss = None
        self.__minimize = None
        self.__constrain_placeholders = None

        self.__x = None
        self.__y = None

    def y(self, x):
        return tf.get_default_session().run(self.__y, feed_dict={self.__x: x})

    def fit(self, feed_dict):
        self.minimize(feed_dict)
        return self.error(feed_dict)

    def minimize(self, feed_dict):
        tf.get_default_session().run(self.__minimize, feed_dict=self.__constrain_feed_dict(feed_dict))

    def error(self, feed_dict):
        return tf.get_default_session().run(self.__loss, feed_dict=self.__constrain_feed_dict(feed_dict))

    def __constrain_feed_dict(self, feed_dict):
        return {self.__constrain_placeholders[k]: v for (k, v) in feed_dict.items()}