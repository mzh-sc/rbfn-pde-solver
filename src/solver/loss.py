import tensorflow as tf
import rbf_network as nn


class Loss(object):
    def __init__(self, problem, model):
        self.__problem = problem
        self.__model = model
        self.__constrain_weights = {}

        self.__constrain_placeholders_dict = {}
        self.__error = None

    @property
    def value(self):
        """

        :return: the loss tensor
        """
        return self.__error

    @property
    def constrain_placeholders_dict(self):
        """

        :return: constrain placeholders to use them later for feeding
        """
        return self.__constrain_placeholders_dict

    def set_constrain_weight(self, constrain_name, weight):
        """

        :param constrain_name:
        :param weight:
        :return:
        """
        if constrain_name not in self.__problem.constrains:
            raise ValueError(constrain_name)

        self.__constrain_weights[constrain_name] = weight

    def compile(self):
        self.__constrain_placeholders_dict = {}

        control_points_errors = []
        for constrain_name in self.__constrain_weights.keys():
            with tf.name_scope("solver-compile-constrain-{}".format(constrain_name)):
                constrain = self.__problem.constrains[constrain_name]
                alpha = self.__constrain_weights[constrain_name]

                xs = tf.placeholder(name='pl_' + constrain_name, dtype=nn.type, shape=(None, constrain.x_dim))
                self.__constrain_placeholders_dict[constrain_name] = xs

                # output: tensor (n) = [error1, error2...]
                def constrain_func(xs):
                    with tf.name_scope("{}-constrain-left".format(constrain_name)):
                        left = constrain.left(self.__model.network.y, xs)
                    with tf.name_scope("{}-constrain-right".format(constrain_name)):
                        right = constrain.right(xs)
                    return left - right

                # xs=[[1, 1], [2, 2], [3, 3]] - xs_split_by_coordinates=list([[1], [2], [3]], [[1], [2], [3]])
                # it is needed to simplify constrains definition
                xs_split_by_coordinates = tf.split(xs, num_or_size_splits=constrain.x_dim, axis=1)
                control_points_errors.append(alpha * tf.square(constrain_func(xs_split_by_coordinates)))

        with tf.name_scope("solver-compile-error-functional"):
            self.__error = tf.reduce_mean(tf.concat(control_points_errors, axis=0))

if __name__ == '__main__':
    pass