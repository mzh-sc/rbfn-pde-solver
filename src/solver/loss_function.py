import tensorflow as tf
import rbf_network as nn


class LossFunction(object):
    def __init__(self, problem, model):
        self.__problem = problem
        self.__model = model
        self.__constrain_control_points = {}
        self.__constrain_control_points_weights = {}

        self.__feed_dict = {}
        self.__error = None

    @property
    def error(self):
        """

        :return: the error tensor
        """
        return self.__error

    @property
    def feed_dict(self):
        """

        :return: aggregated over all control points feed dictionary
        """
        return self.__feed_dict

    def set_control_points(self, constrain_name, weight, points):
        """

        :param constrain_name:
        :param weight:
        :param points: the list of control points [[dim], [dim],...]. For ex.: [[1.0], [2.0]...], [[1.0, 1.0], [2.0, 1.0]...]
        :return:
        """
        if constrain_name not in self.__problem.constrains:
            raise ValueError(constrain_name)

        self.__constrain_control_points_weights[constrain_name] = weight
        self.__constrain_control_points[constrain_name] = points

    def compile(self):
        self.__feed_dict = {}

        control_points_errors = []
        for constrain_name in self.__constrain_control_points.keys():
            with tf.name_scope("solver-compile-constrain-{}".format(constrain_name)):
                constrain = self.__problem.constrains[constrain_name]
                alpha = self.__constrain_control_points_weights[constrain_name]
                train_points = self.__constrain_control_points[constrain_name]

                # todo: use placeholder in the future.
                xs = tf.constant(train_points, dtype=nn.type, shape=(len(train_points), len(train_points[0])))

                #xs = tf.placeholder( dtype=nn.type, shape=(len(train_points), len(train_points[0])))
                #self.__feed_dict[xs] = train_points

                # optimized using bulk computation (see: map_fn in TestTensorflow.test_try_hessian and equation)
                # for index in range(len(train_points)):
                #     x = xs[index]
                #     expected = constrain.right(x)
                #     real = constrain.left(self.__model.network.y, x)
                #     control_points_errors.append(tf.expand_dims(alpha * tf.square(expected - real), 0))

                # output: tensor (n) = [error1, error2...]
                def constrain_func(x):
                    with tf.name_scope("{}-constrain-left".format(constrain_name)):
                        left = constrain.left(self.__model.network.y, x)
                    with tf.name_scope("{}-constrain-right".format(constrain_name)):
                        right = constrain.right(x)
                    return left - right

                control_points_errors.append(alpha * tf.square(
                    tf.map_fn(constrain_func, xs)))

        with tf.name_scope("solver-compile-error-functional"):
            self.__error = tf.reduce_mean(tf.concat(control_points_errors, axis=0))

if __name__ == '__main__':
    pass