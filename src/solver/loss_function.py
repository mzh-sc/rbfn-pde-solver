import tensorflow as tf
import rbf_network as nn


class LossFunction(object):
    def __init__(self, problem, model):
        self.__problem = problem
        self.__model = model
        self.__constrain_weights = {}

        self.__feed_placeholders_dict = {}
        self.__error = None

    @property
    def error(self):
        """

        :return: the error tensor
        """
        return self.__error

    @property
    def feed_placeholders_dict(self):
        """

        :return: aggregated over all control points feed dictionary
        """
        return self.__feed_placeholders_dict

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
        self.__feed_placeholders_dict = {}

        control_points_errors = []
        for constrain_name in self.__constrain_weights.keys():
            with tf.name_scope("solver-compile-constrain-{}".format(constrain_name)):
                constrain = self.__problem.constrains[constrain_name]
                alpha = self.__constrain_weights[constrain_name]

                # todo: use placeholder in the future.
                # xs = tf.constant(train_points, dtype=nn.type, shape=(len(train_points), len(train_points[0])))

                # initial implementation
                # xs = tf.placeholder( dtype=nn.type, shape=(len(train_points), len(train_points[0])))
                # self.__feed_dict[xs] = train_points

                xs = tf.placeholder(dtype=nn.type, shape=(None, constrain.x_dim))
                self.__feed_placeholders_dict[constrain_name] = xs

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