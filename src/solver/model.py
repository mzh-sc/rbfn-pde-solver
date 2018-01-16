from collections import namedtuple

import tensorflow as tf
import rbf_network as nn


class Model(object):
    """
    Model is responsible for RBF neural network construction
    """

    # region class members

    __known_rbfs = {}

    def __new__(cls, *args, **kwargs):
        cls_obj = super(Model, cls).__new__(cls)

        Model._add_known_rbf_type(nn.Gaussian.name, lambda: nn.Gaussian())
        return cls_obj

    def _add_known_rbf_type(rbf_name, create_func):
        Model.__known_rbfs[rbf_name] = create_func

    # endregion

    __rbf = namedtuple('Rbf', ['name', 'weight', 'center', 'parameters'])

    @property
    def network(self):
        return self.__nn

    @property
    def weights(self):
        """
        The RBFN weights
        :return: variable having the shape (rbfs_count,)
        """
        return self.__nn_weights

    @property
    def centers(self):
        """
        The RBFN centers
        :return: variable having the shape (rbfs_count, dimention)
        """
        return self.__nn_centers

    @property
    def parameters(self):
        """
        The RBFN parameters
        :return: variable having the shape (rbfs_count, parameters count)
        """
        return self.__nn_parameters

    def __init__(self):
        self.__rbfs = []

        self.__nn = None
        self.__nn_weights = None
        self.__nn_centers = None
        self.__nn_parameters = None

    def add_rbf(self, weight, rbf_name, center, parameters):
        if rbf_name not in Model.__known_rbfs:
            raise ValueError('Unknown RBF type - {}'.format(rbf_name))
        self.__rbfs.append(Model.__rbf(rbf_name, weight, center, parameters))

    def compile(self):
        rbfs_count = len(self.__rbfs)
        if rbfs_count == 0:
            raise ValueError("Add at least one RBF to network")

        rbf_0 = self.__rbfs[0]
        center_dim = len(rbf_0.center)
        parameters_count = len(rbf_0.parameters)

        # aggregate all wights, centers and parameters to initialize network variables
        rbfs = []
        ws = []
        cs = []
        ps = []
        for i, (rbf_name, w, c, p) in enumerate(self.__rbfs):
            ws.append(w)
            cs.append(c)
            ps.append(p)

        with tf.name_scope("model-compile-aggregated-variables-creation"):
            # the network's aggregated variables (for continence. See test_variables_aggregation in test_network) creation
            self.__nn_weights = tf.get_variable('weights', initializer=tf.constant(ws, dtype=nn.type, shape=(rbfs_count,)), dtype=nn.type)
            self.__nn_centers = tf.get_variable('centers', initializer=tf.constant(cs, dtype=nn.type, shape=(rbfs_count, center_dim)), dtype=nn.type)
            self.__nn_parameters = tf.get_variable('parameters', initializer=tf.constant(ps, dtype=nn.type, shape=(rbfs_count, parameters_count)), dtype=nn.type)

        with tf.name_scope("model-compile-rbfs-creation"):
            # rbfs creation
            for i, (rbf_name, w, c, p) in enumerate(self.__rbfs):
                rbf = self.__known_rbfs[rbf_name]()
                rbf.center = self.__nn_centers[i]
                rbf.parameters = self.__nn_parameters[i]
                rbfs.append(rbf)

        with tf.name_scope("model-compile-network-creation"):
            self.__nn = nn.Network(rbfs)
            self.__nn.weights = self.__nn_weights

        # can not do it here as later have to invoke global initializer that reset these values
        # moreover it is even better to do it via constant initializer as it isn't required to the create session
        # tf.assign(self.__nn_weights, ws).eval()
        # tf.assign(self.__nn_centers, cs).eval()
        # tf.assign(self.__nn_parameters, ps).eval()


if __name__ == '__main__':
    pass