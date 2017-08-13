from collections import namedtuple

from rbf_network import Gaussian
from rbf_network.network import Network


class Model(object):
    #region class members

    __KnownRbf = namedtuple('KnownRbf', ['create', 'get_params'])

    __known_rbfs = {}

    def __new__(cls, *args, **kwargs):
        cls_obj = super(Model, cls).__new__(cls)

        def create_gaussian(center, params):
            f = Gaussian()
            f.center = center
            f.a = params['a']
            return f

        Model._add_known_rbf_type(Gaussian.name,
                                  create_gaussian,
                                  lambda f: {'a': f.a})
        return cls_obj

    def _add_known_rbf_type(rbf_name,
                            create_func,
                            get_params_func):
        Model.__known_rbfs[rbf_name] = \
            Model.__KnownRbf(create=create_func, get_params=get_params_func)

    #endregion

    @property
    def network(self):
        pass

    @property
    def weights(self):
        pass

    @property
    def centers(self):
        pass

    @property
    def parameters(self):
        pass

    def __init__(self):
        self.rbfs = []
        self.weigths = []

    def add_rbf(self, weigth, rbf_name, center, **params):
        if rbf_name not in Model.__known_rbfs:
            raise ValueError('Unknown RBF type - {}'.format(rbf_name))
        self.rbfs.append(Model.__known_rbfs[rbf_name].create(center, params))
        self.weigths.append(weigth)

    def compile(self):
        nn = Network()


if __name__ == '__main__':
    model = Model()
    model.add_rbf(0.5, rbf_name='gaussian', center=[2.3, 1.2], a=2.1)
    model.add_rbf(0.1, rbf_name='gaussian', center=[2.1, 0.2], a=1.1)