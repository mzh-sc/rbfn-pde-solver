from rbf_network.model import Model


class Solver(object):
    def __init__(self, problem, model):
        self.__problem = problem
        self.__model = model
        self.__constrain_control_points = {}
        self.__constrain_control_points_weights = {}

    def set_control_points(self, constrain_name, weight, points):
        if constrain_name not in self.__problem:
            raise ValueError(constrain_name)

        self.__constrain_control_points_weights[constrain_name] = weight
        self.__constrain_control_points[constrain_name] = points

if __name__ == '__main__':
    model = Model()
    model.add_rbf(0.5, rbf_name='gaussian', center=[2.3, 1.2], a=2.1)
    model.add_rbf(0.1, rbf_name='gaussian', center=[2.1, 0.2], a=1.1)