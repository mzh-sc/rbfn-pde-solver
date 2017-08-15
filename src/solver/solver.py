import tensorflow as tf
import rbf_network as nn

from solver.model import Model
from solver.problem import Problem

class Solver(object):
    __known_metrics = {
        'error': lambda self: tf.get_default_session().run(self.__error, feed_dict=self.__feed_dict)
    }

    def __init__(self, problem, model):
        self.__problem = problem
        self.__model = model
        self.__constrain_control_points = {}
        self.__constrain_control_points_weights = {}

        self.__feed_dict = {}
        self.__error = None
        self.__minimize = None
        self.__metric = None

    @property
    def error(self):
        return self.__error

    @property
    def feed_dict(self):
        return self.__feed_dict

    def set_control_points(self, constrain_name, weight, points):
        if constrain_name not in self.__problem.constrains:
            raise ValueError(constrain_name)

        self.__constrain_control_points_weights[constrain_name] = weight
        self.__constrain_control_points[constrain_name] = points

    def compile(self, optimizer, variables, metrics=['error']):
        self.__feed_dict = {}

        control_points_error = tf.constant([], dtype=nn.type, shape=(0,))
        for constrain_name in self.__constrain_control_points.keys():
            constrain = self.__problem.constrains[constrain_name]
            alpha = self.__constrain_control_points_weights[constrain_name]
            train_points = self.__constrain_control_points[constrain_name]

            xs = tf.placeholder(dtype=nn.type, shape=(len(train_points), len(train_points[0])))
            self.__feed_dict[xs] = train_points

            #todo: [opt] optimize using bulk computation (see: map_fn in TestPerformance and equation)
            for index in range(len(train_points)):
                x = xs[index]
                expected = constrain.right(x)
                real = constrain.left(self.__model.network.y, x)
                control_points_error = tf.concat([control_points_error, tf.expand_dims(alpha * tf.square(expected - real), 0)], 0)

        self.__error = control_points_error
        self.__minimize = optimizer.minimize(self.__error, var_list=variables)
        self.__metric = metrics

    def fit(self):
        tf.get_default_session().run(self.__minimize, feed_dict=self.__feed_dict)
        return tuple([Solver.__known_metrics[e](self) if e in Solver.__known_metrics
                              else None
                      for e in self.__metric])

if __name__ == '__main__':
    model = Model()
    model.add_rbf(1, rbf_name='gaussian', center=[1, 0], parameters=[1])
    model.add_rbf(1, rbf_name='gaussian', center=[1, 0], parameters=[1])

    problem = Problem()

    problem.add_constrain('equation',
                          lambda y, x: y(x),
                          lambda x: 1)
    problem.add_constrain('bc1',
                          lambda y, x: y(x),
                          lambda x: 0)

    with tf.Session() as s:
        solver = Solver(model=model, problem=problem)
        solver.set_control_points('equation', 1, [[1, 0], [2, 0]])
        solver.set_control_points('bc1', 100, [[1, 0], [2, 0]])
        solver.compile()
        s.run(solver.error, feed_dict=solver.feed_dict)