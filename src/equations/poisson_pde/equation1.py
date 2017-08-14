from rbf_network.model import Model

import numpy as np
import tensorflow as tf
import math

from rbf_network.problem import Problem
from rbf_network.solver import Solver
from rbf_network.utils import *

# model
rbfs_count = 5

model = Model()
for (w, c, a) in zip(np.ones(rbfs_count),
                      random_points_2d(-0.1, 1.1, -0.1, 1.1, rbfs_count),
                      np.ones(rbfs_count)):
    model.add_rbf(w, 'gaussian', c, a=a)


# problem
problem = Problem()

def equation(y, x):
    h = tf.hessians(y(x), x)[0]
    return h[0][0] + h[1][1]

problem.add_constrain('equation',
                      equation,
                      lambda x: math.sin(math.pi * x[0]) *
                                math.sin(math.pi * x[1]))
problem.add_constrain('bc1',
                      lambda y, x: y(x),
                      lambda x: 0)


# solver
solver = Solver(problem, model)
solver.set_control_points('equation', 1,
                          uniform_points_2d(0.1, 0.9, 10, 0.1, 0.9, 10))
solver.set_control_points('bc1', 100,
                          uniform_points_2d(0, 1, 10, 0, 0, 1) +
                          uniform_points_2d(0, 1, 10, 1, 1, 1) +
                          uniform_points_2d(0, 0, 1, 0, 1, 10) +
                          uniform_points_2d(1, 1, 1, 0, 1, 10))

solver.compile()
while True:
    solver.train()
