from collections import namedtuple


class Problem(object):
    __Constrain = namedtuple('Constrain', ['left', 'right', 'x_dim'])

    def __init__(self):
        self.constrains = {}

    def add_constrain(self, constrain_name, left, right, x_dim):
        """
        Add the problem equation LeftOp(f(x))= RightOp(x)
        :param constrain_name:
        :param left: LeftOp(f(x), x) where f - the solution approximation
        :param right: RightOp(x)
        :param x_dim:
        """
        self.constrains[constrain_name] = Problem.__Constrain(left=left, right=right, x_dim=x_dim)

    def compile(self):
        pass