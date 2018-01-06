from collections import namedtuple


class Problem(object):
    __Constrain = namedtuple('Constrain', ['left', 'right'])

    def __init__(self):
        self.constrains = {}

    def add_constrain(self, constrain_name, left, right):
        """
        Add the problem equation LeftOp(f(x))= RightOp(x)
        :param constrain_name:
        :param left: LeftOp(f(x), x) where f - the solution approximation
        :param right: RightOp(x)
        """
        self.constrains[constrain_name] = Problem.__Constrain(left=left, right=right)

    def compile(self):
        pass