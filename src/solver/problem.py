from collections import namedtuple


class Problem(object):
    __Constrain = namedtuple('Constrain', ['left', 'right'])

    def __init__(self):
        self.constrains = {}

    def add_constrain(self, constrain_name, left, right):
        self.constrains[constrain_name] = Problem.__Constrain(left=left, right=right)

    def compile(self):
        pass