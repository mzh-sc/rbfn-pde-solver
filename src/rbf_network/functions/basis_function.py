from abc import ABCMeta, abstractmethod


class BasisFunction(metaclass=ABCMeta):
    def __init__(self):
        self.center = None

    @abstractmethod
    def y(self): pass

if __name__ == '__main__':
    f = BasisFunction()