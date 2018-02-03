from abc import ABCMeta, abstractmethod


class BasisFunction(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def y(self, x, center, parameters):
        pass

if __name__ == '__main__':
    pass
