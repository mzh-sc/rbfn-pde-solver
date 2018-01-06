from abc import ABCMeta, abstractmethod


class BasisFunction(metaclass=ABCMeta):

    def __init__(self):
        self.center = None
        self.parameters = None

    @property
    def dimention(self):
        return self.center.shape[0] if self.center is not None else 0


    @abstractmethod
    def y(self): pass

if __name__ == '__main__':
    pass
