import numpy as np
from collections import Iterable


def random_points_1d(from1, to1, count1):
    return (to1 - from1) * np.random.rand(count1, 1)


def random_points_2d(from1, to1,
                     from2, to2,
                     count):
    return np.matmul(np.random.rand(count, 2), np.diag([to1 - from1, to2 - from2])).tolist()


def uniform_points_1d(from1, to1, count1):
    return np.linspace(from1, to1, count1)


def uniform_points_2d(from1, to1, count1,
                      from2, to2, count2):
    return [[x, y] for y in np.linspace(from2, to2, count2)
            for x in np.linspace(from1, to1, count1)]


if __name__ == '__main__':
    pass
