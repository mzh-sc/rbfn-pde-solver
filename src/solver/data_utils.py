import numpy as np


def random_points_1d(from1, to1, count1):
    """

    :param from1:
    :param to1:
    :param count1:
    :return: the list of points [[x1], [x2],..]
    """
    return ((to1 - from1) * np.random.rand(count1, 1) + from1).tolist()


def random_points_2d(from1, to1,
                     from2, to2,
                     count):
    """

    :param from1:
    :param to1:
    :param from2:
    :param to2:
    :param count:
    :return: the list of 2d points [[x1, y1], [x2, y2],..]
    """
    return (np.matmul(np.random.rand(count, 2), np.diag([to1 - from1, to2 - from2])) + [from1, from2]).tolist()


def uniform_points_1d(from1, to1, count1):
    """

    :param from1:
    :param to1:
    :param count1:
    :return: the list of points [[x1], [x2],..]
    """
    return np.expand_dims(np.linspace(from1, to1, count1), axis=1).tolist()


def uniform_points_2d(from1, to1, count1,
                      from2, to2, count2):
    """

    :param from1:
    :param to1:
    :param count1:
    :param from2:
    :param to2:
    :param count2:
    :return: the list of 2d points [[x1, y1], [x2, y2],..]
    """
    return [[x, y] for y in np.linspace(from2, to2, count2)
            for x in np.linspace(from1, to1, count1)]


if __name__ == '__main__':
    pass
