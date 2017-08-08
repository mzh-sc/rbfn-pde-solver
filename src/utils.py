from collections import Iterable


def flatten(collection):
    """
    flatten arbitrary combination of collections to collection
     for ex: [[2, 4, [2]], 3] -> [2, 4, 2, 3]
    :param collection:
    :return:
    """
    for i in collection:
        if isinstance(i, Iterable):
            for j in flatten(i):
                yield j
        else:
            yield i
