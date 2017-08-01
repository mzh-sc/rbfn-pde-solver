from collections import Iterable


def flatten(collection):
    for i in collection:
        if isinstance(i, Iterable):
            for j in flatten(i):
                yield j
        else:
            yield i
