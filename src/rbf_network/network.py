import tensorflow as tf

class Network(object):
    def __init__(self, rbfs):
        self._functions = rbfs
        self.weights = None

    @property
    def _dimention(self):
        return self._functions[0].dimention if self._functions else 0

    def y(self, x):
        if len(x.shape) not in (1, 2) or x.shape[0] != self._dimention:
            raise Exception("Unexpected shape {}. The current implementation can handle "
                            "either (:,dim) or (dim) shapes only".format(x.shape))
        return tf.reduce_sum(self.weights * tf.stack([e.y(x) for e in self]))

    def __iter__(self):
        return iter(self._functions)
