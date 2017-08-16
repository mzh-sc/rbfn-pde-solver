import numpy as np
import tensorflow as tf
import rbf_network as nn

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Surface(object):
    def __init__(self, figure, position,
                 x0, x1, x_count,
                 y0, y1, y_count,
                 function):
        self.__axes = figure.add_subplot(position, projection='3d')
        self.__axes.grid(True)
        self.__axes.set_ylabel('Solution')

        X = np.linspace(x0, x1, x_count)
        Y = np.linspace(y0, y1, y_count)
        self.__X, self.__Y = np.meshgrid(X, Y)
        self.__function = function

    def update(self):
        self.__axes.cla()
        self.__axes.plot_surface(self.__X, self.__Y,
            tf.get_default_session().run(
                tf.reshape(
                    tf.map_fn(lambda x: self.__function(x),
                              tf.stack((tf.reshape(self.__X.astype(np.float32), [-1]),
                                        tf.reshape(self.__Y.astype(np.float32), [-1])), axis=-1), dtype=nn.type), shape=[10, 10])),
                                  cmap=cm.coolwarm,
                                  linewidth=0,
                                  antialiased=False)
