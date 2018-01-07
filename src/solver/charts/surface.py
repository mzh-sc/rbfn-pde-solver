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

        X = np.linspace(x0, x1, x_count, dtype=nn.type.as_numpy_dtype)
        Y = np.linspace(y0, y1, y_count, dtype=nn.type.as_numpy_dtype)
        self.__X, self.__Y = np.meshgrid(X, Y) # (N,N), (N,N)
        self.__function = function

    def update(self):
        # clear
        self.__axes.cla()

        with tf.name_scope("chart-surface-z-values"):
            self.__axes.plot_surface(self.__X, self.__Y,
                tf.get_default_session().run(  # get z values in as a grid NxN
                    tf.reshape(
                        tf.map_fn(lambda x: self.__function(x),  # map_fn (N*N,2) to (N*N,1)
                                  tf.stack(
                                      (tf.reshape(self.__X, [-1]),  # (N,N) to (N*N). For ex. [[1, 2, 3], [1, 2, 3], [1, 2, 3]] -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
                                       tf.reshape(self.__Y, [-1])),  # (N,N) to (N*N). For ex. [[1, 1, 1], [2, 2, 2], [3, 3, 3]] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
                                      axis=-1)),  # stack: (N*N,2) [[1, 1], [2, 1], [3, 1], [2, 1], [2, 2]...]
                        shape=self.__X.shape)),  # (N*N,1) to (N, N)
                                      cmap=cm.coolwarm,
                                      linewidth=0,
                                      antialiased=False)
