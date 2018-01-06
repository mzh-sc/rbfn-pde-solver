import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D #to make 3d projection working
from unittest import TestCase
from matplotlib import cm

class TestMatplotlib(TestCase):
    def test_try_3d_surface(self):
        fig = plt.figure()
        axes = fig.gca(projection='3d') #Get the current Axes instance on the current figure matching the given keyword args, or create one.

        # Make data.
        X = np.arange(-5, 5, 0.1)
        Y = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)

        # Plot the surface.
        surface = axes.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        axes.set_zlim(-1.01, 1.01)
        axes.zaxis.set_major_locator(LinearLocator(10)) #linear 10 ticks
        axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surface, shrink=0.5, aspect=5)

        plt.show()

    def test_try_2d_3d(self):
        # Twice as wide as it is tall
        # default size ratio
        w, h = plt.figaspect(0.5)
        fig = plt.figure(figsize=(w,h))
        fig.suptitle('2D & 3D Plots')


        # ----------- 2d -----------
        t1 = np.arange(0.0, 5.0, 0.1)
        t2 = np.arange(0.0, 5.0, 0.02)

        axes = fig.add_subplot(1, 2, 1)  #nrows, ncols, plot_number
        l = axes.plot(t1, np.cos(t1), 'bo',
                      t2, np.sin(t2), 'k--', markerfacecolor='green')
        axes.grid(True)
        axes.set_ylabel('Damped oscillation')


        #----------- 3d -----------
        axes = fig.add_subplot(1, 2, 2, projection='3d') #nrows, ncols, plot_number

        # Make data.
        X = np.arange(-5, 5, 0.1)
        Y = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)

        # Plot the surface.
        surface = axes.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        axes.set_zlim(-1.01, 1.01)
        axes.zaxis.set_major_locator(LinearLocator(10)) #linear 10 ticks
        axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surface, shrink=0.5, aspect=5)

        plt.show()



