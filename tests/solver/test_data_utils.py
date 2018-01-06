from unittest import TestCase
from solver.data_utils import *

import numpy as np

from utils import flatten


class TestDataUtils(TestCase):
    def test_random_points_1d(self):
        points = random_points_1d(1, 2, 10)

        self.assertEquals(len(points), 10)
        self.assertEqual(type(points), list)
        self.assertEqual(np.array(points).shape, (10, 1))

        self.assertTrue(all([(1 <= e[0] <= 2) for e in points]))

    def test_random_points_2d(self):
        points = random_points_2d(1, 2, 2, 3, 10)

        self.assertEquals(len(points), 10)
        self.assertEqual(type(points), list)
        self.assertEqual(np.array(points).shape, (10, 2))

        self.assertTrue(all(
            [(1 <= e[0] <= 2) and
             (2 <= e[1] <= 3) for e in points]))

    def test_uniform_points_1d(self):
        points = uniform_points_1d(1, 2, 6)

        self.assertEquals(len(points), 6)
        self.assertEqual(type(points), list)
        self.assertEqual(np.array(points).shape, (6, 1))

        self.assertSequenceEqual(list(flatten(points)), [1, 1.2, 1.4, 1.6, 1.8, 2.0])

    def test_uniform_points_2d(self):
        points = uniform_points_2d(1, 2, 3, 2, 3, 3)

        self.assertEquals(len(points), 9)
        self.assertEqual(type(points), list)
        self.assertEqual(np.array(points).shape, (9, 2))

        self.assertSequenceEqual(list(flatten(points)), list(flatten(
                                                            [[1.0, 2.0], [1.5, 2.0], [2, 2.0],
                                                            [1.0, 2.5], [1.5, 2.5], [2.0, 2.5],
                                                            [1.0, 3.0], [1.5, 3.0], [2, 3.0]])))