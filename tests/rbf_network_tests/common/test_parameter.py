import unittest

from rbf_network.common.parameter import Parameter


class TestParameter(unittest.TestCase):
    def test_undo(self):
        p = Parameter()

        p.value = 5
        self.assertEqual(p.value, 5)

        p.value = 6
        self.assertEqual(p.value, 6)

        p.undo()
        self.assertEqual(p.value, 5)

        p.undo()
        self.assertEqual(p.value, 0)

        p.undo()
        self.assertEqual(p.value, 0)


if __name__ == '__main__':
    unittest.main()
