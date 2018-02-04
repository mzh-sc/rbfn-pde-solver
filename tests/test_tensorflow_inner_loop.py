import tensorflow as tf
import unittest as ut


class TestTensorflowInnerLoop(ut.TestCase):
    def test_try_while_loop_increment_matrix(self):
        def body(x): # x is previous body output
            return x + 1

        def condition(x):
            return tf.reduce_sum(x) < 10

        x = tf.Variable(tf.constant(1, shape=[2, 2]))

        with tf.Session():
            tf.initialize_all_variables().run()
            result = tf.while_loop(condition, body, [x])
            print(result.eval()) #[[3 3] [3 3]]

    def test__try_while_in_while(self):
        i = tf.constant(0, name='i')

        def outer_while(i):
            j = tf.constant(0, name='j')
            tf.while_loop(lambda i, j: j < (i + 1) * 10,
                          lambda i, j: (i, tf.add(j, 1, name='j_increment')),
                          (i, j), name='inner_while')
            return tf.add(i, 1, name='add1')

        res = tf.while_loop(lambda i: tf.less(i, 10), outer_while, [i], name='outer_while')
        with tf.Session() as s:
            print(s.run(res)) # 10
            print(s.run(i)) # 0 - it is constant

    def _get_tensor_created_in_while_context(self):
        """Creates and returns a tensor from a while context."""
        # Note: I can not declare tensor as None variable here as Python will interpret it as undeclared in the body
        # that's why have to use this list 'hack' just to save variable somewhere
        tensor = None

        def body(i):
            nonlocal tensor
            if tensor is None:
                # the problem is here. Global variable tensor was assigned in the previous iteration and
                # can not change in the next iteration
                # Note: I even can not reuse them at the next iterations
                tensor = tf.constant(1, name='const_1')
            return tf.add(i, tensor, name='add')

        tf.while_loop(lambda i: i < 10, body, [0], name="inner_while")

        return tensor

    def test_try_invalid_context_in_while(self):
        # Accessing  a  while loop tensor in a different while loop is illegal.
        while_tensor = self._get_tensor_created_in_while_context()

        # at the next iteration we can not use tensor created at the previous iteration
        with self.assertRaisesRegex(ValueError,
                                    "Cannot use 'outer_while/NextIteration' as input to 'inner_while/const_1' because "
                                    "they are in different while loops. See info log for more details."):
            tf.while_loop(lambda i: True, lambda i: while_tensor, [0], name='outer_while')

    def test_try_valid_context_in_while(self):
        # Accessing a tensor in a nested while is OK.
        def body(_):
            c = tf.constant(1)
            return tf.while_loop(lambda i: i < 10, lambda i: i + c, [0])

        with tf.Session() as s:
            print(s.run(tf.while_loop(lambda i: i < 5, body, [0])))

    def test_try_wrong_context_to_create_variable_w(self):
        i = tf.constant(0)

        def body(i):
            w = tf.Variable(tf.constant(1))
            return [i + w]

        loop = tf.while_loop(lambda i: tf.less(i, 5), body, [i])
        s = tf.Session()
        s.run(tf.global_variables_initializer())

