import unittest
from miniflow import *
import numpy.testing as npt


class TestMiniflow(unittest.TestCase):
    def test_miniflow_add(self):
        x, y, z = Input(), Input(), Input()
        f = Add(x, y, z)
        feed_dict = {x: 10, y: 5, z: 2}
        sorted_nodes = topological_sort(feed_dict)
        output = forward_pass(f, sorted_nodes)
        self.assertEqual(output, feed_dict[x] + feed_dict[y] + feed_dict[z])

    def test_miniflow_mul(self):
        x, y, z = Input(), Input(), Input()
        f = Mul(x, y, z)
        feed_dict = {x: 10, y: 5, z: 2}
        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)
        self.assertEqual(output, feed_dict[x] * feed_dict[y] * feed_dict[z])

    def test_miniflow_linear(self):
        X, W, b = Input(), Input(), Input()
        f = Linear(X, W, b)
        X_ = np.array([[-1., -2.], [-1, -2]])
        W_ = np.array([[2., -3], [2., -3]])
        b_ = np.array([-3., -5])
        feed_dict = {X: X_, W: W_, b: b_}
        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)
        npt.assert_array_almost_equal(np.array([[-9., 4.], [-9., 4.]]), output)

    def test_miniflow_linear2(self):
        X, W, b = Input(), Input(), Input()
        f = Linear(X, W, b)
        X_ = np.array([[-1., -2.], [-1, -2]])
        W_ = np.array([[2., -3,4], [2., -3,4]])
        b_ = np.array([-3., -5,4])
        feed_dict = {X: X_, W: W_, b: b_}
        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)
        npt.assert_array_almost_equal(np.array([[-9, 4, -8],
                                                [-9, 4, -8]]), output)

    def test_miniflow_linear_sigmoid(self):
        X, W, b = Input(), Input(), Input()
        f = Linear(X, W, b)
        g = Sigmoid(f)
        X_ = np.array([[-1., -2.], [-1, -2]])
        W_ = np.array([[2., -3], [2., -3]])
        b_ = np.array([-3., -5])
        feed_dict = {X: X_, W: W_, b: b_}
        graph = topological_sort(feed_dict)
        output = forward_pass(g, graph)
        npt.assert_array_almost_equal(np.array([[1.23394576e-04, 9.82013790e-01],
                                                [1.23394576e-04, 9.82013790e-01]]), output)

    def test_mse(self):
        y, a = Input(), Input()
        cost = MSE(y, a)
        y_ = np.array([1, 2, 3])
        a_ = np.array([4.5, 5, 10])
        feed_dict = {y: y_, a: a_}
        graph = topological_sort(feed_dict)
        forward_pass(cost,graph)
        self.assertAlmostEqual(23.416,cost.value,2)

    def test_gradients_linear(self):
        X, W, b = Input(), Input(), Input()
        y = Input()
        f = Linear(X, W, b)
        a = Sigmoid(f)
        cost = MSE(y, a)

        X_ = np.array([[-1., -2.], [-1, -2]])
        W_ = np.array([[2.], [3.]])
        b_ = np.array([-3.])
        y_ = np.array([1, 2])

        feed_dict = {
            X: X_,
            y: y_,
            W: W_,
            b: b_,
        }

        graph = topological_sort(feed_dict)
        forward_and_backward(graph)
        # return the gradients for each Input
        gradients = [t.gradients[t] for t in [X, y, W, b]]
        npt.assert_array_almost_equal(np.array([[-3.34 / 100000, -5.01 / 100000],
                                                [-6.68 / 100000, -1 / 10000]]), gradients[0], 6)

