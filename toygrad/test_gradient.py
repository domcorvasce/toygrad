import unittest
from toygrad import gradient


class TestGradient(unittest.TestCase):
    def test_throw_error_if_wrt_param_is_invalid(self):
        try:
            gradient(of=lambda x: x, wrt="y")
        except ValueError as err:
            self.assertEqual(err.__str__(), "'y' is not a valid argument name")
            return
        self.assertEqual(True, False)

    def test_throw_error_if_wrt_is_omitted_for_multivariate_funcs(self):
        try:
            gradient(of=lambda x, y: x + y)
        except ValueError as err:
            self.assertEqual(
                err.__str__(), "wrt must be provided for multivariate functions"
            )
            return
        self.assertEqual(True, False)

    def test_compute_derivative_of_single_variable_funcs(self):
        self.assertEqual(gradient(of=lambda x: x, wrt="x", at=[2]), 1)
        self.assertEqual(gradient(of=lambda x: x - 2, wrt="x", at=[2]), 1)
        self.assertEqual(gradient(of=lambda x: x + 4, wrt="x", at=[2]), 1)
        self.assertEqual(gradient(of=lambda x: x ** 2, wrt="x", at=[2]), 4)

    def test_compute_partial_derivative_of_two_variable_funcs(self):
        # Notice that the partial derivative of the function with respect to x
        # does not retain the y constant, and vice versa. We have:
        # ∂/∂x = 2x
        # ∂/∂y = 2y
        func = lambda x, y: (x ** 2) + (y ** 2)

        self.assertEqual(gradient(of=func, wrt="x", at=[2, 1]), 4)
        self.assertEqual(gradient(of=func, wrt="y", at=[2, 3]), 6)

        # Notice that the both arguments are retained in the partial derivatives
        # of the function defined below. We have:
        # ∂/∂x = (2 * x) / (y ** 2)
        # ∂/∂y = (2 * x ** 2) / (y ** 3)
        func2 = lambda x, y: (x ** 2) / (y ** 2)

        self.assertEqual(gradient(of=func2, wrt="x", at=[2, 3]), 4 / 9)
        self.assertEqual(gradient(of=func2, wrt="y", at=[2, 3]), -(8 / 27))


if __name__ == "__main__":
    unittest.main()
