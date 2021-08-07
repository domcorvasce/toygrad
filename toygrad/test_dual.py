import unittest
from toygrad import dual, var


class TestDual(unittest.TestCase):
    def test_derivative_of_constant(self):
        self.assertEqual(dual(3.5).grad, 0)

    def test_derivative_of_variable(self):
        self.assertEqual(var(3.5).grad, 1)

    def test_sum(self):
        self.assertEqual((dual(3) + dual(4.5)).real, 7.5)
        self.assertEqual((dual(3) + 4.5).real, 7.5)

    def test_sub(self):
        self.assertEqual((dual(3) - dual(4.5)).real, -1.5)
        self.assertEqual((dual(3) - 4.5).real, -1.5)

    def test_mul(self):
        product_of_constants = dual(3) * dual(4)
        derivative_with_respect_to_x = var(3) * dual(4)
        derivative_with_respect_to_y = dual(3) * var(4)

        # Check the product of two constants
        self.assertEqual(product_of_constants.real, 12)
        self.assertEqual(product_of_constants.grad, 0)

        # Check the product of a variable and a constant
        self.assertEqual(derivative_with_respect_to_x.grad, 4)
        self.assertEqual(derivative_with_respect_to_y.grad, 3)

    def test_div(self):
        quotient_of_constants = dual(3) / dual(4)
        derivative_wrt_to_x = var(3) / dual(4)
        derivative_wrt_to_y = dual(3) / var(4)

        # Check the quotient of two constants
        self.assertEqual(quotient_of_constants.real, 0.75)
        self.assertEqual(quotient_of_constants.grad, 0)

        # Check the quotient of a variable and a constant
        self.assertEqual(derivative_wrt_to_x.grad, 0.25)
        self.assertEqual(derivative_wrt_to_y.grad, -0.1875)


if __name__ == "__main__":
    unittest.main()
