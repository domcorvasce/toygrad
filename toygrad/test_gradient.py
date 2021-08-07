import unittest
from __init__ import gradient


class TestGradient(unittest.TestCase):
    def test_throws_error_if_wrt_param_is_invalid(self):
        try:
            gradient(of=lambda x: x, wrt="y")
        except ValueError as err:
            self.assertEqual(err.__str__(), "'y' is not a valid argument name")
            return
        self.assertEqual(True, False)

    def test_throws_error_if_wrt_is_omitted_for_multivariate_funcs(self):
        try:
            gradient(of=lambda x, y: x + y)
        except ValueError as err:
            self.assertEqual(
                err.__str__(), "wrt must be provided for multivariate functions"
            )
            return
        self.assertEqual(True, False)


if __name__ == "__main__":
    unittest.main()
