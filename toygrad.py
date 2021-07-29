from typing import SupportsFloat


class dual:
    """Implements a basic type for dual numbers.
    See https://en.wikipedia.org/wiki/Dual_number.
    """

    def __init__(self, real: SupportsFloat, grad: SupportsFloat = None, var=False):
        """Returns a new dual number.

        Args:
            real: The real part of the dual number.
            grad: The gradient of the dual number.
            var: Whether the dual number stands for a variable instead of a constant.
        Returns:
            A dual number.
        """
        self.real = real
        self.grad = grad

        if grad is None:
            self.grad = 1.0 if var else 0.0

    def __add__(self, other):
        return dual(self.real + other.real, self.grad + other.grad)

    def __sub__(self, other):
        return dual(self.real - other.real, self.grad - other.grad)

    def __mul__(self, other):
        derivative = other.real * self.grad + self.real * other.grad
        return dual(self.real * other.real, derivative)

    def __div__(self, other):
        derivative = (1 / (other.real ** 2)) * (
            other.real * self.grad - self.real * other.grad
        )
        return dual(self.real / other, derivative)

    def __pow__(self, other: float):
        return dual(self.real ** other, other * self.real ** (other - 1))

# TODO: Add partial differentiation
def grad(fn, x: dual):
    return fn(x).grad


assert grad(lambda x: x + dual(3.0), dual(3.0, var=True)) == 1
assert grad(lambda x: x ** 2, dual(3.0, var=True)) == 6.0
assert grad(lambda x: (x ** 2.0) * x, dual(3.0, var=True)) == 27.0
