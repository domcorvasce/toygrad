from inspect import getfullargspec
from typing import SupportsFloat
from math import floor


class dual:
    """Implements a basic type for dual numbers.
    See https://en.wikipedia.org/wiki/Dual_number.
    """

    def __init__(
        self, real: SupportsFloat, grad: SupportsFloat = None, var: bool = False
    ):
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
            self.grad = 1.0 if var is True else 0.0

    def __get_rval(self, other: SupportsFloat) -> "dual":
        """Returns a dual number from the right value of the operation.

        Args:
            other: The right value of the operation.
        Returns:
            A dual number.
        """
        return other if hasattr(other, "grad") else dual(other)

    def __add__(self, other: SupportsFloat) -> "dual":
        rval = self.__get_rval(other)
        return dual(self.real + rval.real, self.grad + rval.grad)

    def __sub__(self, other: SupportsFloat) -> "dual":
        rval = self.__get_rval(other)
        return dual(self.real - rval.real, self.grad - rval.grad)

    def __mul__(self, other: SupportsFloat) -> "dual":
        rval = self.__get_rval(other)
        return dual(
            self.real * rval.real, (rval.real * self.grad) + (self.real * rval.grad)
        )

    def __truediv__(self, other: SupportsFloat) -> "dual":
        rval = self.__get_rval(other)
        return dual(
            self.real / rval.real,
            (1 / (rval.real ** 2))
            * ((rval.real * self.grad) - (self.real * rval.grad)),
        )

    def __floordiv__(self, other: SupportsFloat) -> "dual":
        return floor(self.__truediv__(self, other))

    def __pow__(self, other: SupportsFloat) -> "dual":
        rval = self.__get_rval(other)

        # Handle the special case of e^x
        if self.real == exp.real:
            return dual(self.real ** rval.real, self.real ** rval.real)

        return dual(
            self.real ** rval.real,
            rval.real * (self.real ** (rval.real - 1.0) * self.grad),
        )

    def __float__(self) -> float:
        return self.real


def grad(fn, **args):
    """Returns the gradient of a function.

    Args:
        fn: The function that must be differentiated.
        args: The arguments to be passed to the function.
    Returns:
        The gradient of the function.
    """
    gradient = {}
    variables = dict.fromkeys(getfullargspec(fn).args, dual(0))

    for arg in args.keys():
        # Treat the variable as such, in order to avoid it being treated as a constant and its derivative being 0.
        variables[arg] = dual(args[arg], var=True)
        # Compute the partial derivative
        gradient[arg] = fn(**variables).grad
        # Reset the variable value to the constant 0
        variables[arg] = dual(0.000000001)
    return gradient


exp = dual(2.718281828459045)
