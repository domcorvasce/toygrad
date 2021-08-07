from typing import SupportsFloat
from math import floor


class dual:
    """Implements a type for dual numbers."""

    def __init__(self, real: SupportsFloat, grad: SupportsFloat = 0.0):
        self.real = real
        self.grad = grad

    def __add__(self, rhs: SupportsFloat) -> "dual":
        """Returns the sum of two dual numbers.

        Args:
            rhs: The right-hand side of the sum.
        Returns:
            A dual number.
        """
        rhs = self.to_dual(rhs)
        return dual(self.real + rhs.real, self.grad + rhs.grad)

    def __sub__(self, rhs: SupportsFloat) -> "dual":
        """Returns the difference of two dual numbers.

        Args:
            rhs: The right-hand side of the difference.
        Returns:
            A dual number.
        """
        rhs = self.to_dual(rhs)
        return dual(self.real - rhs.real, self.grad - rhs.grad)

    def __mul__(self, rhs: SupportsFloat) -> "dual":
        """Returns the product of two dual numbers.

        Args:
            rhs: The right-hand side of the product.
        Returns:
            A dual number.
        """
        rhs = self.to_dual(rhs)
        derivative = rhs.real * self.grad + self.real * rhs.grad
        return dual(self.real * rhs.real, derivative)

    def __truediv__(self, rhs: SupportsFloat) -> "dual":
        """Returns the quotient of two dual numbers.

        Args:
            rhs: The right-hand side of the quotient.
        Returns:
            A dual number.
        """
        rhs = self.to_dual(rhs)
        derivative = (1 / (rhs.real ** 2)) * (
            (rhs.real * self.grad) - (self.real * rhs.grad)
        )
        return dual(self.real / rhs.real, derivative)

    def __floordiv__(self, rhs: SupportsFloat) -> "dual":
        quotient = self / rhs
        return dual(floor(quotient.real), quotient.grad)

    def __pow__(self, exp: SupportsFloat) -> "dual":
        """Returns the power of a dual number.

        Args:
            exp: The exponent.
        Returns:
            A dual number.
        """
        exp = self.to_dual(exp)
        # The `self.grad` multiplied in order to compute the chain rule
        derivative = (exp.real * (self.real ** (exp.real - 1))) * self.grad
        return dual(self.real ** exp.real, derivative.real)

    def to_dual(self, value: SupportsFloat):
        """Converts a value to a dual number.

        Args:
            value: The value to convert.
        Returns:
            A dual number.
        """
        return value if hasattr(value, "__dual__") else dual(value)

    def __float__(self):
        """Returns the real part of the dual number."""
        return self.real

    def __dual__(self):
        """Indicates whether an object is a dual number"""
        return True


class var(dual):
    """Implements a type for dual variables."""

    def __init__(self, real: SupportsFloat):
        super().__init__(real, 1.0)
