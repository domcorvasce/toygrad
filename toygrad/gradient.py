from typing import Callable, SupportsFloat
from inspect import getfullargspec


def gradient(of: Callable, wrt: str = None, at: any = []) -> SupportsFloat:
    """Compute the partial derivative of a function with respect to a variable.

    Args:
        of: The function whose partial derivative is to be computed.
        wrt: The variable with respect to which the partial derivative is to be computed.
        at: The arguments to the function.

    Returns:
        The value of the partial derivative of the function.
    """
    func_args = getfullargspec(of).args
    is_multivariate = len(func_args) > 1

    # A constant function has a derivative of zero.
    if len(func_args) == 0:
        return 0.0

    # Check that the user specified the variable of which the derivative must be computed.
    if is_multivariate and wrt is None:
        raise ValueError("wrt must be provided for multivariate functions")
    else:
        # Set default value for the wrt argument if necessary
        wrt = func_args[0] if wrt is None else wrt

    if wrt is not None and wrt not in func_args:
        raise ValueError(f"'{wrt}' is not a valid argument name")

    return of(*at)
