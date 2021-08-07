from typing import Callable, SupportsFloat
from inspect import getfullargspec
from toygrad import dual, var


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

    # Check that the user specified with respect to which variable to take the derivative
    # when working with multi-variable functions.
    if is_multivariate and wrt is None:
        raise ValueError("wrt must be provided for multivariate functions")
    else:
        # Use the first argument when working with single-variable functions
        wrt = func_args[0] if wrt is None else wrt

    # Ensure that `wrt` matches the name of a valid argument for the function
    if wrt is not None and wrt not in func_args:
        raise ValueError(f"'{wrt}' is not a valid argument name")

    # Find the index of the variable with respect to which the derivative is to be taken
    target_variable_index = func_args.index(wrt)
    processed_args = []

    for index, arg in enumerate(at):
        # Treat all arguments, except the target variable's, as constants
        is_wrt = index == target_variable_index
        processed_args.append(var(arg) if is_wrt else dual(arg))

    return of(*processed_args).grad
