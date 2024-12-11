"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two floating-point numbers.

    Args:
    ----
        x (float): The first number
        y (float): The second number

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Return the input unchanged.

    Args:
    ----
        x: Input value

    Returns:
    -------
        The same input value

    """
    return x


def add(x: float, y: float) -> float:
    """Adding two floating-point numbers.

    Args:
    ----
        x (float): The first number to add
        y (float): The second number to add

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a floating-point number.

    Args:
    ----
        x (float): The number to negate.

    Returns:
    -------
        float: The negated value of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if one floating-point number is less than another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is less than y, otherwise False.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if two floating-point numbers are equal.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is equal to y, otherwise False.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two floating-point numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of x and y.

    """
    return x if x > y else y


def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if the numbers are within 1e-2, False otherwise.

    """
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Computing the sigmoid function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply the ReLU activation function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The ReLU of x (max of 0 and x).

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of a floating-point number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential of a floating-point number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal of a floating-point number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The reciprocal of x (1/x).

    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Compute the derivative of the natural logarithm times a second argument.

    Args:
    ----
        x (float): The input value.
        y (float): The second argument.

    Returns:
    -------
        float: The derivative of log(x) * y.

    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Compute the derivative of the reciprocal times a second argument.

    Args:
    ----
        x (float): The input value.
        y (float): The second argument.

    Returns:
    -------
        float: The derivative of 1/x * y.

    """
    return (-1 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Compute the derivative of ReLU times a second argument.

    Args:
    ----
        x (float): The input value.
        y (float): The second argument.

    Returns:
    -------
        float: The derivative of ReLU(x) * y.

    """
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply.

    Returns:
    -------
        List[float]: A list with the function applied to each element.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:  # type: ignore
    """Combine elements from two iterables using a function.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.

    Returns:
    -------
        List[float]: A list with the function applied to corresponding elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value using a function and a starting value.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply to combine elements.
        start (float): The initial value to start the reduction.

    Returns:
    -------
        Callable[[Iterable[float]], float]: A function that applies the reduction to an iterable.

    The returned function:
    ----------------------

    Args:
    ----
        ls (Iterable[float]): The iterable of elements to be reduced.

    Returns:
    -------
        float: The result of reducing the iterable to a single value using the provided function.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The list of numbers.

    Returns:
    -------
        Iterable[float]: A list with all elements negated.

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        ls1 (Iterable[float]): The first list of numbers.
        ls2 (Iterable[float]): The second list of numbers.

    Returns:
    -------
        Iterable[float]: A list with corresponding elements added.

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The list of numbers.

    Returns:
    -------
        float: The sum of all elements in the list.

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The list of numbers.

    Returns:
    -------
        float: The product of all elements in the list.

    """
    return reduce(mul, 1.0)(ls)
