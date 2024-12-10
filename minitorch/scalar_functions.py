"""Scalar Function Definitions"""

from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward method for the addition operation.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The first input to the addition operation.
        b : float
            The second input to the addition operation.

        Returns
        -------
        float
            The result of adding `a` and `b`.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward method for the addition operation.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[float, float]
            The gradients of the loss with respect to the inputs `a` and `b`, both equal to `d_output`.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward method for the logarithm function.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The input to the logarithm function.

        Returns
        -------
        float
            The logarithm of `a`.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward method for the logarithm function.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        float
            The gradient of the loss with respect to the input `a`.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# : Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication $f(x) = mul(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward method for the multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The first input to the multiplication.
        b : float
            The second input to the multiplication.

        Returns
        -------
        float
            The result of multiplying `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward method for the multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[float, float]
            The gradients of the loss with respect to the inputs `a` and `b`.

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = inv(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward method for the inverse (reciprocal) operation.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The input for the inverse operation.

        Returns
        -------
        float
            The reciprocal of `a`.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward method for the inverse operation.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        float
            The gradient of the loss with respect to the input `a`.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward method for the negation operation.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The input to negate.

        Returns
        -------
        float
            The negation of `a`.

        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward method for the negation operation.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        float
            The negation of `d_output`.

        """
        return -d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward method for the ReLU function.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The input to the ReLU function.

        Returns
        -------
        float
            The ReLU result, which is `max(0, a)`.

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward method for the ReLU function.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        float
            The gradient of the loss with respect to the input `a`, adjusted for the ReLU derivative.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(a) = exp(a)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward method for the exponential function.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The input for the exponential function.

        Returns
        -------
        float
            The exponential of `a`.

        """
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward method for the exponential function.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        float
            The gradient of the loss with respect to the input `a`, scaled by the exponential of `a`.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1/(1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward method for the sigmoid function.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The input to the sigmoid function.

        Returns
        -------
        float
            The sigmoid of `a`.

        """
        sig = operators.sigmoid(a)
        ctx.save_for_backward(sig)
        return float(sig)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward method for the sigmoid function.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        float
            The gradient of the loss with respect to the input `a`, based on the sigmoid derivative.

        """
        (sig,) = ctx.saved_values
        return d_output * sig * (1 - sig)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward method for the less-than operation.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The first input to compare.
        b : float
            The second input to compare.

        Returns
        -------
        float
            1.0 if `a < b`, otherwise 0.0.

        """
        # ctx is unused, but needed for the interface
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward method for the less-than operation.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[float, float]
            The gradients with respect to `a` and `b`, both 0 as the less-than operation is not differentiable.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equals function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward method for the equals operation.

        Parameters
        ----------
        ctx : Context
            The context object used to store information for the backward pass.
        a : float
            The first input to compare.
        b : float
            The second input to compare.

        Returns
        -------
        float
            1.0 if `a == b`, otherwise 0.0.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward method for the equals operation.

        Parameters
        ----------
        ctx : Context
            The context object containing stored values from the forward pass.
            This is not being used over here.
        d_output : float
            The gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[float, float]
            The gradients with respect to `a` and `b`, both 0 as the equals operation is not differentiable.

        """
        return 0.0, 0.0
