"""Simplifies einsum expressions."""

from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import Parameter

from einconv.simplifications.opt import TensorNetwork


def simplify(
    equation: str, operands: List[Union[Tensor, Parameter]]
) -> Tuple[str, List[Union[Tensor, Parameter]]]:
    """Simplify an einsum expressions.

    Args:
        equation: Einsum equation.
        operands: Einsum operands.

    Returns:
        Simplified einsum equation.
        Simplified einsum operands.
    """
    tn = TensorNetwork(equation, operands)
    tn.simplify()
    return tn.generate_expression()
