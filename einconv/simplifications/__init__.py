"""Simplifies einsum expressions."""

from typing import List, Tuple, Union

from torch import Tensor
from torch.nn import Parameter


def simplify(
    equation: str, operands: List[Union[Tensor, Parameter]]
) -> Tuple[str, List[Tensor], Tuple[int]]:
    """Simplify an einsum expressions.

    Args:
        equation: Einsum equation.
        operands: Einsum operands.

    # noqa: DAR202

    Returns:
        Simplified einsum equation
        Simplified einsum operands
        Output shape

    Raises:
        NotImplementedError: This function still needs to be implemented.
    """
    raise NotImplementedError
