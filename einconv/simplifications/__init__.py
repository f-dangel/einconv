"""Simplifies einsum expressions."""

from typing import List, Tuple

from torch import Tensor


def simplify(
    equation: str, operands: List[Tensor]
) -> Tuple[str, List[Tensor], Tuple[int]]:
    """Simplify an einsum expressions.

    Args:
        equation: Einsum equation.
        operands: Einsum operands.

    Returns:
        Simplified einsum equation
        Simplified einsum operands
        Output shape

    Raises:
        NotImplementedError
    """
    raise NotImplementedError
