"""Generates einsum expression of the forward pass of a convolution."""

from typing import Optional, Tuple

from torch import Tensor


def einsum_expression(
    simplify: Optional[bool] = False,
) -> Tuple[str, Tuple[Tensor], Tuple[int]]:
    """Generate einsum expression of a convolution's forward pass.

    Args:
        simplify: Whether to simplify the expression. Default: ``False``.

    Returns:
        Einsum equation
        Einsum operands
        Output shape

    Raises:
        NotImplementedError
    """
    raise NotImplementedError
