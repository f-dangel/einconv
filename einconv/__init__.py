"""Einsum implementations of convolutions and related operations."""

from einconv.conv_index_pattern import index_pattern
from einconv.simplifications import simplify

__all__ = [
    "index_pattern",
    "simplify",
]
