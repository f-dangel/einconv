"""Generates einsum expression of the forward pass of a convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

import einconv
from einconv.expressions.utils import create_conv_index_patterns, translate_to_torch


def einsum_expression(
    x: Tensor,
    weight: Union[Tensor, Parameter],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    simplify: bool = True,
) -> Tuple[str, List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Generate einsum expression of a convolution's forward pass.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        weight: Kernel of the convolution. Has shape ``[out_channels,
            in_channels / groups, *kernel_size]`` where ``kernel_size`` is an
            ``N``-tuple of kernel dimensions.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        groups: In how many groups to split the input channels. Default: ``1``.
        simplify: Whether to simplify the einsum expression. Default: ``True``.

    Returns:
        Einsum equation
        Einsum operands in order un-grouped input, patterns, un-grouped weight
        Output shape: ``[batch_size, out_channels, *output_sizes]``.
    """
    N = x.dim() - 2

    # construct einsum equation
    x_str = "n g c_in " + " ".join([f"i{i}" for i in range(N)])
    pattern_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    weight_str = "g c_out c_in " + " ".join([f"k{i}" for i in range(N)])
    lhs = ",".join([x_str, *pattern_strs, weight_str])

    rhs = "n g c_out " + " ".join([f"o{i}" for i in range(N)])

    equation = "->".join([lhs, rhs])
    equation = translate_to_torch(equation)

    # construct einsum operands
    patterns = create_conv_index_patterns(
        N,
        input_size=x.shape[2:],
        kernel_size=weight.shape[2:],
        stride=stride,
        padding=padding,
        dilation=dilation,
        device=weight.device,
        dtype=weight.dtype,
    )
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_out) ... -> g c_out ...", g=groups)
    operands = [x_ungrouped, *patterns, weight_ungrouped]

    # construct output shape
    output_size = [p.shape[1] for p in patterns]
    batch_size = x.shape[0]
    out_channels = weight.shape[0]
    shape = (batch_size, out_channels, *output_size)

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
