"""Generates einsum expression for the forward pass of a transpose convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

import einconv
from einconv.expressions.utils import (
    create_conv_transpose_index_patterns,
    translate_to_torch,
)


def einsum_expression(
    x: Tensor,
    weight: Union[Tensor, Parameter],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    output_padding: Union[int, Tuple[int, ...]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    simplify: bool = True,
) -> Tuple[str, List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Generate einsum expression of a transpose convolution's forward pass.

    Args:
        x: Input to the transpose convolution. Has shape
            ``[batch_size, in_channels, *input_sizes]`` where ``len(input_sizes) == N``.
        weight: Kernel of the transpose convolution. Has shape ``[in_channels,
            out_channels / groups, *kernel_size]`` where ``kernel_size`` is an
            ``N``-tuple of kernel dimensions.
        stride: Stride of the associated convolution. Can be a single integer (shared
            along all spatial dimensions), or an ``N``-tuple of integers.
            Default: ``1``.
        padding: Padding of the associated convolution. Can be a single integer (shared
            along all spatial dimensions) or an ``N``-tuple of integers. Default: ``0``.
        output_padding: Additional padding added to one side of the output shape.
            Serves to make the output space of the transpose convolution unambiguous.
            Default: ``0``.
        groups: In how many groups to split the input channels. Default: ``1``.
        dilation: Dilation of the associated convolution. Can be a single integer
            (shared along all spatial dimensions), or an ``N``-tuple of integers.
            Default: ``1``.
        simplify: Whether to simplify the einsum expression. Default: ``True``.

    Returns:
        Einsum equation
        Einsum operands in order un-grouped input, patterns, un-grouped weight
        Output shape: ``[batch_size, out_channels, *output_sizes]``.
    """
    # NOTE Transpose convolution is often defined relative to a convolution.
    # This leads to confusing names. For instance the spatial dimensions of the
    # **input** to a convolution can be called 'output dimensions' as they
    # correspond to the spatial dimensions of the associated convolution's
    # input. Internally, we try to use names w.r.t. the associated convolution,
    # while PyTorch's naming convention regards transpose convolution
    # independently of its associated convolution.
    N = x.dim() - 2

    # construct einsum equation, use index conventions from convolution for
    # spatial indices (convolution's output space is transpose convolution's
    # input space)
    x_str = "n g c_in " + " ".join([f"o{i}" for i in range(N)])
    pattern_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    weight_str = "g c_in c_out " + " ".join([f"k{i}" for i in range(N)])
    lhs = ",".join([x_str, *pattern_strs, weight_str])

    rhs = "n g c_out " + " ".join([f"i{i}" for i in range(N)])

    equation = "->".join([lhs, rhs])
    equation = translate_to_torch(equation)

    # construct einsum operands
    patterns = create_conv_transpose_index_patterns(
        N,
        output_size=x.shape[2:],
        kernel_size=weight.shape[2:],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        device=weight.device,
        dtype=weight.dtype,
    )
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_in) ... -> g c_in ...", g=groups)
    operands = [x_ungrouped, *patterns, weight_ungrouped]

    # construct output shape
    output_size = [p.shape[2] for p in patterns]
    batch_size = x.shape[0]
    out_channels = weight.shape[1] * groups
    shape = (batch_size, out_channels, *output_size)

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
