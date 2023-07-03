"""Generates einsum expression of the input VJP of a convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

import einconv
from einconv.expressions.utils import create_conv_index_patterns, translate_to_torch
from einconv.utils import _tuple


def einsum_expression(
    weight: Union[Tensor, Parameter],
    v: Tensor,
    input_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    simplify: bool = True,
) -> Tuple[str, List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Generate einsum expression of a convolution's input VJP.

    Args:
        weight: Kernel of the convolution. Has shape ``[out_channels,
            in_channels / groups, *kernel_size]`` where ``kernel_size`` is an
            ``N``-tuple of kernel dimensions.
        v: Vector multiplied by the Jacobian. Has shape
            ``[batch_size, out_channels, *output_sizes]``
            where ``len(output_sizes) == N`` (same shape as the convolution's output).
        input_size: Spatial dimensions of the convolution. Can be a single integer
            (shared along all spatial dimensions), or an ``N``-tuple of integers.
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
        Einsum operands in order un-grouped vector, patterns, un-grouped weight
        Output shape: ``[batch_size, in_channels, *input_sizes]``
    """
    N = weight.dim() - 2

    # construct einsum equation
    v_str = "n g c_out " + " ".join([f"o{i}" for i in range(N)])
    pattern_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    weight_str = "g c_out c_in " + " ".join([f"k{i}" for i in range(N)])
    lhs = ",".join([v_str, *pattern_strs, weight_str])

    rhs = "n g c_in " + " ".join([f"i{i}" for i in range(N)])

    equation = "->".join([lhs, rhs])
    equation = translate_to_torch(equation)

    # construct einsum operands
    patterns = create_conv_index_patterns(
        N,
        input_size,
        kernel_size=weight.shape[2:],
        stride=stride,
        padding=padding,
        dilation=dilation,
        device=weight.device,
        dtype=weight.dtype,
    )
    v_ungrouped = rearrange(v, "n (g c_out) ... -> n g c_out ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_out) ... -> g c_out ...", g=groups)
    operands = [v_ungrouped, *patterns, weight_ungrouped]

    # construct output shape
    batch_size = v.shape[0]
    group_in_channels = weight.shape[1]
    t_input_size = _tuple(input_size, N)
    shape = (batch_size, groups * group_in_channels, *t_input_size)

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
