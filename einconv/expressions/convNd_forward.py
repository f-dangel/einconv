"""Generates einsum expression of the forward pass of a convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

from einconv import index_pattern
from einconv.expressions.utils import create_conv_index_patterns
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    weight: Union[Tensor, Parameter],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
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

    Returns:
        Einsum equation
        Einsum operands in order un-grouped input, patterns, un-grouped weight
        Output shape: ``[batch_size, out_channels, *output_sizes]``.
    """
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x, weight, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    return equation, operands, shape


def _operands_and_shape(
    x: Tensor,
    weight: Tensor,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Prepare operands for contraction with einsum.

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

    Returns:
        Tensor list containing the operands in order un-grouped input, index patterns, \
        un-grouped weight.
        Output shape.
    """
    N = x.dim() - 2
    input_size = x.shape[2:]
    kernel_size = weight.shape[2:]
    patterns = create_conv_index_patterns(
        N,
        input_size,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        device=weight.device,
        dtype=weight.dtype,
    )
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_out) ... -> g c_out ...", g=groups)
    operands = [x_ungrouped, *patterns, weight_ungrouped]

    output_sizes = [p.shape[1] for p in patterns]
    batch_size = x.shape[0]
    out_channels = weight.shape[0]
    shape = (batch_size, out_channels, *output_sizes)

    return operands, shape


def _equation(N: int) -> str:
    """Generate einsum equation for convolution.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for N-dimensional convolution. Operand order is un-grouped \
        input, patterns, un-grouped weight.
    """
    input_str = ""
    output_str = ""
    pattern_strs: List[str] = []
    kernel_str = ""

    # requires 4 + 3 * N letters
    letters = get_letters(4 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    input_str += batch_letter
    output_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    input_str += group_letter
    output_str += group_letter
    kernel_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    input_str += in_channel_letter
    output_str += out_channel_letter
    kernel_str += out_channel_letter + in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        input_str += input_letter
        output_str += output_letter
        kernel_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([input_str] + pattern_strs + [kernel_str])

    return "->".join([input_equation, output_str])
