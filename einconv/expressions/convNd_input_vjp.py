"""Generates einsum expression of the input VJP of a convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

from einconv.expressions.utils import create_conv_index_patterns
from einconv.utils import _tuple, get_letters


def einsum_expression(
    weight: Union[Tensor, Parameter],
    v: Tensor,
    input_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
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

    Returns:
        Einsum equation
        Einsum operands in order un-grouped vector, patterns, un-grouped weight
        Output shape: ``[batch_size, in_channels, *input_sizes]``
    """
    N = weight.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        weight,
        v,
        input_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return equation, operands, shape


def _operands_and_shape(
    weight: Union[Tensor, Parameter],
    v: Tensor,
    input_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Prepare operands for contraction with einsum.

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

    Returns:
        Tensor list containing the operands in order un-grouped vector, patterns, \
        un-grouped weight.
        Output shape
    """
    N = weight.dim() - 2
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
    v_ungrouped = rearrange(v, "n (g c_out) ... -> n g c_out ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_out) ... -> g c_out ...", g=groups)
    operands = [v_ungrouped, *patterns, weight_ungrouped]

    batch_size = v.shape[0]
    group_in_channels = weight.shape[1]
    t_input_size = _tuple(input_size, N)
    shape = (batch_size, groups * group_in_channels, *t_input_size)

    return operands, shape


def _equation(N: int) -> str:
    """Return the einsum equation for an input VJP.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for the input VJP of N-dimensional convolution. Operand \
        order is un-grouped vector, patterns, un-grouped weight.
    """
    v_str = ""
    output_str = ""
    pattern_strs: List[str] = []
    weight_str = ""

    # requires 4 + 3 * N letters
    letters = get_letters(4 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    v_str += batch_letter
    output_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    v_str += group_letter
    output_str += group_letter
    weight_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    v_str += out_channel_letter
    output_str += in_channel_letter
    weight_str += out_channel_letter + in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        v_str += output_letter
        output_str += input_letter
        weight_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([v_str] + pattern_strs + [weight_str])

    return "->".join([input_equation, output_str])
