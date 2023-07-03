"""Generates einsum expression of the weight VJP of a convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor

from einconv import index_pattern
from einconv.expressions.utils import create_conv_index_patterns
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    v: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expression of a convolution's weight VJP.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        v: Vector multiplied by the Jacobian. Has shape
            ``[batch_size, out_channels, *output_sizes]``
            where ``len(output_sizes) == N`` (same shape as the convolution's output).
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
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
        Einsum operands in order ungrouped input, patterns, ungrouped vector.
        Output shape: ``[out_channels, in_channels // groups, *kernel_size]``
    """
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x,
        v,
        kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
        groups=groups,
    )
    return equation, operands, shape


def _operands_and_shape(
    x: Tensor,
    v: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> Tuple[List[Tensor], Tuple[int, ...]]:
    """Prepare the tensor contraction operands for the VJP.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        v: Vector multiplied by the Jacobian. Has shape
            ``[batch_size, out_channels, *output_sizes]``
            where ``len(output_sizes) == N`` (same shape as the convolution's output).
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        groups: In how many groups to split the input channels. Default: ``1``.

    Returns:
        Einsum operands in order un-grouped input, patterns, un-grouped vector
        Output shape
    """
    N = x.dim() - 2
    input_size = x.shape[2:]
    patterns = create_conv_index_patterns(
        N,
        input_size,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        device=x.device,
        dtype=x.dtype,
    )
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    v_ungrouped = rearrange(v, "n (g c_out) ... -> n g c_out ...", g=groups)
    operands = [x_ungrouped, *patterns, v_ungrouped]

    in_channels = x.shape[1]
    out_channels = v.shape[1]
    t_kernel_size = _tuple(kernel_size, N)
    shape = (out_channels, in_channels // groups, *t_kernel_size)

    return operands, shape


def _equation(N: int) -> str:
    """Return the einsum equation for a weight VJP.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for the weight VJP. Argument order is assumed to
        be ``x, *patterns, v``.
    """
    v_str = ""
    x_str = ""
    output_str = ""
    pattern_strs: List[str] = []

    # requires 4 + 3 * N letters
    letters = get_letters(4 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    v_str += batch_letter
    x_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    v_str += group_letter
    output_str += group_letter
    x_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    v_str += out_channel_letter
    output_str += out_channel_letter + in_channel_letter
    x_str += in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        x_str += input_letter
        v_str += output_letter
        output_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([x_str] + pattern_strs + [v_str])

    return "->".join([input_equation, output_str])
