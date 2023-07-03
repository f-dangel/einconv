from typing import List, Tuple, Union

from torch import Tensor

from einconv.expressions.utils import create_conv_index_patterns
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[str, int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expression to unfold the input of a convolution.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.

    Returns:
        Einsum equation
        Einsum operands in order input, patterns
        Output shape: ``[batch_size, in_channels, tot_output_sizes]``
    """
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x, kernel_size, padding=padding, stride=stride, dilation=dilation
    )
    return equation, operands, shape


def _operands_and_shape(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
) -> Tuple[List[Tensor], Tuple[int, ...]]:
    """Prepare operands for contraction with einsum.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.

    Returns:
        Tensor list containing the operands in order input, patterns.
        Output shape.
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
    operands = [x, *patterns]

    output_tot_size = int(Tensor([p.shape[1] for p in patterns]).int().prod())
    t_kernel_size = _tuple(kernel_size, N)
    kernel_tot_size = int(Tensor(t_kernel_size).int().prod())
    batch_size, in_channels = x.shape[:2]
    shape = (batch_size, in_channels * kernel_tot_size, output_tot_size)

    return operands, shape


def _equation(N: int) -> str:
    """Generate einsum equation for unfold.

    The arguments are ``input, *index_patterns -> output``.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for N-dimensional convolution.
    """
    input_str = ""
    output_str = ""
    pattern_strs: List[str] = []

    # requires 2 + 3 * N letters
    letters = get_letters(2 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    input_str += batch_letter
    output_str += batch_letter

    # input channel dimension
    in_channel_letter = letters.pop()
    input_str += in_channel_letter
    output_str += in_channel_letter

    # coupling of input and index pattern
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        input_str += input_letter
        output_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    for n in range(N):
        output_str += pattern_strs[n][1]

    input_equation = ",".join([input_str] + pattern_strs)

    return "->".join([input_equation, output_str])
