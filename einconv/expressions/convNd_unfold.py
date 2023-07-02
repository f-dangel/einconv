from typing import List, Optional, Tuple, Union

from torch import Tensor

from einconv import index_pattern
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
):
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x, kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    return equation, operands, shape


def _operands_and_shape(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
    padding: Optional[Union[int, Tuple[int, ...], str]] = 0,
    stride: Optional[Union[int, Tuple[int, ...]]] = 1,
) -> Tuple[List[Tensor], Tuple[int]]:
    N = x.dim() - 2
    (batch_size, in_channels), input_sizes = x.shape[:2], x.shape[2:]

    # convert into tuple format
    t_kernel_size: Tuple[int, ...] = _tuple(kernel_size, N)
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)
    t_padding: Union[Tuple[int, ...], str] = _tuple(padding, N)
    t_stride: Tuple[int, ...] = _tuple(stride, N)

    patterns: List[Tensor] = [
        index_pattern(
            input_sizes[n],
            t_kernel_size[n],
            stride=t_stride[n],
            padding=t_padding[n],
            dilation=t_dilation[n],
            device=x.device,
            dtype=x.dtype,
        )
        for n in range(N)
    ]

    output_tot_size = Tensor([p.shape[1] for p in patterns]).int().prod()
    kernel_tot_size = Tensor(t_kernel_size).int().prod()
    output_shape = (batch_size, in_channels * kernel_tot_size, output_tot_size)

    return [x] + patterns, output_shape


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
