"""Implementation of the input-based factor of K-FAC reduce for convolutions.

Details see:

- Eschenhagen, R. (2022). Kronecker-factored approximate curvature for linear
  weight-sharing layers.
"""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor

from einconv import index_pattern
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    return equation, operands, shape


def _operands_and_shape(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[List[Tensor], Tuple[int, ...]]:
    N = x.dim() - 2
    (batch_size, in_channels), input_sizes = x.shape[:2], x.shape[2:]

    # convert into tuple format
    t_kernel_size: Tuple[int, ...] = _tuple(kernel_size, N)
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)
    t_padding: Union[Tuple[int, ...], str] = (
        padding if isinstance(padding, str) else _tuple(padding, N)
    )
    t_stride: Tuple[int, ...] = _tuple(stride, N)

    patterns: List[Tensor] = [
        index_pattern(
            input_sizes[n],
            t_kernel_size[n],
            stride=t_stride[n],
            padding=t_padding if isinstance(t_padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=x.device,
            dtype=x.dtype,
        )
        for n in range(N)
    ]
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)

    output_tot_size = Tensor([p.shape[1] for p in patterns]).int().prod()
    kernel_tot_size = Tensor(t_kernel_size).int().prod()
    scale = Tensor([1.0 / (batch_size * output_tot_size**2)]).to(x.device).to(x.dtype)

    output_shape = (
        groups,
        in_channels // groups * kernel_tot_size,
        in_channels // groups * kernel_tot_size,
    )

    return [x_ungrouped, *patterns, *patterns, x_ungrouped, scale], output_shape


def _equation(N: int) -> str:
    """Generate einsum equation for KFAC reduce factor.

    The arguments are ``input, *index_patterns, *index_patterns, input -> output``.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for KFAC reduce factor of N-dimensional convolution.
    """
    x1_str = ""
    pattern1_strs: List[str] = []
    x2_str = ""
    pattern2_strs: List[str] = []
    output_str = ""

    # requires 5 + 6 * N letters
    letters = get_letters(5 + 6 * N)

    # batch dimension
    batch_letter = letters.pop()
    x1_str += batch_letter
    x2_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    x1_str += group_letter
    x2_str += group_letter
    output_str += group_letter

    # input channel dimension for first input
    in_channel_letter = letters.pop()
    x1_str += in_channel_letter
    output_str += in_channel_letter

    # coupling of first input and index pattern
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        x1_str += input_letter
        output_str += kernel_letter
        pattern1_strs.append(kernel_letter + output_letter + input_letter)

    # input channel dimension for second input
    in_channel_letter = letters.pop()
    x2_str += in_channel_letter
    output_str += in_channel_letter

    # coupling of second input and index pattern
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        x2_str += input_letter
        output_str += kernel_letter
        pattern2_strs.append(kernel_letter + output_letter + input_letter)

    scale_str = letters.pop()

    input_equation = ",".join(
        [x1_str] + pattern1_strs + pattern2_strs + [x2_str, scale_str]
    )

    return "->".join([input_equation, output_str])
