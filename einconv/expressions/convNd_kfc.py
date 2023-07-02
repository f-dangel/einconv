"""Input-based factor of the KFC Fisher approximation for convolutions.

KFC was introduced by:

- Grosse, R., & Martens, J. (2016). A Kronecker-factored approximate Fisher matrix
  for convolution layers. International Conference on Machine Learning (ICML).
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
    """Generate einsum expression of input-based KFC factor for convolution.

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
        groups: In how many groups to split the input channels. Default: ``1``.

    Returns:
        Einsum equation
        Einsum operands in order un-grouped input, patterns, un-grouped input, \
        patterns, normalization scaling
        Output shape: ``[groups, in_channels //groups * tot_kernel_sizes,\
        in_channels //groups * tot_kernel_sizes]``
    """
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
    """Generate einsum operands for KFC factor.

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
        groups: In how many groups to split the input channels. Default: ``1``.

    Returns:
        Tensor list containing the operands. Convention: Un-grouped input, patterns, \
        un-grouped input, patterns, normalization scaling.
        Output shape
    """
    # convert into tuple format
    N = x.dim() - 2
    input_sizes = x.shape[2:]
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
    batch_size = x.shape[0]
    scale = Tensor([1.0 / batch_size]).to(x.device).to(x.dtype)
    operands = [x_ungrouped, *patterns, *patterns, x_ungrouped, scale]

    kernel_tot_sizes = int(Tensor(t_kernel_size).int().prod())
    in_channels = x.shape[1]
    shape = (
        groups,
        in_channels // groups * kernel_tot_sizes,
        in_channels // groups * kernel_tot_sizes,
    )

    return operands, shape


def _equation(N: int) -> str:
    """Generate einsum equation for KFC factor.

    The arguments are ``x, *patterns, *patterns, x, scale -> output``.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for KFC factor of N-dimensional convolution.
    """
    x1_str = ""
    pattern1_strs: List[str] = []
    x2_str = ""
    pattern2_strs: List[str] = []
    output_str = ""

    # requires 5 + 5 * N letters
    letters = get_letters(5 + 5 * N)

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
    for n in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = pattern1_strs[n][1]

        x2_str += input_letter
        output_str += kernel_letter
        pattern2_strs.append(kernel_letter + output_letter + input_letter)

    scale_str = letters.pop()

    input_equation = ",".join(
        [x1_str] + pattern1_strs + pattern2_strs + [x2_str, scale_str]
    )

    return "->".join([input_equation, output_str])
