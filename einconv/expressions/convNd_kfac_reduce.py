"""Input-based factor of the K-FAC reduce approximation for convolutions.

KFAC-reduce was introduced by:

- Eschenhagen, R. (2022). Kronecker-factored approximate curvature for linear
  weight-sharing layers, Master thesis.
"""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor

from einconv.expressions.utils import create_conv_index_patterns, translate_to_torch
from einconv.utils import _tuple


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expression of input-based KFAC-reduce factor for convolution.

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
    """Generate einsum operands for KFAC-reduce factor.

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
    output_tot_size = Tensor([p.shape[1] for p in patterns]).int().prod()
    batch_size = x.shape[0]
    scale = Tensor([1.0 / (batch_size * output_tot_size**2)]).to(x.device).to(x.dtype)
    operands = [x_ungrouped, *patterns, *patterns, x_ungrouped, scale]

    t_kernel_size = _tuple(kernel_size, N)
    kernel_tot_size = int(Tensor(t_kernel_size).int().prod())
    in_channels = x.shape[1]
    shape = (
        groups,
        in_channels // groups * kernel_tot_size,
        in_channels // groups * kernel_tot_size,
    )

    return operands, shape


def _equation(N: int) -> str:
    """Generate einsum equation for KFAC reduce factor.

    The arguments are
    ``input, *index_patterns, *index_patterns, input, scale -> output``.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for KFAC reduce factor of N-dimensional convolution.
    """
    x1_str = "n g c_in " + " ".join([f"i{i}" for i in range(N)])
    x2_str = "n g c_in_ " + " ".join([f"i{i}_" for i in range(N)])
    pattern1_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    pattern2_strs: List[str] = [f"k{i}_ o{i}_ i{i}_" for i in range(N)]
    scale_str = "s"
    lhs = ",".join([x1_str, *pattern1_strs, *pattern2_strs, x2_str, scale_str])

    rhs = (
        "g c_in "
        + " ".join([f"k{i}" for i in range(N)])
        + " c_in_ "
        + " ".join([f"k{i}_" for i in range(N)])
    )

    equation = "->".join([lhs, rhs])
    return translate_to_torch(equation)
