from typing import List, Tuple, Union

from torch import Tensor

import einconv
from einconv.expressions.utils import create_conv_index_patterns, translate_to_torch
from einconv.utils import _tuple


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[str, int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    simplify: bool = True,
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
        simplify: Whether to simplify the einsum expression. Default: ``True``.

    Returns:
        Einsum equation
        Einsum operands in order input, patterns
        Output shape: ``[batch_size, in_channels, tot_output_size]``
    """
    N = x.dim() - 2

    # construct einsum equation
    x_str = "n c_in " + " ".join([f"i{i}" for i in range(N)])
    pattern_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    lhs = ",".join([x_str, *pattern_strs])

    rhs = (
        "n c_in "
        + " ".join([f"k{i}" for i in range(N)])
        + " "
        + " ".join([f"o{i}" for i in range(N)])
    )

    equation = "->".join([lhs, rhs])
    equation = translate_to_torch(equation)

    # construct einsum operands
    patterns = create_conv_index_patterns(
        N,
        input_size=x.shape[2:],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        device=x.device,
        dtype=x.dtype,
    )
    operands = [x, *patterns]

    # construct output shape
    output_tot_size = int(Tensor([p.shape[1] for p in patterns]).int().prod())
    t_kernel_size = _tuple(kernel_size, N)
    kernel_tot_size = int(Tensor(t_kernel_size).int().prod())
    batch_size, in_channels = x.shape[:2]
    shape = (batch_size, in_channels * kernel_tot_size, output_tot_size)

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
