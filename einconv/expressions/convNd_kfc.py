"""Input-based factor of the KFC Fisher approximation for convolutions.

KFC was introduced by:

- [Grosse, R., & Martens, J. (2016). A Kronecker-factored approximate Fisher matrix
  for convolution layers. International Conference on Machine Learning (ICML).]\
(https://arxiv.org/abs/1602.01407)
"""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor

import einconv
from einconv.expressions.utils import create_conv_index_patterns, translate_to_torch
from einconv.utils import _tuple


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    simplify: bool = True,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expression of input-based KFC factor for convolution.

    Let $\\mathbf{X}\\in\\mathbb{R}^{C_\\text{in}\\times I_1\\times I_2\\times\\dots}$
    denote the input of a convolution. The unfolded input $[[\\mathbf{X}]]$
    has dimension $(C_\\text{in} \\cdot K_1 \\cdot K_2 \\cdots) \\times (O_1 \\cdot O_2
    \\cdots)$ where $K_i$ and $O_i$ are the kernel and output sizes of the convolution.
    The input-based KFC factor is the batch-averaged outer product of the unfolded
    input,

    $$
    \\mathbf{\\Omega} =
    \\frac{1}{B} \\sum_{b=1}^B
    [[\\mathbf{X}_b]] [[\\mathbf{X}_b]]^\\top
    \\in \\mathbb{R}^{(C_\\text{in} \\cdot K_1 \\cdot K_2 \\cdots) \\times
    (C_\\text{in} \\cdot K_1 \\cdot K_2 \\cdots)}
    \\,,
    $$

    where $B$ is the batch size and $\\mathbf{X}_b$ is the convolution's input from the
    $b$th data point.

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
        simplify: Whether to simplify the einsum expression. Default: ``True``.

    Returns:
        Einsum equation
        Einsum operands in order un-grouped input, patterns, un-grouped input, \
        patterns, normalization scaling
        Output shape: ``[groups, in_channels //groups * tot_kernel_sizes,\
        in_channels //groups * tot_kernel_sizes]``
    """
    N = x.dim() - 2

    # construct einsum equation
    x1_str = "n g c_in " + " ".join([f"i{i}" for i in range(N)])
    x2_str = "n g c_in_ " + " ".join([f"i{i}_" for i in range(N)])
    pattern1_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    pattern2_strs: List[str] = [f"k{i}_ o{i} i{i}_" for i in range(N)]
    scale_str = "s"
    lhs = ",".join([x1_str, *pattern1_strs, *pattern2_strs, x2_str, scale_str])

    rhs = (
        "g c_in "
        + " ".join([f"k{i}" for i in range(N)])
        + " c_in_ "
        + " ".join([f"k{i}_" for i in range(N)])
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
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    batch_size = x.shape[0]
    scale = Tensor([1.0 / batch_size]).to(x.device).to(x.dtype)
    operands = [x_ungrouped, *patterns, *patterns, x_ungrouped, scale]

    # construct output shape
    t_kernel_size = _tuple(kernel_size, N)
    kernel_tot_sizes = int(Tensor(t_kernel_size).int().prod())
    in_channels = x.shape[1]
    shape = (
        groups,
        in_channels // groups * kernel_tot_sizes,
        in_channels // groups * kernel_tot_sizes,
    )

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
