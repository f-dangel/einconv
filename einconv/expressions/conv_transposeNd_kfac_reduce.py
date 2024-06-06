"""Input-based factor of the KFAC-reduce approximation for transpose convolutions.

KFAC-reduce was introduced by:

- [Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., & Hennig, P.
  (2023). Kronecker-factored approximate curvature for modern neural network
  architectures. In Advances in Neural Information Processing Systems (NeurIPS)]\
(https://arxiv.org/abs/2311.00636).
"""

from typing import List, Optional, Tuple, Union

from einops import rearrange
from torch import Tensor

import einconv
from einconv.expressions.utils import create_conv_index_patterns, translate_to_torch
from einconv.utils import _tuple, get_conv_input_size


def einsum_expression(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    output_padding: Union[int, Tuple[int, ...]] = 0,
    output_size: Optional[Union[int, Tuple[int, ...]]] = None,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    simplify: bool = True,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expr. of input-based KFAC-reduce factor for transp. convolution.

    We describe the `N`d transpose convolution using its associated `N`d convolution
    which maps an input of shape `[batch_size, in_channels, *input_sizes]` to an output
    of shape `[batch_size, out_channels, *output_sizes]`. The transpose convolution's
    input has shape `[batch_size, out_channels, *output_sizes]` and the output has shape
    `[batch_size, in_channels, *input_sizes]`.

    Let $\\mathbf{X}\\in\\mathbb{R}^{C_\\text{out}\\times O_1\\times O_2\\times\\dots}$
    denote the input of a transpose convolution. The unfolded input $[[\\mathbf{X}
    ]]_\\top$ has dimension $(C_\\text{out} \\cdot K_1 \\cdot K_2 \\cdots) \\times
    (I_1 \\cdot I_2 \\cdots)$ where $K_i$ and $I_i$ are the kernel and input sizes of
    the associated convolution. The input-based KFAC-reduce factor is the batch-averaged
    outer product of the column-averaged unfolded input,

    $$
    \\hat{\\mathbf{\\Omega}} =
    \\frac{1}{B \\cdot (I_1 \\cdot I_2 \\cdots)^2} \\sum_{b=1}^B
    ( [[\\mathbf{X}_b]]^\\top_\\top \\mathbf{1} )
    ( [[\\mathbf{X}_b]]^\\top_\\top \\mathbf{1} )^\\top
    \\in \\mathbb{R}^{(C_\\text{out} \\cdot K_1 \\cdot K_2 \\cdots) \\times
    (C_\\text{out} \\cdot K_1 \\cdot K_2 \\cdots)}
    \\,,
    $$

    where $B$ is the batch size and $\\mathbf{X}_b$ is the transpose convolution's
    input from the $b$th data point.

    Args:
        x: Input tensor of shape `[batch_size, out_channels, *output_sizes]`.
        kernel_size: Size of the convolutional kernel. Can be a single integer (shared
            along all spatial dimensions), or an `N`-tuple of integers.
        stride: Stride of the associated convolution. Can be a single integer (shared
            along all spatial dimensions), or an `N`-tuple of integers. Default: `1`.
        padding: Padding of the associated convolution. Can be a single integer (shared
            along all spatial dimensions), or an `N`-tuple of integers. Default: `0`.
        output_padding: Number of unused pixels at the end of the spatial domain.
            This is used to resolve the ambiguity that a convolution can map different
            input sizes to the same output size if its stride is different from 1.
            Instead of specifying this argument, you can directly specify the output
            size of the transpose convolution (i.e. the input size of the associated
            convolution via the `output_size` argument). Can be a single integer
            (shared along all spatial dimensions), or an `N`-tuple. Default: `0`.
        output_size: Size of the output of the transpose convolution (i.e. the input
            size of the associated convolution). Specifying this argument will override
            the `output_padding` argument. Can be a single integer (shared along all
            spatial dimensions), or an `N`-tuple of integers. Default: `None`.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an `N`-tuple of integers. Default: `1`.
        groups: In how many groups to split the channels. Default: `1`.
        simplify: Whether to simplify the einsum expression. Default: `True`.

    Returns:
        Einsum equation
        Einsum operands in order un-grouped input, patterns, un-grouped input, \
        patterns, normalization scaling
        Output shape: `[groups, out_channels //groups * tot_kernel_sizes,\
        out_channels //groups * tot_kernel_sizes]`
    """
    N = x.dim() - 2

    # construct einsum equation
    x1_str = "n g c_out " + " ".join([f"o{i}" for i in range(N)])
    x2_str = "n g c_out_ " + " ".join([f"o{i}_" for i in range(N)])
    pattern1_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    pattern2_strs: List[str] = [f"k{i}_ o{i}_ i{i}_" for i in range(N)]
    scale_str = "s"
    lhs = ",".join([x1_str, *pattern1_strs, *pattern2_strs, x2_str, scale_str])
    rhs = (
        "g c_out "
        + " ".join([f"k{i}" for i in range(N)])
        + " c_out_ "
        + " ".join([f"k{i}_" for i in range(N)])
    )
    equation = "->".join([lhs, rhs])
    equation = translate_to_torch(equation)

    conv_output_size = x.shape[2:]
    t_kernel_size = _tuple(kernel_size, N)
    t_stride = _tuple(stride, N)
    t_padding = _tuple(padding, N)
    t_dilation = _tuple(dilation, N)

    # infer output_padding from convolution's input size
    if output_size is not None:
        t_output_size = _tuple(output_size, N)
        t_output_padding = tuple(
            output_size - get_conv_input_size(conv_out_size, K, S, P, 0, D)
            for output_size, conv_out_size, K, S, P, D in zip(
                t_output_size,
                conv_output_size,
                t_kernel_size,
                t_stride,
                t_padding,
                t_dilation,
            )
        )
    else:
        t_output_padding = _tuple(output_padding, N)

    conv_input_size = tuple(
        get_conv_input_size(output_size, K, S, P, output_padding, D)
        for output_size, K, S, P, output_padding, D in zip(
            conv_output_size,
            t_kernel_size,
            t_stride,
            t_padding,
            t_output_padding,
            t_dilation,
        )
    )

    # construct einsum operands
    patterns = create_conv_index_patterns(
        N,
        input_size=conv_input_size,
        kernel_size=t_kernel_size,
        stride=t_stride,
        padding=t_padding,
        dilation=dilation,
        device=x.device,
        dtype=x.dtype,
    )
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    conv_input_tot_size = Tensor(conv_input_size).int().prod()
    batch_size, out_channels = x.shape[:2]
    scale = Tensor([1.0 / (batch_size * conv_input_tot_size**2)]).to(x.device, x.dtype)
    operands = [x_ungrouped, *patterns, *patterns, x_ungrouped, scale]

    # construct output shape
    t_kernel_size = _tuple(kernel_size, N)
    kernel_tot_size = int(Tensor(t_kernel_size).int().prod())
    shape = (
        groups,
        out_channels // groups * kernel_tot_size,
        out_channels // groups * kernel_tot_size,
    )

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
