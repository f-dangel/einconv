from typing import List, Tuple, Union

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
    dilation: Union[int, Tuple[int, ...]] = 1,
    simplify: bool = True,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expression to unfold the input of a transpose convolution.

    The unfolded input for a transpose convolution flattens and concatenates all
    elements of the input tensor into a matrix such that the transpose convolution
    can be written as matrix multiplication between the unfolded input and the
    matricized kernel.

    We will use the associated convolution's hyper-parameters to describe all arguments.
    Consider an `N`d convolution which maps an input tensor of shape
    `[batch_size, in_channels, *input_sizes]` to an output tensor of shape
    `[batch_size, out_channels, *output_sizes]`. The transpose convolution's input
    has shape `[batch_size, out_channels, *output_sizes]` and the output has shape
    `[batch_size, in_channels, *input_sizes]`.

    Args:
        x: Input to the `N`d transpose convolution. Has shape
            `[batch_size, in_channels, *input_sizes]` where `len(input_sizes) == N`.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an `N`-tuple of integers.
        stride: Stride of the associated convolution. Can be a single integer (shared
            along all spatial dimensions), or an `N`-tuple of integers. Default: `1`.
        padding: Padding of the associated convolution. Can be a single integer (shared
            along all spatial dimensions)  or an `N`-tuple of integers, Default: `0`.
        output_padding: The associated convolution's number of unused pixels at the end
            of a spatial dimension. This is required to resolve the ambiguity that a
            convolution can produce the same output shape for different input shapes if
            it has non-unit stride. Can be a single integer (shared along all spatial
            dimensions), or an `N`-tuple of integers. Default: `0`.
        dilation: Dilation of the associated convolution. Can be a single integer
            (shared along all spatial dimensions), or an `N`-tuple of integers.
            Default: `1`.
        simplify: Whether to simplify the einsum expression. Default: `True`.

    Returns:
        Einsum equation
        Einsum operands in order input, patterns
        Output shape: `[batch_size, out_channels * tot_kernel_size, tot_input_size]`
    """
    N = x.dim() - 2

    # construct einsum equation
    x_str = "n c_out " + " ".join([f"o{i}" for i in range(N)])
    pattern_strs: List[str] = [f"k{i} o{i} i{i}" for i in range(N)]
    lhs = ",".join([x_str, *pattern_strs])

    rhs = (
        "n c_out "
        + " ".join([f"k{i}" for i in range(N)])
        + " "
        + " ".join([f"i{i}" for i in range(N)])
    )

    equation = "->".join([lhs, rhs])
    equation = translate_to_torch(equation)

    # compute input sizes
    t_output_size = x.shape[2:]
    t_stride = _tuple(stride, N)
    t_kernel_size = _tuple(kernel_size, N)
    t_padding = _tuple(padding, N)
    t_dilation = _tuple(dilation, N)
    t_output_padding = _tuple(output_padding, N)
    t_input_size = tuple(
        get_conv_input_size(
            output_size, kernel_size, stride, padding, output_padding, dilation
        )
        for output_size, kernel_size, stride, padding, output_padding, dilation in zip(
            t_output_size,
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
        input_size=t_input_size,
        kernel_size=t_kernel_size,
        stride=t_stride,
        padding=t_padding,
        dilation=t_dilation,
        device=x.device,
        dtype=x.dtype,
    )
    operands = [x, *patterns]

    # construct shape
    input_tot_size = int(Tensor(t_input_size).int().prod())
    kernel_tot_size = int(Tensor(t_kernel_size).int().prod())
    batch_size, out_channels = x.shape[:2]
    shape = (batch_size, out_channels * kernel_tot_size, input_tot_size)

    print(equation)
    print([op.shape for op in operands])

    if simplify:
        equation, operands = einconv.simplify(equation, operands)

    return equation, operands, shape
