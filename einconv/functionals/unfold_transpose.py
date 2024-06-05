"""Implements a `torch.nn.functional` for input unfolding of a transpose convolution."""

from typing import Tuple, Union

from torch import Tensor, einsum

from einconv.expressions import conv_transposeNd_unfold


def unfoldNd_transpose(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    output_padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    simplify: bool = True,
) -> Tensor:
    """Torch functional for N-dimensional input unfolding of a transpose convolution.

    Extracts elements that overlap with the transpose convolution's kernel at a time
    into a matrix. This matrix can then be used to formulate transpose convolution as
    matrix multiplication between the unfolded input and the matricized kernel.

    This function uses `einsum` under the hood and does not have a PyTorch equivalent.

    We will use the hyper-parameters of an `N`d convolution which maps an input of shape
    `[batch_size, in_channels, *input_sizes]` to an output of shape
    `[batch_size, out_channels, *output_sizes]`. The transpose convolution's input has
    shape `[batch_size, out_channels, *output_sizes]` and the output has shape
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
        simplify: Whether to simplify the einsum equation before evaluating it.
            Default: `True`.

    Returns:
        Unfolded input tensor of shape \
        shape `[batch_size, in_channels * tot_kernel_size, tot_input_size]` where \
        `tot_kernel_size`, `tot_input_size` are the total number of kernel elements and
        spatial input elements to the associated convolution. In `einops` notation, the
        index structure is `n (c_out k1 k2 ...) (i1 i2 ...)`.
    """
    equation, operands, shape = conv_transposeNd_unfold.einsum_expression(
        x,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        simplify=simplify,
    )
    return einsum(equation, *operands).reshape(shape)
