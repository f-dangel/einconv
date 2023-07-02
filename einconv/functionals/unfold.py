"""Equivalent of ``torch.functional.unfold`` for arbitrary dimensions."""
from typing import Tuple, Union

from torch import Tensor, einsum

from einconv.expressions import convNd_unfold


def unfoldNd(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
) -> Tensor:
    """Torch functional for N-dimensional input unfolding that uses einsum.

    Extracts sliding local blocks from a batched input tensor (``im2col``).

    Accepts batched tensors with ``N`` spatial dimensions. Acts like
    ``torch.nn.functional.unfold`` for a 4d input (batched images), but works for
    arbitrary ``N``. See https://pytorch.org/docs/stable/nn.functional.html#unfold.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.

    Returns:
        Unfolded input. Has shape \
        ``[batch_size, in_channels * tot_kernel_size, tot_output_size]`` where \
        ``tot_kernel_size`` is the kernel dimension product and \
        ``tot_output_size`` is the product of the output spatial dimensions. In \
        ``einops`` notation, the index structure is \
        ``n (c_in k1 k2 ...) (o1 o2 ...)``.
    """
    equation, operands, shape = convNd_unfold.einsum_expression(
        x, kernel_size, dilation, padding, stride
    )
    return einsum(equation, *operands).reshape(shape)
