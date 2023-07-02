"""Equivalent of ``torch.functional.unfold`` for arbitrary dimensions."""
from typing import Optional, Tuple, Union

from torch import Tensor, einsum

from einconv.expressions import convNd_unfold


def unfoldNd(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
) -> Tensor:
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    Pytorch functional that accepts N-dimensional inputs. Acts like
    ``torch.nn.functional.unfold`` for a 4d input. Uses tensor networks under the hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html#unfold.


    Returns:
        Unfolded input of shape
        ``[batch_size, in_channels * kernel_size_numel, output.shape[2:].numel()]``
        where output.shape[2:].numel()`` means the product of spatial output dimensions
        of the related convolution.
    """
    equation, operands, shape = convNd_unfold.einsum_expression(
        x, kernel_size, dilation, padding, stride
    )
    return einsum(equation, *operands).reshape(shape)
