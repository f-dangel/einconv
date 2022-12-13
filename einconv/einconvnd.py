"""PyTorch modules and functionals implementing N-dimensional convolution."""

from typing import Union

from torch import Tensor, einsum

from einconv.index_pattern import conv_index_pattern


def einconv1d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """Equivalent of ``torch.nn.functionals.conv1d``, but uses tensor contractions.

    Args:
        See documentation of ``torch.nn.functional.conv1d``. # noqa: DAR101

    Returns:
        Result of the convolution.

    Raises:
        ValueError: If the input is not a 3d tensor, or the weight is not a 3d tensor.
        NotImplementedError: If the supplied hyperparameters are not supported.
    """
    if bias is not None:
        raise NotImplementedError
    if groups != 1:
        raise NotImplementedError
    if isinstance(padding, str):
        raise NotImplementedError
    if padding != 0:
        raise NotImplementedError

    if input.dim() != 3:
        raise ValueError
    if weight.dim() != 3:
        raise ValueError

    input_size = input.shape[2]
    kernel_size = weight.shape[2]

    index_pattern = conv_index_pattern(
        input_size, kernel_size, stride=stride, dilation=dilation, device=input.device
    ).to(input.dtype)

    return einsum("nix,kyx,oik->noy", input, index_pattern, weight)
