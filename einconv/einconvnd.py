"""PyTorch modules and functionals implementing N-dimensional convolution."""

from typing import Tuple, Union

from torch import Tensor, einsum
from torch.nn.functional import pad
from torch.nn.modules.utils import _single

from einconv.index_pattern import conv_index_pattern


def einconv1d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, str, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
):
    """Equivalent of ``torch.nn.functionals.conv1d``, but uses tensor contractions.

    Args:
        See documentation of ``torch.nn.functional.conv1d``. # noqa: DAR101

    Returns:
        Result of the convolution.

    Raises:
        NotImplementedError: If the supplied hyperparameters are not supported.
    """
    if isinstance(padding, str):
        raise NotImplementedError("String-valued padding not yet supported.")

    # convert into tuple format
    t_stride: Tuple[int] = _single(stride)
    t_padding: Tuple[int] = _single(padding)
    t_dilation: Tuple[int] = _single(dilation)

    if padding != (0,):
        input = pad(input, 2 * t_padding)

    batch_size, in_channels, input_size = input.shape
    out_channels, _, kernel_size = weight.shape

    index_pattern = conv_index_pattern(
        input_size,
        kernel_size,
        stride=t_stride[0],
        dilation=t_dilation[0],
        device=input.device,
    )

    output = einsum(
        "ngix,kyx,goik->ngoy",
        input.reshape(batch_size, groups, in_channels // groups, input_size),
        index_pattern.to(input.dtype),
        weight.reshape(
            groups, out_channels // groups, in_channels // groups, kernel_size
        ),
    )
    output = output.flatten(start_dim=1, end_dim=2)

    if bias is not None:
        output += bias.unsqueeze(0).unsqueeze(-1).expand_as(output)

    return output
