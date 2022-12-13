"""PyTorch modules and functionals implementing N-dimensional convolution."""

from typing import Tuple, Union

from torch import Tensor, einsum
from torch.nn.functional import pad
from torch.nn.modules.utils import _pair, _single, _triple

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
    """Equivalent of ``torch.nn.functional.conv1d``, but uses tensor contractions.

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


def einconv2d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, str, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
):
    """Equivalent of ``torch.nn.functional.conv2d``, but uses tensor contractions.

    Args:
        See documentation of ``torch.nn.functional.conv2d``. # noqa: DAR101

    Returns:
        Result of the convolution.

    Raises:
        NotImplementedError: If the supplied hyperparameters are not supported.
    """
    if isinstance(padding, str):
        raise NotImplementedError("String-valued padding not yet supported.")

    # convert into tuple format
    t_stride: Tuple[int, int] = _pair(stride)
    t_padding: Tuple[int, int] = _pair(padding)
    t_dilation: Tuple[int, int] = _pair(dilation)

    if padding != (0, 0):
        paddings = []
        for p in t_padding:
            paddings = [p, p] + paddings
        input = pad(input, tuple(paddings))

    batch_size, in_channels, input_size_h, input_size_w = input.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape

    index_pattern_h = conv_index_pattern(
        input_size_h,
        kernel_size_h,
        stride=t_stride[0],
        dilation=t_dilation[0],
        device=input.device,
    )
    index_pattern_w = conv_index_pattern(
        input_size_w,
        kernel_size_w,
        stride=t_stride[1],
        dilation=t_dilation[1],
        device=input.device,
    )

    output = einsum(
        "ngiab,kza,lyb,goikl->ngozy",
        input.reshape(
            batch_size, groups, in_channels // groups, input_size_h, input_size_w
        ),
        index_pattern_h.to(input.dtype),
        index_pattern_w.to(input.dtype),
        weight.reshape(
            groups,
            out_channels // groups,
            in_channels // groups,
            kernel_size_h,
            kernel_size_w,
        ),
    )
    output = output.flatten(start_dim=1, end_dim=2)

    if bias is not None:
        output += bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(output)

    return output


def einconv3d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, str, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
):
    """Equivalent of ``torch.nn.functional.conv3d``, but uses tensor contractions.

    Args:
        See documentation of ``torch.nn.functional.conv3d``. # noqa: DAR101

    Returns:
        Result of the convolution.

    Raises:
        NotImplementedError: If the supplied hyperparameters are not supported.
    """
    if isinstance(padding, str):
        raise NotImplementedError("String-valued padding not yet supported.")

    # convert into tuple format
    t_stride: Tuple[int, int, int] = _triple(stride)
    t_padding: Tuple[int, int, int] = _triple(padding)
    t_dilation: Tuple[int, int, int] = _triple(dilation)

    if padding != (0, 0, 0):
        paddings = []
        for p in t_padding:
            paddings = [p, p] + paddings
        input = pad(input, tuple(paddings))

    batch_size, in_channels, input_size_d, input_size_h, input_size_w = input.shape
    out_channels, _, kernel_size_d, kernel_size_h, kernel_size_w = weight.shape

    index_pattern_d = conv_index_pattern(
        input_size_d,
        kernel_size_d,
        stride=t_stride[0],
        dilation=t_dilation[0],
        device=input.device,
    )
    index_pattern_h = conv_index_pattern(
        input_size_h,
        kernel_size_h,
        stride=t_stride[1],
        dilation=t_dilation[1],
        device=input.device,
    )
    index_pattern_w = conv_index_pattern(
        input_size_w,
        kernel_size_w,
        stride=t_stride[2],
        dilation=t_dilation[2],
        device=input.device,
    )

    output = einsum(
        "ngiabc,kza,lyb,mxc,goiklm->ngozyx",
        input.reshape(
            batch_size,
            groups,
            in_channels // groups,
            input_size_d,
            input_size_h,
            input_size_w,
        ),
        index_pattern_d.to(input.dtype),
        index_pattern_h.to(input.dtype),
        index_pattern_w.to(input.dtype),
        weight.reshape(
            groups,
            out_channels // groups,
            in_channels // groups,
            kernel_size_d,
            kernel_size_h,
            kernel_size_w,
        ),
    )
    output = output.flatten(start_dim=1, end_dim=2)

    if bias is not None:
        output += (
            bias.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(output)
        )

    return output
