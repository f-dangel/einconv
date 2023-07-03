"""PyTorch equivalent of ``nn.functional.conv{1,2,3}d`` implemented as einsum."""

from typing import Tuple, Union

from torch import Tensor, einsum
from torch.nn import Parameter

from einconv.expressions import convNd_forward


def convNd(
    x: Tensor,
    weight: Union[Tensor, Parameter],
    bias: Union[Tensor, Parameter, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    simplify: bool = True,
) -> Tensor:
    """Generalization of ``torch.nn.functional.conv{1,2,3}d`` to ``N``d.

    ``N`` is determined from the input tensor: It's first axis is the batch dimension,
    the second axis the channel dimension, and the remaining number of dimensions is
    interpreted as spatial dimension (with number of spatial dimensions ``N``)

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        weight: Kernel of the convolution. Has shape ``[out_channels,
            in_channels / groups, *kernel_size]`` where ``kernel_size`` is an
            ``N``-tuple of kernel dimensions.
        bias: Optional bias vector of the convolution. Has shape ``[out_channels]``.
            Default: ``None``.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        groups: In how many groups to split the input channels. Default: ``1``.
        simplify: Whether to use a simplified einsum expression. Default: ``True``.

    Returns:
        Result of the convolution. Has shape \
        ``[batch_size, out_channels, *output_sizes]`` where \
        ``len(output_sizes) == N``. In ``einops`` notation, the index structure \
        is ``n (g c_out) o1 o2 ...``.
    """
    _check_args(x, weight, bias=bias, groups=groups)
    equation, operands, shape = convNd_forward.einsum_expression(
        x,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        simplify=simplify,
    )
    output = einsum(equation, *operands).reshape(shape)

    if bias is not None:
        N = x.dim() - 2
        out_channels = weight.shape[0]
        shape_before_expand = (1, out_channels) + N * (1,)
        output += bias.reshape(*shape_before_expand).expand_as(output)

    return output


def _check_args(
    x: Tensor,
    weight: Tensor,
    bias: Union[Tensor, Parameter, None] = None,
    groups: int = 1,
):
    """Check the input arguments to ``convNd``.

    Args:
        x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
            where ``len(input_sizes) == N``.
        weight: Kernel of the convolution. Has shape ``[out_channels,
            in_channels / groups, *kernel_size]`` where ``kernel_size`` is an
            ``N``-tuple of kernel dimensions.
        bias: Optional bias vector of the convolution. Has shape ``[out_channels]``.
            Default: ``None``.
        groups: In how many groups to split the input channels. Default: ``1``.

    Raises:
        ValueError: If bias has incorrect shape.
        ValueError: If weight dimension is incorrect.
        ValueError: If weight shape is invalid.
    """
    N = x.dim() - 2
    in_channels = x.shape[1]
    out_channels = weight.shape[0]

    if weight.dim() != N + 2:
        raise ValueError(
            f"For Conv(N={N})d, the kernel must be {N+2}d. Got {weight.dim()}d."
        )

    if weight.shape[0] % groups != 0:
        raise ValueError(
            f"Groups ({groups}) must divide out_channels ({weight.shape[0]})."
        )

    if weight.shape[1] * groups != in_channels:
        raise ValueError(
            f"Kernel dimension 1 ({weight.shape[1]}) multiplied by groups ({groups})"
            + f" must equal in_channels ({in_channels})."
        )

    if bias is not None and (bias.dim() != 1 or bias.numel() != out_channels):
        raise ValueError(f"Bias should have shape [{out_channels}]. Got {bias.shape}.")
