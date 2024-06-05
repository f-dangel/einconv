"""Utility functions for tests comparing against JAX's N-dimensional convolution."""

from typing import Callable, Tuple, Union

import jax
import numpy
import torch

from einconv.modules import ConvNd
from einconv.utils import _tuple


def jax_convNd(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """JAX implementation of ``nn.functional.convNd``.

    ``N`` is determined from the input tensor: It's first axis is the batch dimension,
    the second axis the channel dimension, and the remaining number of dimensions is
    interpreted as spatial dimension (with number of spatial dimensions ``N``)

    Args:
        input: Input of the convolution. Has shape ``[batch_size,
            in_channels, *]`` where ``*`` can be an arbitrary shape. The
            convolution dimension is ``len(*)``.
        weight: Kernel of the convolution. Has shape ``[out_channels,
            in_channels / groups, *]`` where ``*`` contains the kernel sizes and has
            length ``N``.
        bias: Optional bias vector of the convolution. Has shape ``[out_channels]``.
            Default: ``None``.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along all
            spatial dimensions), an ``N``-tuple of integers, or a string. Allowed
            strings are ``'same'`` and ``'valid'``. Default: ``0``.
        dilation: Dilation of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        groups: How to split the input into groups. Default: ``1``.

    Returns:
        Result of the convolution. Has shape ``[batch_size, out_channels, *]`` where
        ``*`` is the the spatial output dimension shape.
    """
    N = input.dim() - 2

    # convert torch Tensors to JAX arrays
    input_jax = jax.numpy.array(input.detach().cpu().numpy())
    weight_jax = jax.numpy.array(weight.detach().cpu().numpy())

    # convert hyperparameters
    stride = _tuple(stride, N)
    if isinstance(padding, str):
        padding = padding.upper()
    else:
        padding = tuple(
            (p, p) for p in _tuple(padding, N)
        )  # padding is ((p, p), (q, q), ...)
    dilation = _tuple(dilation, N)

    output_jax = jax.lax.conv_general_dilated(
        input_jax,
        weight_jax,
        stride,
        padding,
        rhs_dilation=dilation,
        feature_group_count=groups,  # see
        # https://www.tensorflow.org/xla/operation_semantics
    )

    # convert JAX array to torch Tensor
    output = torch.from_numpy(numpy.array(output_jax)).to(input.device).to(input.dtype)

    if bias is not None:
        bias_expanded = bias.unsqueeze(0)
        for _ in range(N):
            bias_expanded = bias_expanded.unsqueeze(-1)
        output += bias_expanded.expand_as(output)

    return output


def to_ConvNd_jax(einconv_module: ConvNd) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create JAX convolution 'layer' (callable) for high-dimensional convolutions.

    Args:
        einconv_module: Einconv layer.

    Returns:
        Convolution layer for high-dimensional convolutions. Uses JAX under the hood.

    Raises:
        NotImplementedError: For unsupported padding modes.
    """
    if einconv_module.padding_mode != "zeros":
        raise NotImplementedError("Only padding_mode='zeros' supported.")

    def ConvNd_jax(input: torch.Tensor) -> torch.Tensor:
        return jax_convNd(
            input,
            einconv_module.weight.data.clone(),
            bias=(
                None
                if einconv_module.bias is None
                else einconv_module.bias.data.clone()
            ),
            stride=einconv_module.stride,
            padding=einconv_module.padding,
            dilation=einconv_module.dilation,
            groups=einconv_module.groups,
        )

    return ConvNd_jax
