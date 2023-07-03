"""Utility functions for creating einsum expressions."""

from typing import List, Tuple, Union

import torch
from torch import Tensor

from einconv import index_pattern
from einconv.utils import _tuple, cpu


def create_conv_index_patterns(
    N: int,
    input_size: Union[int, Tuple[int, ...]],
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = cpu,
    dtype: torch.dtype = torch.bool,
) -> List[Tensor]:
    """Create the index pattern tensors for all dimensions of a convolution.

    Args:
        N: Convolution dimension.
        input_size: Spatial dimensions of the convolution. Can be a single integer
            (shared along all spatial dimensions), or an ``N``-tuple of integers.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        device: Device to create the tensors on. Default: ``'cpu'``.
        dtype: Data type of the pattern tensor. Default: ``torch.bool``.

    Returns:
        List of index pattern tensors for dimensions ``1, ..., N``.
    """
    # convert into tuple format
    t_input_size = _tuple(input_size, N)
    t_kernel_size = _tuple(kernel_size, N)
    t_stride: Tuple[int, ...] = _tuple(stride, N)
    t_padding = padding if isinstance(padding, str) else _tuple(padding, N)
    t_dilation = _tuple(dilation, N)

    return [
        index_pattern(
            t_input_size[n],
            t_kernel_size[n],
            stride=t_stride[n],
            padding=t_padding if isinstance(t_padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=device,
            dtype=dtype,
        )
        for n in range(N)
    ]
