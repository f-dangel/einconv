"""Contains functionality to implement convolution as tensor contraction (einsum)."""

from typing import Union

import torch
from torch import Tensor, arange, eye, logical_and, nonzero, ones_like, zeros
from torch.nn.functional import conv1d

from einconv.utils import get_conv_output_size, get_conv_paddings

cpu = torch.device("cpu")


def index_pattern(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    device: torch.device = cpu,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Compute the connectivity pattern tensor of a convolution along one dimension.

    Uses one-dimensional convolution under the hood.

    Args:
        input_size: Spatial input dimension of the convolution.
        kernel_size: Kernel size along dimension.
        stride: Stride along dimension. Default: ``1``.
        padding: Padding along dimension. Can be an integer or a string. Allowed
            strings are ``'same'`` and ``'valid'``. Default: ``0``.
        dilation: Dilation along dimension. Default: ``1``.
        device: Execution device. Default: ``'cpu'``.
        dtype: Data type of the pattern tensor. Default: ``torch.bool``.

    Returns:
        Index pattern tensor. Has shape ``[kernel_size, output_size, input_size]`` and \
        the specified data type. Its element ``[k, o, i]`` is ``True`` (or equivalent \
        cast) if element ``i`` of the input contributes to output element ``o`` via \
        the ``k``th kernel entry (``False`` otherwise). The hyper-parameters are \
        stored under the tensor's ``._pattern_hyperparams`` attribute.
    """
    in_idxs_dtype = torch.int32
    # in some cases, conv1d does not support int32 inputs.
    if dilation != 1 or device != cpu:
        in_idxs_dtype = torch.float32

    in_idxs = (
        arange(
            start=1,  # index 0 is used for elements from padding
            end=input_size + 1,
            dtype=in_idxs_dtype,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # shape [N=1, C_in=1, input_size]
    weight = eye(kernel_size, dtype=in_idxs.dtype, device=device).unsqueeze(
        1
    )  # shape [C_out=kernel_size, C_in=1, K=kernel_size], entries [k, 1, k] = 1 else 0
    out_idxs = (
        conv1d(in_idxs, weight, stride=stride, padding=padding, dilation=dilation)
        .squeeze(0)
        .unsqueeze(-1)
    )  # shape [K, O, 1], entry [k, o, 0] contains index of the input that
    # contributes to the o-th output element via the k-th kernel element

    # scatter True to [k, o, out_idxs[k, o]] ∀ k, o
    output_size = out_idxs.shape[1]
    pattern = zeros(
        kernel_size, output_size, input_size + 1, dtype=torch.bool, device=device
    )
    pattern.scatter_add_(2, out_idxs.long(), ones_like(pattern))
    pattern = pattern.narrow(2, 1, input_size)  # remove the padding bin

    pattern = pattern.to(dtype)
    # store convolution parameters in pattern tensor
    pattern._pattern_hyperparams = {
        "input_size": input_size,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
    }

    return pattern  # shape [kernel_size, output_size, input_size]


def index_pattern_logical(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    device: torch.device = cpu,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Compute the connectivity pattern tensor of a convolution along one dimension.

    Uses logical expressions under the hood. This an alternative to ``index_pattern``
    whose implementation makes the index pattern's structure clearer as it does not
    rely on convolution.

    Args:
        input_size: Spatial input dimension of the convolution.
        kernel_size: Kernel size along dimension.
        stride: Stride along dimension. Default: ``1``.
        padding: Padding along dimension. Can be an integer or a string. Allowed
            strings are ``'same'`` and ``'valid'``. Default: ``0``.
        dilation: Dilation along dimension. Default: ``1``.
        device: Execution device. Default: ``'cpu'``.
        dtype: Data type of the pattern tensor. Default: ``torch.bool``.

    Returns:
        Index pattern tensor. Has shape ``[kernel_size, output_size, input_size]`` and \
        the specified data type. Its element ``[k, o, i]`` is ``True`` (or equivalent \
        cast) if element ``i`` of the input contributes to output element ``o`` via \
        the ``k``th kernel entry (``False`` otherwise). The hyper-parameters are \
        stored under the tensor's ``._pattern_hyperparams`` attribute.
    """
    output_size = get_conv_output_size(
        input_size, kernel_size, stride, padding, dilation
    )
    pattern = zeros(
        kernel_size, output_size, input_size, dtype=torch.bool, device=device
    )

    padding_left, _ = get_conv_paddings(kernel_size, stride, padding, dilation)
    o_idx = torch.arange(output_size, device=device, dtype=torch.long)

    for k in range(kernel_size):
        i_idx = -padding_left + k * dilation + stride * o_idx
        in_bounds = nonzero(logical_and(i_idx >= 0, i_idx < input_size))

        pattern[k, o_idx[in_bounds], i_idx[in_bounds]] = True

    pattern = pattern.to(dtype)
    # store convolution parameters in pattern tensor
    pattern._pattern_hyperparams = {
        "input_size": input_size,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
    }

    return pattern
