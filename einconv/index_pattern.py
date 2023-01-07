"""Contains functionality to implement convolution as tensor contraction (einsum)."""

from typing import Union

import torch
from torch import Tensor, arange, device, eye, ones_like, zeros
from torch.nn.functional import conv1d

cpu = device("cpu")


def conv_index_pattern(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    device: device = cpu,
) -> Tensor:
    """Compute the 'dummy tensor' containing the index pattern of a conv. dimension.

    The dummy tensor is denoted 𝒫 in the paper (see page 3):

    - Hayashi, K., Yamaguchi, T., Sugawara, Y., & Maeda, S. (2019). Exploring
      unexplored tensor network decompositions for convolutional neural networks.
      Advances in Neural Information Processing Systems (NeurIPS).

    Args:
        input_size: Number of pixels along dimension.
        kernel_size: Kernel size along dimension.
        stride: Stride along dimension. Default: ``1``.
        padding: Padding along dimension. Can be an integer or a string. Allowed
            strings are ``'same'`` and ``'valid'``. Default: ``0``.
        dilation: Dilation along dimension. Default: ``1``.
        device: Execution device. Default: ``'cpu'``.

    Returns:
        Boolean tensor of shape ``[input_size, output_size, kernel_size]`` representing
        the index pattern. Its element ``[i, o, k]`` is ``True`` If element ``i`` if the
        input element ``i`` contributes to output element ``o`` via the ``k`` the kernel
        entry (``False`` otherwise).
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

    return pattern  # shape [kernel_size, output_size, input_size]


def conv_index_pattern_logical(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    device: device = cpu,
) -> Tensor:
    # TODO Support 'valid' padding
    # TODO Support 'same' padding
    # TODO Check padding value if string
    if isinstance(padding, str):
        raise NotImplementedError("String-valued padding not supported.")

    padding_left, padding_right = padding, padding

    output_size = 1 + int(
        (
            (input_size + padding_left + padding_right)
            - (kernel_size + (kernel_size - 1) * (dilation - 1))
        )
        / stride
    )

    pattern = zeros(
        kernel_size, output_size, input_size, dtype=torch.bool, device=device
    )

    for k in range(kernel_size):
        for o in range(output_size):
            i = stride * o - padding_left + k * dilation
            # TODO Integrate this constraint into o's range
            if 0 <= i < input_size:
                pattern[k, o, i] = True

    return pattern
