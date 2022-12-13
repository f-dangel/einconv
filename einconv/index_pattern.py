"""Contains functionality to implement convolution as tensor contraction (einsum)."""

import torch
from torch import Tensor, arange, device, eye, ones_like, zeros
from torch.nn.functional import conv1d

cpu = device("cpu")


def conv_index_pattern(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    device: device = cpu,
) -> Tensor:
    """Compute the 'dummy tensor' containing the index pattern a convolution dimension.

    The dummy tensor is denoted 𝒫 in the paper (see page 3):

    - Hayashi, K., Yamaguchi, T., Sugawara, Y., & Maeda, S. (2019). Exploring
      unexplored tensor network decompositions for convolutional neural networks.
      Advances in Neural Information Processing Systems (NeurIPS).

    Args:
        input_size: Number of pixels along dimension.
        kernel_size: Kernel size along dimension.
        stride: Stride along dimension.
        dilation: Dilation along dimension.
        device: Execution device. Default: ``'cpu'``.

    Returns:
        Boolean tensor of shape ``[input_size, output_size, kernel_size]`` representing
        the index pattern. Its element ``[i, o, k]`` is ``True`` If element ``i`` if the
        input element ``i`` contributes to output element ``o`` via the ``k`` the kernel
        entry (``False`` otherwise).
    """
    in_idxs = (
        arange(
            end=input_size,
            dtype=torch.int32 if dilation == 1 else torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # shape [N=1, C_in=1, input_size]
    weight = eye(kernel_size, dtype=in_idxs.dtype, device=device).unsqueeze(
        1
    )  # shape [C_out=kernel_size, C_in=1, K=kernel_size], entries [k, 1, k] = 1 else 0
    out_idxs = (
        conv1d(in_idxs, weight, stride=stride, dilation=dilation)
        .squeeze(0)
        .unsqueeze(-1)
    )  # shape [K, O, 1], entry [k, o, 0] contains index of the input that
    # contributes to the o-th output element via the k-th kernel element

    # scatter True to [k, o, out_idxs[k, o]] ∀ k, o
    output_size = out_idxs.shape[1]
    pattern = zeros(kernel_size, output_size, input_size, dtype=torch.bool)
    pattern.scatter_add_(2, out_idxs.long(), ones_like(pattern))

    return pattern  # shape [kernel_size, output_size, input_size]
