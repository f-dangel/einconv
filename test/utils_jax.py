"""Utility functions for tests comparing against JAX's N-dimensional convolution."""

from typing import Tuple, Union

import jax
import numpy
import torch

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
    """JAX implementation of ``nn.functional.convNd``."""
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
