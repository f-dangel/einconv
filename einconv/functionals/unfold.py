from typing import List, Tuple, Union

from torch import Tensor, einsum

from einconv.index_pattern import conv_index_pattern
from einconv.utils import _tuple


def unfoldNd(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
) -> Tensor:
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    Pytorch functional that accepts N-dimensional inputs. Acts like
    ``torch.nn.functional.unfold`` for a 4d input. Uses tensor networks under the hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html#unfold.

    # noqa: DAR101
    # noqa: DAR201
    """
    N = input.dim() - 2
    input_sizes = input.shape[2:]

    # convert into tuple format
    t_kernel_size: Tuple[int, ...] = _tuple(kernel_size, N)
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)
    t_padding: Union[Tuple[int, ...], str] = _tuple(padding, N)
    t_stride: Tuple[int, ...] = _tuple(stride, N)

    index_patterns: List[Tensor] = [
        conv_index_pattern(
            input_sizes[n],
            t_kernel_size[n],
            stride=t_stride[n],
            padding=t_padding[n],
            dilation=t_dilation[n],
            device=input.device,
            dtype=input.dtype,
        )
        for n in range(N)
    ]

    equation = _unfold_einsum_equation(N)

    # [batch_size, in_channels, *kernel_size, *output_size]
    output = einsum(equation, input, *index_patterns)

    # [batch_size, in_channels * kernel_size_numel, output_size_numel]
    output = output.flatten(start_dim=1, end_dim=1 + N).flatten(start_dim=2)

    return output
