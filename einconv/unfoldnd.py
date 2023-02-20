"""Implementation of ``im2col`` for convolutions using tensor networks."""

from typing import List, Tuple, Union

from torch import Tensor, einsum
from torch.nn import Module

from einconv.index_pattern import conv_index_pattern
from einconv.utils import _tuple


class UnfoldNd(Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    PyTorch module that accepts Nd tensors. Acts like ``torch.nn.Unfold``
    for a 4d input. Uses tensor networks under the hood.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        stride: Union[int, Tuple[int, ...]] = 1,
    ):  # noqa: D107
        super().__init__()

        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input: Tensor) -> Tensor:  # noqa: D102
        return unfoldNd(
            input,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )


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
        ).to(input.dtype)
        for n in range(N)
    ]

    equation = _unfold_einsum_equation(N)

    # [batch_size, in_channels, *kernel_size, *output_size]
    output = einsum(equation, input, *index_patterns)

    # [batch_size, in_channels * kernel_size_numel, output_size_numel]
    output = output.flatten(start_dim=1, end_dim=1 + N).flatten(start_dim=2)

    return output


def _unfold_einsum_equation(N: int) -> str:
    """Generate einsum equation for unfold.

    The arguments are ``input, *index_patterns -> output``.

    Args:
        N: Convolution dimension.

    Raises:
        ValueError: If the equation cannot be realized without exceeding the alphabet.

    Returns:
        Einsum equation for N-dimensional convolution.
    """
    input_str = ""
    output_str = ""
    pattern_strs: List[str] = []

    # requires 2 + 3 * N letters
    # einsum can deal with the 26 lowercase letters of the alphabet
    max_letters, required_letters = 26, 2 + 3 * N
    if required_letters > max_letters:
        raise ValueError(
            f"Cannot form einsum equation. Need {required_letters} letters."
            + f" But einsum only supports {max_letters}."
        )
    letters = [chr(ord("a") + i) for i in range(required_letters)]

    # batch dimension
    batch_letter = letters.pop()
    input_str += batch_letter
    output_str += batch_letter

    # input channel dimension
    in_channel_letter = letters.pop()
    input_str += in_channel_letter
    output_str += in_channel_letter

    # coupling of input and index pattern
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        input_str += input_letter
        output_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    for n in range(N):
        output_str += pattern_strs[n][1]

    input_equation = ",".join([input_str] + pattern_strs)

    return "->".join([input_equation, output_str])
