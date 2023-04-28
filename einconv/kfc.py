"""Implementation of the input-based of the KFC Fisher approximation for convolutions.

Details see:

- Grosse, R., & Martens, J. (2016). A Kronecker-factored approximate Fisher matrix
  for convolution layers. International Conference on Machine Learning (ICML).
"""

from typing import List, Tuple, Union

from torch import Tensor, einsum

from einconv.index_pattern import conv_index_pattern
from einconv.utils import _tuple, get_letters


def kfc_factor(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
) -> Tensor:
    """Compute the mean of the unfolded's input inner product.

    The inner product is taken over the spatial axes.

    Args:
        input: (N+2)-dimensional tensor containing the input of the convolution.
        kernel_size: N-dimensional tuple or integer containing the kernel size.
        dilation: N-dimensional tuple or integer containing the dilation.
            Default: ``1``.
        padding: N-dimensional tuple or integer containing the padding. Default: ``0``.
        stride: N-dimensional tuple or integer containing the stride. Default: ``1``.

    Returns:
        KFC factor of shape ``[in_channels * kernel_size_numel,
        in_channels * kernel_size_numel]``.
    """
    N = input.dim() - 2
    batch_size = input.shape[0]

    equation = _kfc_factor_einsum_equation(N)
    operands = _kfc_factor_einsum_operands(
        input, kernel_size, stride, padding, dilation
    )

    # [in_channels, *kernel_size, in_channels, *kernel_size]
    output = einsum(equation, *operands)

    # [in_channels * kernel_size_numel, in_channels * kernel_size_numel]
    output = output.flatten(end_dim=N).flatten(start_dim=1)

    return output / batch_size


def _kfc_factor_einsum_equation(N: int) -> str:
    """Generate einsum equation for KFC factor.

    The arguments are ``input, *index_patterns, *index_patterns, input -> output``.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for KFC factor of N-dimensional convolution.
    """
    input1_str = ""
    pattern1_strs: List[str] = []
    input2_str = ""
    pattern2_strs: List[str] = []
    output_str = ""

    # requires 3 + 5 * N letters
    letters = get_letters(3 + 5 * N)

    # batch dimension
    batch_letter = letters.pop()
    input1_str += batch_letter
    input2_str += batch_letter

    # input channel dimension for first input
    in_channel_letter = letters.pop()
    input1_str += in_channel_letter
    output_str += in_channel_letter

    # coupling of first input and index pattern
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        input1_str += input_letter
        output_str += kernel_letter
        pattern1_strs.append(kernel_letter + output_letter + input_letter)

    # input channel dimension for second input
    in_channel_letter = letters.pop()
    input2_str += in_channel_letter
    output_str += in_channel_letter

    # coupling of second input and index pattern
    for n in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = pattern1_strs[n][1]

        input2_str += input_letter
        output_str += kernel_letter
        pattern2_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join(
        [input1_str] + pattern1_strs + pattern2_strs + [input2_str]
    )

    return "->".join([input_equation, output_str])


def _kfc_factor_einsum_operands(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
) -> List[Tensor]:
    """Generate einsum operands for KFC factor.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Returns:
        Tensor list containing the operands. Convention: Input, followed by
        index pattern tensors, followed by index pattern tensors, followed by input.
    """
    N = input.dim() - 2
    input_sizes = input.shape[2:]

    # convert into tuple format
    t_kernel_size: Tuple[int, ...] = _tuple(kernel_size, N)
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)
    t_padding: Union[Tuple[int, ...], str] = (
        padding if isinstance(padding, str) else _tuple(padding, N)
    )
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

    return [input, *index_patterns, *index_patterns, input]
