"""PyTorch equivalent of ``nn.functional.conv{1,2,3}d`` implemented as einsum."""

from typing import List, Tuple, Union

from torch import Tensor


def convNd(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
):
    """Generalization of ``torch.nn.functional.conv{1,2,3}d`` to ``N``d.

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
    _conv_check_args(input, weight, bias, groups)

    N = input.dim() - 2
    equation = _conv_einsum_equation(N)
    operands: List[Tensor] = _conv_einsum_operands(
        input, weight, stride, padding, dilation, groups
    )
    output = einsum(equation, *operands)
    output = output.flatten(start_dim=1, end_dim=2)

    if bias is not None:
        out_channels = weight.shape[0]
        shape_before_expand = (1, out_channels) + N * (1,)
        output += bias.reshape(*shape_before_expand).expand_as(output)

    return output


def _conv_check_args(
    input: Tensor, weight: Tensor, bias: Union[Tensor, None], groups: int
):
    """Check the input arguments to ``einconvNd``.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Raises:
        ValueError: If bias has incorrect shape.
        ValueError: If weight dimension is incorrect.
        ValueError: If weight shape is invalid.
    """
    N = input.dim() - 2
    in_channels = input.shape[1]
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


def _conv_einsum_equation(N: int) -> str:
    """Generate einsum equation for convolution.

    The arguments are ``input, *index_patterns, weight -> output``.

    See https://arxiv.org/pdf/1908.04471.pdf, figure 2a for a visualization of the 3d
    case (neglecting the groups). The Nd case follows identically, and groups can be
    supported by a separate axis in the input, weight, and output.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for N-dimensional convolution.
    """
    input_str = ""
    output_str = ""
    pattern_strs: List[str] = []
    kernel_str = ""

    # requires 4 + 3 * N letters
    letters = get_letters(4 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    input_str += batch_letter
    output_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    input_str += group_letter
    output_str += group_letter
    kernel_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    input_str += in_channel_letter
    output_str += out_channel_letter
    kernel_str += out_channel_letter + in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        input_str += input_letter
        output_str += output_letter
        kernel_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([input_str] + pattern_strs + [kernel_str])

    return "->".join([input_equation, output_str])


def _conv_einsum_operands(
    input: Tensor,
    weight: Tensor,
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> List[Tensor]:
    """Prepare the tensor contraction operands.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Returns:
        Tensor list containing the operands. Convention: reshaped input, followed by
        index pattern tensors, followed by reshaped weight.
    """
    N = input.dim() - 2

    (batch_size, in_channels), input_sizes = input.shape[:2], input.shape[2:]
    (out_channels, _), kernel_sizes = weight.shape[:2], weight.shape[2:]

    operands = [
        input.reshape(batch_size, groups, in_channels // groups, *input_sizes),
    ]

    # convert into tuple format
    t_stride: Tuple[int, ...] = _tuple(stride, N)
    t_padding: Union[Tuple[int, ...], str] = (
        padding if isinstance(padding, str) else _tuple(padding, N)
    )
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)

    operands.extend(
        conv_index_pattern(
            input_sizes[n],
            kernel_sizes[n],
            stride=t_stride[n],
            padding=padding if isinstance(padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=input.device,
            dtype=input.dtype,
        )
        for n in range(N)
    )

    operands.append(
        weight.reshape(
            groups, out_channels // groups, in_channels // groups, *kernel_sizes
        )
    )

    return operands
