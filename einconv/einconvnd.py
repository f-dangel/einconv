"""PyTorch modules and functionals implementing N-dimensional convolution."""

from typing import List, Tuple, Union

from torch import Tensor, einsum
from torch.nn.functional import pad

from einconv.index_pattern import conv_index_pattern
from einconv.utils import _tuple


def einconvNd(
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
            spatial dimensions), or an ``N``-tuple of integers. Default: ``0``.
        dilation: Dilation of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        groups: How to split the input into groups. Default: ``1``.

    Returns:
        Result of the convolution. Has shape ``[batch_size, out_channels, *]`` where
        ``*`` is the the spatial output dimension shape.

    Raises:
        NotImplementedError: If the supplied hyperparameters are not supported.
        ValueError: If bias has incorrect shape.
        ValueError: If weight dimension is incorrect.
        ValueError: If weight shape is invalid.
    """
    if isinstance(padding, str):
        raise NotImplementedError("String-valued padding not yet supported.")

    N = input.dim() - 2

    if weight.dim() != N + 2:
        raise ValueError(
            f"For Conv(N={N})d, the kernel must be {N+2}d. Got {weight.dim()}d."
        )

    (out_channels, _), kernel_sizes = weight.shape[:2], weight.shape[2:]

    if bias is not None and (bias.dim() != 1 or bias.numel() != out_channels):
        raise ValueError(f"Bias should have shape [{out_channels}]. Got {bias.shape}.")

    # convert into tuple format
    t_stride: Tuple[int, ...] = _tuple(stride, N)
    t_padding: Tuple[int, ...] = _tuple(padding, N)
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)

    if any(p != 0 for p in t_padding):
        paddings = sum(([p, p] for p in reversed(t_padding)), [])
        input = pad(input, tuple(paddings))

    (batch_size, in_channels), input_sizes = input.shape[:2], input.shape[2:]

    if weight.shape[0] % groups != 0:
        raise ValueError(
            f"Groups ({groups}) must divide out_channels ({weight.shape[0]})."
        )

    if weight.shape[1] * groups != in_channels:
        raise ValueError(
            f"Kernel dimension 1 ({weight.shape[1]}) multiplied by groups ({groups})"
            + f" must equal in_channels ({in_channels})."
        )

    index_patterns: List[Tensor] = [
        conv_index_pattern(
            input_sizes[n],
            kernel_sizes[n],
            stride=t_stride[n],
            dilation=t_dilation[n],
            device=input.device,
        ).to(input.dtype)
        for n in range(N)
    ]

    equation = _conv_einsum_equation(N)

    output = einsum(
        equation,
        input.reshape(batch_size, groups, in_channels // groups, *input_sizes),
        *index_patterns,
        weight.reshape(
            groups, out_channels // groups, in_channels // groups, *kernel_sizes
        ),
    )
    output = output.flatten(start_dim=1, end_dim=2)

    if bias is not None:
        shape_before_expand = (1, out_channels) + N * (1,)
        output += bias.reshape(*shape_before_expand).expand_as(output)

    return output


def _conv_einsum_equation(N: int) -> str:
    """Generate einsum equation for convolution.

    The arguments are ``input, *index_patterns, weight -> output``.

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
    kernel_str = ""

    # requires 4 + 3 * N letters
    # einsum can deal with the 26 lowercase letters of the alphabet
    max_letters, required_letters = 26, 4 + 3 * N
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
