"""Generates einsum expression of the forward pass of a convolution."""

from typing import List, Optional, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

from einconv import index_pattern
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    weight: Union[Tensor, Parameter],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    simplify: Optional[bool] = False,
) -> Tuple[str, Tuple[Tensor], Tuple[int]]:
    """Generate einsum expression of a convolution's forward pass.

    Args:
        simplify: Whether to simplify the expression. Default: ``False``.

    Returns:
        Einsum equation
        Einsum operands
        Output shape

    Raises:
        NotImplementedError
    """
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x, weight, dilation=dilation, padding=padding, stride=stride, groups=groups
    )
    return equation, operands, shape


def _operands_and_shape(
    x: Tensor,
    weight: Tensor,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Prepare the tensor contraction operands.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Returns:
        Tensor list containing the operands. Convention: reshaped input, followed by
        index pattern tensors, followed by reshaped weight.
    """
    N = input.dim() - 2

    batch_size, input_sizes = input.shape[0], input.shape[2:]
    (out_channels, _), kernel_sizes = weight.shape[:2], weight.shape[2:]

    # convert into tuple format
    t_stride: Tuple[int, ...] = _tuple(stride, N)
    t_padding: Union[Tuple[int, ...], str] = (
        padding if isinstance(padding, str) else _tuple(padding, N)
    )
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)

    patterns = [
        index_pattern(
            input_sizes[n],
            kernel_sizes[n],
            stride=t_stride[n],
            padding=padding if isinstance(padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=x.device,
            dtype=x.dtype,
        )
        for n in range(N)
    ]
    output_sizes = [p.shape[1] for p in patterns]
    output_shape = (batch_size, out_channels, *output_sizes)

    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_out) ... -> g c_out ...", g=groups)

    return [x_ungrouped] + patterns + [weight_ungrouped], output_shape


def _equation(N: int) -> str:
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
