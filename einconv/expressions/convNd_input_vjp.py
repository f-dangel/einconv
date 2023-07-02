"""Generates einsum expression of the input VJP of a convolution."""

from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter

from einconv import index_pattern
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    weight: Union[Tensor, Parameter],
    v: Tensor,
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[str, List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Generate einsum expression of a convolution's input VJP.

    Args:
        simplify: Whether to simplify the expression. Default: ``False``.

    Returns:
        Einsum equation,
        Einsum operands,
        Output shape.

    Raises:
        NotImplementedError
    """
    N = x.dim() - 2
    input_sizes = x.shape[2:]
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        weight,
        v,
        input_sizes,
        dilation=dilation,
        padding=padding,
        stride=stride,
        groups=groups,
    )
    return equation, operands, shape


def _equation(N: int) -> str:
    """Return the einsum equation for an input VJP.

    Args:
        N: Convolution dimensionality

    Returns:
        Einsum equation for the input VJP. Argument order is assumed to
        be ``v, *index_patterns, weight``.
    """
    v_str = ""
    output_str = ""
    pattern_strs: List[str] = []
    weight_str = ""

    # requires 4 + 3 * N letters
    letters = get_letters(4 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    v_str += batch_letter
    output_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    v_str += group_letter
    output_str += group_letter
    weight_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    v_str += out_channel_letter
    output_str += in_channel_letter
    weight_str += out_channel_letter + in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        v_str += output_letter
        output_str += input_letter
        weight_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([v_str] + pattern_strs + [weight_str])

    return "->".join([input_equation, output_str])


def _operands_and_shape(
    weight: Union[Tensor, Parameter],
    v: Tensor,
    input_sizes: Tuple[int, ...],
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> Tuple[List[Union[Tensor, Parameter]], Tuple[int, ...]]:
    """Prepare the tensor contraction operands for the VJP.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Returns:
        Tensor list containing the operands. Convention: reshaped grad_output, followed
        by index pattern tensors, followed by reshaped kernel.
    """
    N = weight.dim() - 2
    batch_size = v.shape[0]
    in_channels, kernel_size = weight.shape[1], weight.shape[2:]

    # convert into tuple format
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)
    t_padding: Union[Tuple[int, ...], str] = _tuple(padding, N)
    t_stride: Tuple[int, ...] = _tuple(stride, N)

    patterns: List[Tensor] = [
        index_pattern(
            input_sizes[n],
            kernel_size[n],
            stride=t_stride[n],
            padding=t_padding[n],
            dilation=t_dilation[n],
            device=weight.device,
            dtype=weight.dtype,
        )
        for n in range(N)
    ]
    output_shape = (batch_size, in_channels, *input_sizes)

    v_ungrouped = rearrange(v, "n (g c_out) ... -> n g c_out ...", g=groups)
    weight_ungrouped = rearrange(weight, "(g c_out) ... -> g c_out ...", g=groups)

    return [v_ungrouped] + patterns + [weight_ungrouped], output_shape
