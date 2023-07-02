from typing import List, Tuple, Union

from einops import rearrange
from torch import Tensor

from einconv import index_pattern
from einconv.utils import _tuple, get_letters


def einsum_expression(
    x: Tensor,
    v: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    stride: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
) -> Tuple[str, List[Tensor], Tuple[int, ...]]:
    """Generate einsum expression of a convolution's weight VJP.

    Returns:
        Einsum equation,
        Einsum operands,
        Output shape.
    """
    N = x.dim() - 2
    equation = _equation(N)
    operands, shape = _operands_and_shape(
        x,
        v,
        kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
        groups=groups,
    )
    return equation, operands, shape


def _equation(N: int) -> str:
    """Return the einsum equation for a weight VJP.

    Args:
        N: Convolution dimensionality

    Returns:
        Einsum equation for the weight VJP. Argument order is assumed to
        be ``x, *patterns, v``.
    """
    v_str = ""
    x_str = ""
    output_str = ""
    pattern_strs: List[str] = []

    # requires 4 + 3 * N letters
    letters = get_letters(4 + 3 * N)

    # batch dimension
    batch_letter = letters.pop()
    v_str += batch_letter
    x_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    v_str += group_letter
    output_str += group_letter
    x_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    v_str += out_channel_letter
    output_str += out_channel_letter + in_channel_letter
    x_str += in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        v_str += output_letter
        output_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([x_str] + pattern_strs + [v_str])

    return "->".join([input_equation, output_str])


def _operands_and_shape(
    x: Tensor,
    v: Tensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> List[Tensor]:
    """Prepare the tensor contraction operands for the VJP.

    Returns:
        Tensor list containing the operands. Convention: reshaped input, followed by
        index pattern tensors, followed by reshaped grad_output.
    """
    # convert into tuple format
    N = x.dim() - 2
    input_sizes = x.shape[2:]
    t_kernel_size: Tuple[int, ...] = _tuple(kernel_size, N)
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)
    t_padding: Union[Tuple[int, ...], str] = (
        padding if isinstance(padding, str) else _tuple(padding, N)
    )
    t_stride: Tuple[int, ...] = _tuple(stride, N)

    patterns: List[Tensor] = [
        index_pattern(
            input_sizes[n],
            t_kernel_size[n],
            stride=t_stride[n],
            padding=t_padding if isinstance(t_padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=x.device,
            dtype=x.dtype,
        )
        for n in range(N)
    ]
    x_ungrouped = rearrange(x, "n (g c_in) ... -> n g c_in ...", g=groups)
    v_ungrouped = rearrange(v, "n (g c_out) ... -> n g c_out ...", g=groups)

    operands = [x_ungrouped, *patterns, v_ungrouped]

    in_channels = x.shape[1]
    out_channels = v.shape[1]
    shape = (out_channels, in_channels // groups, *t_kernel_size)

    return operands, shape
