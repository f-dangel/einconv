from typing import List, Tuple, Union

from torch import Tensor, einsum
from torch.nn import Module

from einconv.index_pattern import conv_index_pattern
from einconv.utils import _tuple, get_letters


def _unfold_einsum_equation(N: int) -> str:
    """Generate einsum equation for unfold.

    The arguments are ``input, *index_patterns -> output``.

    Args:
        N: Convolution dimension.

    Returns:
        Einsum equation for N-dimensional convolution.
    """
    input_str = ""
    output_str = ""
    pattern_strs: List[str] = []

    # requires 2 + 3 * N letters
    letters = get_letters(2 + 3 * N)

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
