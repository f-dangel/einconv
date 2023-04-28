"""Contains functionality for tensor network GGN diagonal of convolutions."""

from typing import List, Tuple, Union

from torch import Tensor

from einconv.einconvnd import _conv_einsum_operands
from einconv.utils import get_letters


def _conv_diag_ggn_einsum_equation(N: int) -> str:
    """Return the einsum equation for the GGN diagonal extraction.

    Args:
        N: Convolution dimensionality

    Returns:
        Einsum equation for the GGN diagonal extraction. Argument order is assumed to
        be ``input, *index_patterns, sqrt_ggn, input, *index_patterns, sqrt_ggn``.
    """
    input1_str = ""
    pattern1_strs: List[str] = []
    sqrt_ggn1_str = ""
    input2_str = ""
    pattern2_strs: List[str] = []
    sqrt_ggn2_str = ""
    result_str = ""

    # requires 6 + 5 * N letters
    letters = get_letters(6 + 5 * N)

    # class dimension
    class_letter = letters.pop()
    sqrt_ggn1_str += class_letter
    sqrt_ggn2_str += class_letter

    # batch dimension
    batch_letter = letters.pop()
    input1_str += batch_letter
    input2_str += batch_letter
    sqrt_ggn1_str += batch_letter
    sqrt_ggn2_str += batch_letter

    # group dimension
    group1_letter = letters.pop()
    input1_str += group1_letter
    sqrt_ggn1_str += group1_letter

    group2_letter = letters.pop()
    input2_str += group2_letter
    sqrt_ggn2_str += group2_letter

    # output channel dimension
    out_channel_letter = letters.pop()
    sqrt_ggn1_str += out_channel_letter
    sqrt_ggn2_str += out_channel_letter
    result_str += out_channel_letter

    # input channel dimensions
    in_channel_letter = letters.pop()
    input1_str += in_channel_letter
    input2_str += in_channel_letter
    result_str += in_channel_letter

    # kernel dimensions
    for _ in range(N):
        kernel_letter = letters.pop()
        pattern1_strs.append(kernel_letter)
        pattern2_strs.append(kernel_letter)
        result_str += kernel_letter

    # spatial output dimension
    for n in range(N):
        output1_letter = letters.pop()
        sqrt_ggn1_str += output1_letter
        pattern1_strs[n] += output1_letter

        output2_letter = letters.pop()
        sqrt_ggn2_str += output2_letter
        pattern2_strs[n] += output2_letter

    # spatial input dimensions
    for n in range(N):
        input1_letter = letters.pop()
        input1_str += input1_letter
        pattern1_strs[n] += input1_letter

        input2_letter = letters.pop()
        input2_str += input2_letter
        pattern2_strs[n] += input2_letter

    input_equation = ",".join(
        [input1_str]
        + pattern1_strs
        + [sqrt_ggn1_str, input2_str]
        + pattern2_strs
        + [sqrt_ggn2_str]
    )

    return "->".join([input_equation, result_str])


def _conv_diag_ggn_einsum_operands(
    input: Tensor,
    weight: Tensor,
    sqrt_ggn: Tensor,
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> List[Tensor]:
    """Prepare the tensor contraction operands for the GGN diagonal extraction.

    Args:
        See ``einconvNd``.
        # noqa: DAR101
        sqrt_ggn: Square root of the GGN diagonal has shape
            ``[C, N, C_out, H_out, W_out]`` where ``C`` is number of classes or
            MC-samples and ``N`` is the batch size.

    Returns:
        Tensor list containing the operands. Convention: reshaped input, index pattern
        tensors, GGN matrix square root, reshaped input, index pattern tensors, GGN
        matrix square root.
    """
    operands = _conv_einsum_operands(input, weight, stride, padding, dilation, groups)
    # drop kernel (last)
    operands.pop()

    # separate groups
    num_classes, batch_size, out_channels = sqrt_ggn.shape[:3]
    output_spatial_dims = sqrt_ggn.shape[3:]
    operands.append(
        sqrt_ggn.reshape(
            num_classes,
            batch_size,
            groups,
            out_channels // groups,
            *output_spatial_dims,
        )
    )

    return operands + operands
