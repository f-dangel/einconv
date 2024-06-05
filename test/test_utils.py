"""Tests for ``einconv/utils``'s."""

from test.utils_cases import (
    INPUT_SIZE_CASES,
    INPUT_SIZE_IDS,
    OUTPUT_SIZE_CASES,
    OUTPUT_SIZE_IDS,
)
from typing import Dict

from pytest import mark
from torch import zeros
from torch.nn.functional import conv1d, conv_transpose1d

from einconv.utils import get_conv_input_size, get_conv_output_size


@mark.parametrize("case", OUTPUT_SIZE_CASES, ids=OUTPUT_SIZE_IDS)
def test_get_conv_output_size(case: Dict):
    """Test output size computation of a convolution.

    Args:
        case: Dictionary describing the test case.
    """
    output_torch = conv1d(
        zeros(1, 1, case["input_size"]),  # [N, C_in, I]
        zeros(1, 1, case["kernel_size"]),  # [C_out, C_in, K]
        bias=None,
        stride=case["stride"],
        padding=case["padding"],
        dilation=case["dilation"],
    )  # [N, C_out, O]
    output_size_torch = output_torch.shape[2]

    output_size = get_conv_output_size(**case)

    assert output_size_torch == output_size


@mark.parametrize("case", INPUT_SIZE_CASES, ids=INPUT_SIZE_IDS)
def test_get_conv_input_size(case: Dict):
    """Test input size computation of a convolution.

    Args:
        case: Dictionary describing the test case.
    """
    input_torch = conv_transpose1d(
        zeros(1, 1, case["output_size"]),  # [N, C_out, O]
        zeros(1, 1, case["kernel_size"]),  # [C_out, C_in, K]
        bias=None,
        stride=case["stride"],
        padding=case["padding"],
        output_padding=case["output_padding"],
        dilation=case["dilation"],
    )
    input_size_torch = input_torch.shape[2]

    input_size = get_conv_input_size(**case)

    assert input_size_torch == input_size
