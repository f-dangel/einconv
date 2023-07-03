"""Tests for ``einconv/utils``'s."""

from test.utils_cases import OUTPUT_SIZE_CASES, OUTPUT_SIZE_IDS
from typing import Dict

from pytest import mark
from torch import zeros
from torch.nn.functional import conv1d

from einconv.utils import get_conv_output_size


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
