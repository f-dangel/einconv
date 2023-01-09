"""Tests for ``einconv/index_pattern``."""

from test.index_pattern_cases import INDEX_PATTERN_CASES, INDEX_PATTERN_IDS
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict

from pytest import mark
from torch import device

from einconv.index_pattern import conv_index_pattern, conv_index_pattern_logical


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", INDEX_PATTERN_CASES, ids=INDEX_PATTERN_IDS)
def test_conv_index_pattern(case: Dict, device: device):
    """Compare index pattern computations (convolution vs. logical).

    Args:
        case: Dictionary specifying the test case.
        device: Device to carry out the computation.
    """
    pattern_conv = conv_index_pattern(**case, device=device)
    pattern_logical = conv_index_pattern_logical(**case, device=device)

    report_nonclose(pattern_conv, pattern_logical)
