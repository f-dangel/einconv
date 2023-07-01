"""Tests for ``einconv/index_pattern``."""

from test.index_pattern_cases import INDEX_PATTERN_CASES, INDEX_PATTERN_IDS
from test.utils import DEVICE_IDS, DEVICES, DTYPE_IDS, DTYPES, report_nonclose
from typing import Dict

from pytest import mark
from torch import device, dtype

from einconv import index_pattern
from einconv.conv_index_pattern import index_pattern_logical


@mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", INDEX_PATTERN_CASES, ids=INDEX_PATTERN_IDS)
def test_index_pattern(case: Dict, device: device, dtype: dtype):
    """Compare index pattern computations (convolution vs. logical).

    Args:
        case: Dictionary specifying the test case.
        device: Device to carry out the computation.
        dtype: Data type of the pattern tensor.
    """
    pattern_conv = index_pattern(**case, device=device, dtype=dtype)
    pattern_logical = index_pattern_logical(**case, device=device, dtype=dtype)

    for p in [pattern_conv, pattern_logical]:
        assert p.dtype == dtype
        assert p.device == device
        assert hasattr(p, "_pattern_hyperparams")

    report_nonclose(pattern_conv, pattern_logical)
