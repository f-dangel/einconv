"""Tests for ``einconv/einconvnd``'s convolution module operations."""

from test.conv_module_cases import (
    CONV_1D_MODULE_CASES,
    CONV_1D_MODULE_IDS,
    conv_module_from_case,
    einconv_module_from_case,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose, sync_parameters
from typing import Dict, Union

import torch
from pytest import mark
from torch import device, manual_seed


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_MODULE_CASES, ids=CONV_1D_MODULE_IDS)
def test_Einconv1d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv3d layer with einconv's EinconvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)

    N = 1
    conv_module = conv_module_from_case(N, case, device, dtype=dtype)
    einconv_module = einconv_module_from_case(N, case, device, dtype=dtype)

    # use same parameters
    sync_parameters(conv_module, einconv_module)

    report_nonclose(conv_module(x), einconv_module(x), atol=1e-7)
