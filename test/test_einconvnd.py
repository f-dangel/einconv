"""Tests for ``einconv/einconvnd``."""

from test.cases import (
    CONV_1D_FUNCTIONAL_CASES,
    CONV_1D_FUNCTIONAL_IDS,
    DEVICE_IDS,
    DEVICES,
)
from test.utils import report_nonclose
from typing import Dict

from pytest import mark
from torch import device, manual_seed
from torch.nn.functional import conv1d

from einconv.einconvnd import einconv1d


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_FUNCTIONAL_CASES, ids=CONV_1D_FUNCTIONAL_IDS)
def test_einconv1d(case: Dict, device: device):
    """Compare PyTorch's conv1d with einconv's einconv1d.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)
    weight = case["weight_fn"]().to(device)
    bias = case["bias_fn"]()
    if bias is not None:
        bias = bias.to(device)

    conv1d_output = conv1d(x, weight, bias=bias, **case["conv_kwargs"])
    einconv1d_output = einconv1d(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(conv1d_output, einconv1d_output)
