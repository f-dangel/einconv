"""Tests for ``einconv/einconvnd``."""

from test.cases import (
    CONV_1D_FUNCTIONAL_CASES,
    CONV_1D_FUNCTIONAL_IDS,
    CONV_2D_FUNCTIONAL_CASES,
    CONV_2D_FUNCTIONAL_IDS,
    CONV_3D_FUNCTIONAL_CASES,
    CONV_3D_FUNCTIONAL_IDS,
    DEVICE_IDS,
    DEVICES,
)
from test.utils import report_nonclose
from typing import Dict

from pytest import mark
from torch import device, manual_seed
from torch.nn.functional import conv1d, conv2d, conv3d

from einconv.einconvnd import einconv1d, einconv2d, einconv3d


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

    conv_output = conv1d(x, weight, bias=bias, **case["conv_kwargs"])
    einconv_output = einconv1d(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(conv_output, einconv_output)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_2D_FUNCTIONAL_CASES, ids=CONV_2D_FUNCTIONAL_IDS)
def test_einconv2d(case: Dict, device: device):
    """Compare PyTorch's conv2d with einconv's einconv2d.

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

    conv_output = conv2d(x, weight, bias=bias, **case["conv_kwargs"])
    einconv_output = einconv2d(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(conv_output, einconv_output)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_3D_FUNCTIONAL_CASES, ids=CONV_3D_FUNCTIONAL_IDS)
def test_einconv3d(case: Dict, device: device):
    """Compare PyTorch's conv3d with einconv's einconv3d.

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

    conv_output = conv3d(x, weight, bias=bias, **case["conv_kwargs"])
    einconv_output = einconv3d(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(conv_output, einconv_output)
