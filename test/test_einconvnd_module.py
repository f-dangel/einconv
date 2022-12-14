"""Tests for ``einconv/einconvnd``'s convolution module operations."""

from test.conv_module_cases import CONV_1D_MODULE_CASES, CONV_1D_MODULE_IDS
from test.utils import DEVICE_IDS, DEVICES, compare_attributes, report_nonclose
from typing import Dict

from pytest import mark
from torch import device, manual_seed
from torch.nn import Conv1d

from einconv.einconvnd import EinconvNd


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_MODULE_CASES, ids=CONV_1D_MODULE_IDS)
def test_Einconv1d(case: Dict, device: device):
    """Compare PyTorch's Conv3d layer with einconv's EinconvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)

    conv_module = Conv1d(
        case["in_channels"],
        case["out_channels"],
        case["kernel_size"],
        **case["conv_kwargs"],
        device=device,
    )
    conv_output = conv_module(x)

    einconv_module = EinconvNd(
        1,
        case["in_channels"],
        case["out_channels"],
        case["kernel_size"],
        **case["conv_kwargs"],
        device=device,
    )
    attributes = ["shape", "dtype", "device"]
    compare_attributes(conv_module.weight, einconv_module.weight, attributes)
    einconv_module.weight = conv_module.weight

    if conv_module.bias is None:
        assert einconv_module.bias is None
    else:
        compare_attributes(conv_module.bias, einconv_module.bias, attributes)
        einconv_module.bias = conv_module.bias

    einconv_output = einconv_module(x)

    report_nonclose(conv_output, einconv_output, atol=1e-7)
