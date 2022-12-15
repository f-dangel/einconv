"""Tests for ``einconv/einconvnd``'s convolution module operations."""

from test.conv_module_cases import (
    CONV_1D_MODULE_CASES,
    CONV_1D_MODULE_IDS,
    CONV_2D_MODULE_CASES,
    CONV_2D_MODULE_IDS,
    CONV_3D_MODULE_CASES,
    CONV_3D_MODULE_IDS,
    CONV_4D_MODULE_CASES,
    CONV_4D_MODULE_IDS,
    conv_module_from_case,
    einconv_module_from_case,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from test.utils_third_party import to_ConvNd_third_party
from typing import Dict, Union

import torch
from pytest import mark, skip
from torch import device, manual_seed

from einconv.einconvnd import EinconvNd


def _test_EinconvNd(
    N: int, case: Dict, device: device, dtype: Union[torch.dtype, None] = None
):
    """Compare PyTorch's ConvNd layer with einconv's EinconvNd layer.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)

    conv_module = conv_module_from_case(N, case, device, dtype=dtype)
    einconv_module = EinconvNd.from_nn_Conv(conv_module)
    print(einconv_module)

    report_nonclose(conv_module(x), einconv_module(x), atol=1e-6, rtol=1e-5)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_MODULE_CASES, ids=CONV_1D_MODULE_IDS)
def test_Einconv1d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv3d layer with einconv's EinconvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 1
    _test_EinconvNd(N, case, device, dtype=dtype)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_2D_MODULE_CASES, ids=CONV_2D_MODULE_IDS)
def test_Einconv2d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv2d layer with einconv's EinconvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 2
    _test_EinconvNd(N, case, device, dtype=dtype)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_3D_MODULE_CASES, ids=CONV_3D_MODULE_IDS)
def test_Einconv3d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv3d layer with einconv's EinconvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 3
    _test_EinconvNd(N, case, device, dtype=dtype)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_4D_MODULE_CASES, ids=CONV_4D_MODULE_IDS)
def test_Einconv4d_integration(
    case: Dict, device: device, dtype: Union[torch.dtype, None] = None
):
    """Run a forward pass of einconv's Einconv4d layer without verifying correctness.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 4
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)

    einconv_module_from_case(N, case, device, dtype=dtype)(x)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_4D_MODULE_CASES, ids=CONV_4D_MODULE_IDS)
def test_Einconv4d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare forward pass of einconv's Einconv4d layer with 3rd-party implementation.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 4
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)

    einconv_module = einconv_module_from_case(N, case, device, dtype=dtype)
    einconv_output = einconv_module(x)

    if (
        N != 4
        or einconv_module.bias is not None
        or any(d != 1 for d in einconv_module.dilation)
    ):
        skip()

    third_party_module = to_ConvNd_third_party(einconv_module)
    third_party_output = third_party_module(x)

    report_nonclose(einconv_output, third_party_output, atol=1e-6)
