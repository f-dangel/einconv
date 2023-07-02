"""Tests convolution module."""

from test.modules.conv_cases import (
    CONV_1D_CASES,
    CONV_1D_IDS,
    CONV_2D_CASES,
    CONV_2D_IDS,
    CONV_3D_CASES,
    CONV_3D_IDS,
    CONV_4D_CASES,
    CONV_4D_IDS,
    CONV_5D_CASES,
    CONV_5D_IDS,
    CONV_6D_CASES,
    CONV_6D_IDS,
    conv_module_from_case,
    convNd_module_from_case,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from test.utils_jax import to_ConvNd_jax
from typing import Dict, Union

import torch
from pytest import mark
from torch import device, manual_seed

from einconv.modules import ConvNd


def _test_ConvNd(
    N: int, case: Dict, device: device, dtype: Union[torch.dtype, None] = None
):
    """Compare PyTorch's ConvNd layer with einconv's ConvNd layer.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)

    conv_module = conv_module_from_case(N, case, device, dtype=dtype)
    einconv_module = ConvNd.from_nn_Conv(conv_module)

    report_nonclose(conv_module(x), einconv_module(x), atol=1e-6, rtol=1e-5)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_CASES, ids=CONV_1D_IDS)
def test_Conv1d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv1d layer with einconv's ConvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 1
    _test_ConvNd(N, case, device, dtype=dtype)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_2D_CASES, ids=CONV_2D_IDS)
def test_Conv2d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv2d layer with einconv's ConvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 2
    _test_ConvNd(N, case, device, dtype=dtype)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_3D_CASES, ids=CONV_3D_IDS)
def test_Conv3d(case: Dict, device: device, dtype: Union[torch.dtype, None] = None):
    """Compare PyTorch's Conv3d layer with einconv's ConvNd layer.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    N = 3
    _test_ConvNd(N, case, device, dtype=dtype)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize(
    "case",
    CONV_4D_CASES + CONV_5D_CASES + CONV_6D_CASES,
    ids=CONV_4D_IDS + CONV_5D_IDS + CONV_6D_IDS,
)
def test_Conv_higher_d_jax(
    case: Dict, device: device, dtype: Union[torch.dtype, None] = None
):
    """Compare forward pass of einconv's Conv>=4d layer with JAX implementation.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)
    N = x.dim() - 2

    einconv_module = convNd_module_from_case(N, case, device, dtype=dtype)
    einconv_output = einconv_module(x)

    jax_module = to_ConvNd_jax(einconv_module)
    jax_output = jax_module(x)

    report_nonclose(einconv_output, jax_output, atol=2e-6)
