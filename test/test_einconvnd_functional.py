"""Tests for ``einconv/einconvnd``'s convolution functional operations."""

from test.conv_functional_cases import (
    CONV_1D_FUNCTIONAL_CASES,
    CONV_1D_FUNCTIONAL_IDS,
    CONV_2D_FUNCTIONAL_CASES,
    CONV_2D_FUNCTIONAL_IDS,
    CONV_3D_FUNCTIONAL_CASES,
    CONV_3D_FUNCTIONAL_IDS,
    CONV_4D_FUNCTIONAL_CASES,
    CONV_4D_FUNCTIONAL_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from test.utils_jax import jax_convNd
from typing import Dict, Tuple, Union

from pytest import mark
from torch import Tensor, device, manual_seed
from torch.nn.functional import conv1d, conv2d, conv3d

from einconv.einconvnd import einconvNd


def _test_einconvNd(N: int, case: Dict, device: device):
    """Compare PyTorch's convNd with einconv's einconvNd.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    x, weight, bias = _setup(case, device)

    conv_func = {1: conv1d, 2: conv2d, 3: conv3d}[N]
    conv_output = conv_func(x, weight, bias=bias, **case["conv_kwargs"])
    einconv_output = einconvNd(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(conv_output, einconv_output)


def _setup(case: Dict, device: device) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
    """Set random seed, then construct input, weight, bias, and load to device.

    Args:
        case: Dictionary describing the test case.
        device: Device to load all tensors to.

    Returns:
        input, weight, bias.
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)
    weight = case["weight_fn"]().to(device)
    bias = case["bias_fn"]()
    if bias is not None:
        bias = bias.to(device)

    return x, weight, bias


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_FUNCTIONAL_CASES, ids=CONV_1D_FUNCTIONAL_IDS)
def test_einconv1d(case: Dict, device: device):
    """Compare PyTorch's conv1d with einconv's einconvNd.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    N = 1
    _test_einconvNd(N, case, device)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_2D_FUNCTIONAL_CASES, ids=CONV_2D_FUNCTIONAL_IDS)
def test_einconv2d(case: Dict, device: device):
    """Compare PyTorch's conv2d with einconv's einconvNd.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    N = 2
    _test_einconvNd(N, case, device)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_3D_FUNCTIONAL_CASES, ids=CONV_3D_FUNCTIONAL_IDS)
def test_einconv3d(case: Dict, device: device):
    """Compare PyTorch's conv3d with einconv's einconvNd.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    N = 3
    _test_einconvNd(N, case, device)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_4D_FUNCTIONAL_CASES, ids=CONV_4D_FUNCTIONAL_IDS)
def test_einconv4d(case: Dict, device: device):
    """Compare einconv's einconvNd for N=4 with JAX implementation.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
    """
    x, weight, bias = _setup(case, device)
    einconv_output = einconvNd(x, weight, bias=bias, **case["conv_kwargs"])
    jax_output = jax_convNd(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(einconv_output, jax_output)
