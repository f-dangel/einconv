"""Test ``einconv.functionals.conv``."""

from test.functionals.conv_cases import (
    CONV_1D_CASES,
    CONV_1D_IDS,
    CONV_2D_CASES,
    CONV_2D_IDS,
    CONV_3D_CASES,
    CONV_3D_IDS,
    CONV_4D_CASES,
    CONV_4D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from test.utils_jax import jax_convNd
from typing import Dict, Tuple, Union

from pytest import mark
from torch import Tensor, device, manual_seed
from torch.nn.functional import conv1d, conv2d, conv3d

from einconv import functionals


def _test_convNd(N: int, case: Dict, device: device, simplify: bool):
    """Compare PyTorch's convNd with einconv's convNd.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the test case.
        device: Device for executing the test.
        simplify: Whether to use a simplified einsum expression.
    """
    x, weight, bias = _setup(case, device)

    conv_func = {1: conv1d, 2: conv2d, 3: conv3d}[N]
    conv_output = conv_func(x, weight, bias=bias, **case["kwargs"])
    einconv_output = functionals.convNd(
        x, weight, bias=bias, **case["kwargs"], simplify=simplify
    )

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


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_1D_CASES, ids=CONV_1D_IDS)
def test_conv1d(case: Dict, device: device, simplify: bool):
    """Compare PyTorch's conv1d with einconv's convNd.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        simplify: Whether to use a simplified einsum expression.
    """
    N = 1
    _test_convNd(N, case, device, simplify)


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_2D_CASES, ids=CONV_2D_IDS)
def test_conv2d(case: Dict, device: device, simplify: bool):
    """Compare PyTorch's conv2d with einconv's convNd.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        simplify: Whether to use a simplified einsum expression.
    """
    N = 2
    _test_convNd(N, case, device, simplify)


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_3D_CASES, ids=CONV_3D_IDS)
def test_conv3d(case: Dict, device: device, simplify: bool):
    """Compare PyTorch's conv3d with einconv's convNd.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        simplify: Whether to use a simplified einsum expression.
    """
    N = 3
    _test_convNd(N, case, device, simplify)


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("case", CONV_4D_CASES, ids=CONV_4D_IDS)
def test_conv4d(case: Dict, device: device, simplify: bool):
    """Compare einconv's convNd for N=4 with JAX implementation.

    Args:
        case: Dictionary describing the test case.
        device: Device for executing the test.
        simplify: Whether to use a simplified einsum expression.
    """
    x, weight, bias = _setup(case, device)
    einconv_output = functionals.convNd(
        x, weight, bias=bias, **case["kwargs"], simplify=simplify
    )
    jax_output = jax_convNd(x, weight, bias=bias, **case["kwargs"])

    report_nonclose(einconv_output, jax_output)
