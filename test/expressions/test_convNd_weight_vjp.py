"""Test einsum expression for weight VJP of N-dimensional convolution."""

from test.expressions.convNd_weight_vjp_cases import (
    WEIGHT_VJP_1D_CASES,
    WEIGHT_VJP_1D_IDS,
    WEIGHT_VJP_2D_CASES,
    WEIGHT_VJP_2D_IDS,
    WEIGHT_VJP_3D_CASES,
    WEIGHT_VJP_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from typing import Dict

from pytest import mark
from torch import device, einsum, manual_seed, rand_like
from torch.autograd import grad
from torch.nn.functional import conv1d, conv2d, conv3d

from einconv.expressions import convNd_weight_vjp


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize(
    "case",
    WEIGHT_VJP_1D_CASES + WEIGHT_VJP_2D_CASES + WEIGHT_VJP_3D_CASES,
    ids=WEIGHT_VJP_1D_IDS + WEIGHT_VJP_2D_IDS + WEIGHT_VJP_3D_IDS,
)
def test_einsum_expression(case: Dict, device: device, simplify: bool):
    """Compare weight JVP of autograd with einsum expression.

    Args:
        case: Dictionary describing the module.
        device: Device to load the module to.
        simplify: Whether to simplify the einsum expression.
    """
    manual_seed(case["seed"])
    kwargs = case["kwargs"]
    weight = case["weight_fn"]().to(device).requires_grad_()
    x = case["input_fn"]().to(device)
    N = x.dim() - 2

    # ground truth
    conv = {1: conv1d, 2: conv2d, 3: conv3d}[N]
    output = conv(x, weight, **kwargs)
    v = rand_like(output)
    (weight_vjp,) = grad(output, weight, v)

    kernel_size = weight.shape[2:]
    equation, operands, shape = convNd_weight_vjp.einsum_expression(
        x, v, kernel_size, **kwargs, simplify=simplify
    )
    ein_weight_vjp = einsum(equation, *operands).reshape(shape)

    report_nonclose(weight_vjp, ein_weight_vjp)
