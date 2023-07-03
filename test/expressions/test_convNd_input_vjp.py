"""Test ``einconv.expressions.convNd_input_vjp``."""

from test.expressions.convNd_input_vjp_cases import (
    INPUT_VJP_1D_CASES,
    INPUT_VJP_1D_IDS,
    INPUT_VJP_2D_CASES,
    INPUT_VJP_2D_IDS,
    INPUT_VJP_3D_CASES,
    INPUT_VJP_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from typing import Dict

from pytest import mark
from torch import device, einsum, manual_seed, rand_like
from torch.autograd import grad
from torch.nn.functional import conv1d, conv2d, conv3d

from einconv.expressions import convNd_input_vjp


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize(
    "case",
    INPUT_VJP_1D_CASES + INPUT_VJP_2D_CASES + INPUT_VJP_3D_CASES,
    ids=INPUT_VJP_1D_IDS + INPUT_VJP_2D_IDS + INPUT_VJP_3D_IDS,
)
def test_einsum_expression(case: Dict, device: device, simplify: bool):
    """Compare input JVP of autograd with einsum expression.

    Args:
        case: Dictionary describing the module.
        device: Device to load the module to.
        simplify: Whether to simplify the einsum expression.
    """
    manual_seed(case["seed"])
    kwargs = case["kwargs"]
    weight = case["weight_fn"]().to(device)
    x = case["input_fn"]().to(device).requires_grad_()
    N = x.dim() - 2

    # ground truth
    conv = {1: conv1d, 2: conv2d, 3: conv3d}[N]
    output = conv(x, weight, **kwargs)
    v = rand_like(output)
    (x_vjp,) = grad(output, x, v)

    input_sizes = x.shape[2:]
    equation, operands, shape = convNd_input_vjp.einsum_expression(
        weight, v, input_sizes, **kwargs, simplify=simplify
    )
    ein_x_vjp = einsum(equation, *operands).reshape(shape)

    report_nonclose(x_vjp, ein_x_vjp, atol=5e-7)
