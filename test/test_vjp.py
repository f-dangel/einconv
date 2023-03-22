"""Tests for einconv/vjp``."""

from test.conv_module_cases import (
    CONV_1D_MODULE_CASES,
    CONV_1D_MODULE_IDS,
    CONV_2D_MODULE_CASES,
    CONV_2D_MODULE_IDS,
    CONV_3D_MODULE_CASES,
    CONV_3D_MODULE_IDS,
    CONV_4D_MODULE_CASES,
    CONV_4D_MODULE_IDS,
    CONV_5D_MODULE_CASES,
    CONV_5D_MODULE_IDS,
    CONV_6D_MODULE_CASES,
    CONV_6D_MODULE_IDS,
    conv_module_from_case,
    einconv_module_from_case,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict, Union

import torch
from pytest import mark
from torch import Tensor, device, einsum, manual_seed, rand_like
from torch.autograd import grad
from torch.nn import Conv1d, Conv2d, Conv3d

from einconv.einconvnd import EinconvNd
from einconv.vjp import (
    _conv_weight_vjp_einsum_equation,
    _conv_weight_vjp_einsum_operands,
)


def _einsum_weight_vjp(
    layer: Union[Conv1d, Conv2d, Conv3d, EinconvNd], inputs: Tensor, grad_output: Tensor
) -> Tensor:
    """Perform a VJP using tensor networks.

    Args:
        layer: Convolution layer (hyperparameter info).
        inputs: Input to the convolution layer.
        grad_output: Vector for weight VJP.

    Returns:
        Result of the weight VJP.
    """
    N = inputs.dim() - 2
    equation = _conv_weight_vjp_einsum_equation(N)
    operands = _conv_weight_vjp_einsum_operands(
        inputs,
        layer.weight,
        grad_output,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
    )
    return einsum(equation, *operands).flatten(end_dim=1)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize(
    "case",
    CONV_1D_MODULE_CASES
    + CONV_2D_MODULE_CASES
    + CONV_3D_MODULE_CASES
    + CONV_4D_MODULE_CASES
    + CONV_5D_MODULE_CASES
    + CONV_6D_MODULE_CASES,
    ids=CONV_1D_MODULE_IDS
    + CONV_2D_MODULE_IDS
    + CONV_3D_MODULE_IDS
    + CONV_4D_MODULE_IDS
    + CONV_5D_MODULE_IDS
    + CONV_6D_MODULE_IDS,
)
def test_conv_weight_vjp(
    case: Dict, device: device, dtype: Union[torch.dtype, None] = None
):
    """Compare weight JVP of PyTorch convolution layers with einsum expression.

    Args:
        case: Dictionary describing the module.
        device: Device to load the module to.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)
    N = x.dim() - 2

    make_layer = einconv_module_from_case if N > 3 else conv_module_from_case
    layer = make_layer(N, case, device, dtype=dtype)
    output = layer(x)
    grad_output = rand_like(output)

    (weight_vjp,) = grad(output, layer.weight, grad_output)
    ein_weight_vjp = _einsum_weight_vjp(layer, x, grad_output)

    report_nonclose(weight_vjp, ein_weight_vjp)
