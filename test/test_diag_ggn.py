"""Tests for ``einconv/diag_ggn``."""

from test.conv_module_cases import (
    CONV_1D_MODULE_CASES,
    CONV_1D_MODULE_IDS,
    CONV_2D_MODULE_CASES,
    CONV_2D_MODULE_IDS,
    CONV_3D_MODULE_CASES,
    CONV_3D_MODULE_IDS,
    conv_module_from_case,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict, Union

import torch
from backpack.utils.conv import extract_weight_diagonal, unfold_input
from pytest import mark, skip
from torch import Tensor, device, einsum, manual_seed, rand
from torch.nn import Conv1d, Conv2d, Conv3d

from einconv.diag_ggn import (
    _conv_diag_ggn_einsum_equation,
    _conv_diag_ggn_einsum_operands,
)

NUM_CLASSES = [1, 3]
NUM_CLASS_IDS = [f"num_classes_{c}" for c in NUM_CLASSES]


def _einsum_diag_ggn(
    layer: Union[Conv1d, Conv2d, Conv3d], inputs: Tensor, sqrt_ggn: Tensor
) -> Tensor:
    """Perform a GGN diagonal extraction using tensor networks.

    Args:
        layer: Convolution layer (hyperparameter info).
        inputs: Input to the convolution layer.
        sqrt_ggn: Matrix square root of the GGN.

    Returns:
        GGN diagonal.
    """
    N = inputs.dim() - 2
    equation = _conv_diag_ggn_einsum_equation(N)
    operands = _conv_diag_ggn_einsum_operands(
        inputs,
        layer.weight,
        sqrt_ggn,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
    )
    return einsum(equation, *operands).flatten(end_dim=1)


def _backpack_diag_ggn(
    layer: Union[Conv1d, Conv2d, Conv3d], inputs: Tensor, sqrt_ggn: Tensor
) -> Tensor:
    """Perform a GGN diagonal extraction using BackPACK.

    Args:
        layer: Convolution layer (hyperparameter info).
        inputs: Input to the convolution layer.
        sqrt_ggn: Matrix square root of the GGN.

    Returns:
        GGN diagonal.
    """
    if isinstance(layer.padding, str):
        skip("PyTorch's unfold does notupport string padding.")
    else:
        unfolded_inputs = unfold_input(layer, inputs)
        return extract_weight_diagonal(layer, unfolded_inputs, sqrt_ggn, sum_batch=True)


@mark.parametrize("num_classes", NUM_CLASSES, ids=NUM_CLASS_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize(
    "case",
    CONV_1D_MODULE_CASES + CONV_2D_MODULE_CASES + CONV_3D_MODULE_CASES,
    ids=CONV_1D_MODULE_IDS + CONV_2D_MODULE_IDS + CONV_3D_MODULE_IDS,
)
def test_diag_ggn(
    case: Dict, device: device, num_classes: int, dtype: Union[torch.dtype, None] = None
):
    """Compare GGN diagonal of BackPACK with tensor network implementation.

    Args:
        case: Dictionary describing the module.
        device: Device to load the module to.
        num_classes: Number of classes or MC samples for the GGN.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).
    """
    manual_seed(case["seed"])
    x = case["input_fn"]().to(device)
    N = x.dim() - 2

    layer = conv_module_from_case(N, case, device, dtype=dtype)
    output_shape = layer(x).shape
    sqrt_ggn = rand(num_classes, *output_shape, device=device, dtype=dtype)

    ein_diag_ggn = _einsum_diag_ggn(layer, x, sqrt_ggn)
    assert ein_diag_ggn.shape == layer.weight.shape
    backpack_diag_ggn = _backpack_diag_ggn(layer, x, sqrt_ggn)

    report_nonclose(ein_diag_ggn, backpack_diag_ggn)
