"""Tests ``einconv.expressions.conv_transposeNd``."""

from test.expressions.conv_transposeNd_forward_cases import (
    FORWARD_1D_CASES,
    FORWARD_1D_IDS,
    FORWARD_2D_CASES,
    FORWARD_2D_IDS,
    FORWARD_3D_CASES,
    FORWARD_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from typing import Dict

import torch
from pytest import mark
from torch import einsum
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d

from einconv.expressions import conv_transposeNd_forward


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize(
    "case",
    FORWARD_1D_CASES + FORWARD_2D_CASES + FORWARD_3D_CASES,
    ids=FORWARD_1D_IDS + FORWARD_2D_IDS + FORWARD_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_einsum_expression(case: Dict, device: torch.device, simplify: bool):
    """Compare einsum expression of N-dimensional transpose convolution with PyTorch.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
        simplify: Whether to simplify the einsum expression.
    """
    torch.manual_seed(case["seed"])
    x = case["input_fn"]().to(device)
    weight = case["weight_fn"]().to(device)
    kwargs = case["kwargs"]

    # ground truth
    N = x.dim() - 2
    conv_t = {1: conv_transpose1d, 2: conv_transpose2d, 3: conv_transpose3d}[N]
    out_torch = conv_t(x, weight, bias=None, **kwargs)

    equation, operands, shape = conv_transposeNd_forward.einsum_expression(
        x, weight, **kwargs, simplify=simplify
    )
    out_einconv = einsum(equation, *operands).reshape(shape)

    report_nonclose(out_torch, out_einconv, rtol=5e-5)
