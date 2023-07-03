"""Tests for ``einconv.modules.unfold``."""

from test.modules.unfold_cases import (
    UNFOLD_1D_CASES,
    UNFOLD_1D_IDS,
    UNFOLD_2D_CASES,
    UNFOLD_2D_IDS,
    UNFOLD_3D_CASES,
    UNFOLD_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from typing import Dict

import torch
import unfoldNd
from pytest import mark

from einconv import modules


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize(
    "case",
    UNFOLD_1D_CASES + UNFOLD_2D_CASES + UNFOLD_3D_CASES,
    ids=UNFOLD_1D_IDS + UNFOLD_2D_IDS + UNFOLD_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_UnfoldNd(case: Dict, device: torch.device, simplify: bool):
    """Compare unfold module with that of the ``unfoldNd`` package.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
        simplify: Whether to use a simplified einsum expression.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    kwargs = case["kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    result_unfold = unfoldNd.UnfoldNd(kernel_size, **kwargs).to(device)(inputs)
    result_einconv = modules.UnfoldNd(kernel_size, **kwargs, simplify=simplify).to(
        device
    )(inputs)

    report_nonclose(result_unfold, result_einconv)
