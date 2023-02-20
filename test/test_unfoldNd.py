"""Contains tests for ``einconv/unfoldnd.py``."""

from test.unfold_cases import (
    UNFOLD_1D_CASES,
    UNFOLD_1D_IDS,
    UNFOLD_2D_CASES,
    UNFOLD_2D_IDS,
    UNFOLD_3D_CASES,
    UNFOLD_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict

import torch
import unfoldNd
from pytest import mark

import einconv


@mark.parametrize(
    "case",
    UNFOLD_1D_CASES + UNFOLD_2D_CASES + UNFOLD_3D_CASES,
    ids=UNFOLD_1D_IDS + UNFOLD_2D_IDS + UNFOLD_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_unfoldNd(case: Dict, device: torch.device):
    """Compare ``einconv.unfoldnd.unfoldNd`` with ``unfoldNd`` package.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    unfold_kwargs = case["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    result_unfold = unfoldNd.unfoldNd(inputs, kernel_size, **unfold_kwargs)
    result_einconv = einconv.unfoldNd(inputs, kernel_size, **unfold_kwargs)

    report_nonclose(result_unfold, result_einconv)


@mark.parametrize(
    "case",
    UNFOLD_1D_CASES + UNFOLD_2D_CASES + UNFOLD_3D_CASES,
    ids=UNFOLD_1D_IDS + UNFOLD_2D_IDS + UNFOLD_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_UnfoldNd(case: Dict, device: torch.device):
    """Compare ``einconv.unfoldnd.UnfoldNd`` with ``unfoldNd`` package.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    unfold_kwargs = case["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    result_unfold = unfoldNd.UnfoldNd(kernel_size, **unfold_kwargs).to(device)(inputs)
    result_einconv = einconv.UnfoldNd(kernel_size, **unfold_kwargs).to(device)(inputs)

    report_nonclose(result_unfold, result_einconv)
