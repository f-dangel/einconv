"""Contains tests for ``einconv/kfc.py``."""

from test.kfc_factor_cases import (
    KFC_FACTOR_1D_CASES,
    KFC_FACTOR_1D_IDS,
    KFC_FACTOR_2D_CASES,
    KFC_FACTOR_2D_IDS,
    KFC_FACTOR_3D_CASES,
    KFC_FACTOR_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict

import torch
import unfoldNd
from pytest import mark

import einconv


@mark.parametrize(
    "case",
    KFC_FACTOR_1D_CASES + KFC_FACTOR_2D_CASES + KFC_FACTOR_3D_CASES,
    ids=KFC_FACTOR_1D_IDS + KFC_FACTOR_2D_IDS + KFC_FACTOR_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_kfc_factor(case: Dict, device: torch.device):
    """Compare ``einconv.kfc.kfc_factor`` with ``unfoldNd`` package.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    kfc_factor_kwargs = case["kfc_factor_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    unfold_kwargs = {
        key: value for key, value in kfc_factor_kwargs.items() if key != "groups"
    }
    unfolded_input = unfoldNd.unfoldNd(inputs, kernel_size, **unfold_kwargs)
    batch_size, in_channels_times_k, output_size = unfolded_input.shape
    groups = kfc_factor_kwargs.get("groups", 1)
    unfolded_input = unfolded_input.reshape(
        batch_size, groups, in_channels_times_k // groups, output_size
    )
    result_unfold = (
        torch.einsum("ngik,ngjk->gij", unfolded_input, unfolded_input) / batch_size
    )
    result_einconv = einconv.kfc_factor(inputs, kernel_size, **kfc_factor_kwargs)

    report_nonclose(result_unfold, result_einconv, rtol=5e-5)
