"""Contains tests for ``einconv/kfac_reduce.py``."""

from test.kfac_reduce_factor_cases import (
    KFAC_REDUCE_FACTOR_1D_CASES,
    KFAC_REDUCE_FACTOR_1D_IDS,
    KFAC_REDUCE_FACTOR_2D_CASES,
    KFAC_REDUCE_FACTOR_2D_IDS,
    KFAC_REDUCE_FACTOR_3D_CASES,
    KFAC_REDUCE_FACTOR_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict

import torch
import unfoldNd
from pytest import mark

from einconv.kfac_reduce import kfac_reduce_factor


@mark.parametrize(
    "case",
    KFAC_REDUCE_FACTOR_1D_CASES
    + KFAC_REDUCE_FACTOR_2D_CASES
    + KFAC_REDUCE_FACTOR_3D_CASES,
    ids=KFAC_REDUCE_FACTOR_1D_IDS
    + KFAC_REDUCE_FACTOR_2D_IDS
    + KFAC_REDUCE_FACTOR_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_kfac_reduce_factor(case: Dict, device: torch.device):
    """Compare ``einconv.kfac_reduce.kfac_reduce_factor`` with ``unfoldNd`` package.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    kfac_reduce_factor_kwargs = case["kfac_reduce_factor_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)
    batch_size = inputs.shape[0]

    unfolded_input = unfoldNd.unfoldNd(inputs, kernel_size, **kfac_reduce_factor_kwargs)
    output_size = unfolded_input.shape[-1]

    result_unfold = torch.einsum("nik,njk->ij", unfolded_input, unfolded_input) / (
        batch_size * output_size**2
    )
    result_einconv = kfac_reduce_factor(
        inputs, kernel_size, **kfac_reduce_factor_kwargs
    )

    report_nonclose(result_unfold, result_einconv)
