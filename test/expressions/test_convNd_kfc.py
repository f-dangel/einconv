"""Tests ``einconv.expressions.convNd_kfc``."""

from test.expressions.convNd_kfc_cases import (
    KFC_1D_CASES,
    KFC_1D_IDS,
    KFC_2D_CASES,
    KFC_2D_IDS,
    KFC_3D_CASES,
    KFC_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Dict

import torch
import unfoldNd
from einops import rearrange
from pytest import mark
from torch import einsum

from einconv.expressions import convNd_kfc


@mark.parametrize(
    "case",
    KFC_1D_CASES + KFC_2D_CASES + KFC_3D_CASES,
    ids=KFC_1D_IDS + KFC_2D_IDS + KFC_3D_IDS,
)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_einsum_expression(case: Dict, device: torch.device):
    """Compare einsum expression of KFC with implementation via ``unfoldNd``.

    Args:
        case: Dictionary describing the test case.
        device: Device to execute the test on.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    kwargs = case["kwargs"]

    torch.manual_seed(seed)
    x = input_fn().to(device)

    # ground truth
    unfold_kwargs = {key: value for key, value in kwargs.items() if key != "groups"}
    unfolded_x = unfoldNd.unfoldNd(x, kernel_size, **unfold_kwargs)
    batch_size = unfolded_x.shape[0]
    groups = kwargs.get("groups", 1)
    unfolded_x = rearrange(unfolded_x, "n (g c_in_k) ... -> n g c_in_k ...", g=groups)
    kfc_unfold = einsum("ngik,ngjk->gij", unfolded_x, unfolded_x) / batch_size

    equation, operands, shape = convNd_kfc.einsum_expression(x, kernel_size, **kwargs)
    kfc_einconv = einsum(equation, *operands).reshape(shape)

    report_nonclose(kfc_unfold, kfc_einconv, rtol=5e-5)
