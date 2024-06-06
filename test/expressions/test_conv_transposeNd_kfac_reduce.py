"""Tests ``einconv.expressions.conv_transposeNd_kfac_reduce``."""

from test.expressions.conv_transposeNd_kfac_reduce_cases import (
    TRANSPOSE_KFAC_REDUCE_1D_CASES,
    TRANSPOSE_KFAC_REDUCE_1D_IDS,
    TRANSPOSE_KFAC_REDUCE_2D_CASES,
    TRANSPOSE_KFAC_REDUCE_2D_IDS,
    TRANSPOSE_KFAC_REDUCE_3D_CASES,
    TRANSPOSE_KFAC_REDUCE_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from typing import Dict

import unfoldNd
from einops import rearrange
from pytest import mark
from torch import device, einsum, manual_seed

from einconv.expressions import conv_transposeNd_kfac_reduce


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize(
    "case",
    TRANSPOSE_KFAC_REDUCE_1D_CASES
    + TRANSPOSE_KFAC_REDUCE_2D_CASES
    + TRANSPOSE_KFAC_REDUCE_3D_CASES,
    ids=TRANSPOSE_KFAC_REDUCE_1D_IDS
    + TRANSPOSE_KFAC_REDUCE_2D_IDS
    + TRANSPOSE_KFAC_REDUCE_3D_IDS,
)
@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_einsum_expression(case: Dict, dev: device, simplify: bool):
    """Compare einsum expression of KFAC reduce with implementation via unfolding.

    Args:
        case: Dictionary describing the test case.
        dev: Device to execute the test on.
        simplify: Whether to simplify the einsum expression.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    kwargs = case["kwargs"]

    manual_seed(seed)
    x = input_fn().to(dev)
    batch_size = x.shape[0]

    # ground truth
    unfold_kwargs = {key: value for key, value in kwargs.items() if key != "groups"}
    unfolded_x = unfoldNd.unfold_transposeNd(x, kernel_size, **unfold_kwargs)
    avg_unfolded_x = unfolded_x.mean(dim=-1)
    groups = kwargs.get("groups", 1)
    avg_unfolded_x = rearrange(avg_unfolded_x, "n (g c_in_k) -> n g c_in_k", g=groups)
    kfac_unfold = einsum("ngi,ngj->gij", avg_unfolded_x, avg_unfolded_x) / batch_size

    equation, operands, shape = conv_transposeNd_kfac_reduce.einsum_expression(
        x, kernel_size, **kwargs, simplify=simplify
    )
    kfac_einconv = einsum(equation, *operands).reshape(shape)

    report_nonclose(kfac_unfold, kfac_einconv)
