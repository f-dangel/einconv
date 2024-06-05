"""Contains tests for ``einconv/expressions/conv_transposeNd_unfold.py``."""

from test.functionals.transpose_unfold_cases import (
    TRANSPOSE_UNFOLD_1D_CASES,
    TRANSPOSE_UNFOLD_1D_IDS,
    TRANSPOSE_UNFOLD_2D_CASES,
    TRANSPOSE_UNFOLD_2D_IDS,
    TRANSPOSE_UNFOLD_3D_CASES,
    TRANSPOSE_UNFOLD_3D_IDS,
)
from test.utils import DEVICE_IDS, DEVICES, SIMPLIFIES, SIMPLIFY_IDS, report_nonclose
from typing import Dict

from einops import einsum, rearrange
from pytest import mark
from torch import (
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    device,
    manual_seed,
    rand,
)

from einconv.functionals import unfoldNd_transpose
from einconv.utils import _tuple


@mark.parametrize("simplify", SIMPLIFIES, ids=SIMPLIFY_IDS)
@mark.parametrize(
    "case",
    TRANSPOSE_UNFOLD_1D_CASES + TRANSPOSE_UNFOLD_2D_CASES + TRANSPOSE_UNFOLD_3D_CASES,
    ids=TRANSPOSE_UNFOLD_1D_IDS + TRANSPOSE_UNFOLD_2D_IDS + TRANSPOSE_UNFOLD_3D_IDS,
)
@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_unfoldNd_transpose(case: Dict, dev: device, simplify: bool):
    """Compare transpose convolution via input unfolding with built-in one.

    Args:
        case: Dictionary describing the test case.
        dev: Device to execute the test on.
        simplify: Whether to use a simplified einsum expression.
    """
    seed = case["seed"]
    input_fn = case["input_fn"]
    kernel_size = case["kernel_size"]
    kwargs = case["kwargs"]

    manual_seed(seed)
    inputs = input_fn().to(dev)
    N = inputs.ndim - 2
    batch_size, C_out = inputs.shape[:2]
    C_in, G = 3, 1  # hard-coded for now
    t_kernel_size = _tuple(kernel_size, N)
    weight = rand(C_out, C_in // G, *t_kernel_size).to(dev)

    # ground truth: PyTorch's built-in transpose convolution
    conv_func = {1: conv_transpose1d, 2: conv_transpose2d, 3: conv_transpose3d}[N]
    result = conv_func(inputs, weight, **kwargs)

    # transpose convolution via matrix-multiplication perspective using unfolded input
    # and matricized kernel
    inputs_unfolded = unfoldNd_transpose(
        inputs, kernel_size, **kwargs, simplify=simplify
    )
    k_s = " ".join([f"k{i}" for i in range(N)])
    weight_mat = rearrange(weight, f"c_out c_in {k_s} -> c_in (c_out {k_s})")
    result_mat = einsum(
        inputs_unfolded, weight_mat, "n c_out_k i, c_in c_out_k -> n c_in i"
    )

    report_nonclose(result, result_mat.reshape(batch_size, C_in, *result.shape[2:]))
