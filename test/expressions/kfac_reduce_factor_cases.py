"""Contains test cases for ``einconv``'s ``kfac_reduce`` implementation."""

from test.conv_module_cases import CONV_1D_MODULE_CASES, CONV_2D_MODULE_CASES
from test.unfold_cases import UNFOLD_1D_CASES, UNFOLD_2D_CASES, UNFOLD_3D_CASES
from test.utils import make_id

KFAC_REDUCE_FACTOR_1D_CASES = []
for case in UNFOLD_1D_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("unfold_kwargs")
    KFAC_REDUCE_FACTOR_1D_CASES.append(case_copy)

for case in CONV_1D_MODULE_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("conv_kwargs")
    if "bias" in case_copy["kfac_reduce_factor_kwargs"]:
        case_copy["kfac_reduce_factor_kwargs"].pop("bias")
    if "padding_mode" in case_copy["kfac_reduce_factor_kwargs"]:
        case_copy["kfac_reduce_factor_kwargs"].pop("padding_mode")
    KFAC_REDUCE_FACTOR_1D_CASES.append(case_copy)
KFAC_REDUCE_FACTOR_1D_IDS = [make_id(case) for case in KFAC_REDUCE_FACTOR_1D_CASES]

KFAC_REDUCE_FACTOR_2D_CASES = []
for case in UNFOLD_2D_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("unfold_kwargs")
    KFAC_REDUCE_FACTOR_2D_CASES.append(case_copy)

for case in CONV_2D_MODULE_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("conv_kwargs")
    if "bias" in case_copy["kfac_reduce_factor_kwargs"]:
        case_copy["kfac_reduce_factor_kwargs"].pop("bias")
    if "padding_mode" in case_copy["kfac_reduce_factor_kwargs"]:
        case_copy["kfac_reduce_factor_kwargs"].pop("padding_mode")
    KFAC_REDUCE_FACTOR_2D_CASES.append(case_copy)
KFAC_REDUCE_FACTOR_2D_IDS = [make_id(case) for case in KFAC_REDUCE_FACTOR_2D_CASES]

KFAC_REDUCE_FACTOR_3D_CASES = []
for case in UNFOLD_3D_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("unfold_kwargs")
    KFAC_REDUCE_FACTOR_3D_CASES.append(case_copy)

# Ignore cases from 3d convolution as they require a lot of time+memory
KFAC_REDUCE_FACTOR_3D_IDS = [make_id(case) for case in KFAC_REDUCE_FACTOR_3D_CASES]
