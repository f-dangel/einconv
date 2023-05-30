"""Contains test cases for ``einconv``'s ``kfac_reduce`` implementation."""

from test.unfold_cases import UNFOLD_1D_CASES, UNFOLD_2D_CASES, UNFOLD_3D_CASES
from test.utils import make_id

KFAC_REDUCE_FACTOR_1D_CASES = []
for case in UNFOLD_1D_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("unfold_kwargs")
    KFAC_REDUCE_FACTOR_1D_CASES.append(case_copy)

KFAC_REDUCE_FACTOR_1D_IDS = [make_id(case) for case in KFAC_REDUCE_FACTOR_1D_CASES]

KFAC_REDUCE_FACTOR_2D_CASES = []
for case in UNFOLD_2D_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("unfold_kwargs")
    KFAC_REDUCE_FACTOR_2D_CASES.append(case_copy)

KFAC_REDUCE_FACTOR_2D_IDS = [make_id(case) for case in KFAC_REDUCE_FACTOR_2D_CASES]

KFAC_REDUCE_FACTOR_3D_CASES = []
for case in UNFOLD_3D_CASES:
    case_copy = {**case}
    case_copy["kfac_reduce_factor_kwargs"] = case_copy.pop("unfold_kwargs")
    KFAC_REDUCE_FACTOR_3D_CASES.append(case_copy)

KFAC_REDUCE_FACTOR_3D_IDS = [make_id(case) for case in KFAC_REDUCE_FACTOR_3D_CASES]
