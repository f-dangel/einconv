"""Contains test cases for ``einconv/utils``'s functionality."""

from test.utils import make_id

OUTPUT_SIZE_CASES = [
    # default hyperparameters
    {"input_size": 10, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1},
    # nontrivial hyperparameters
    {"input_size": 31, "kernel_size": 3, "stride": 2, "padding": 10, "dilation": 2},
    # nontrivial hyperparameters (non-overlapping patches)
    {"input_size": 51, "kernel_size": 4, "stride": 2, "padding": 10, "dilation": 15},
]

OUTPUT_SIZE_IDS = [make_id(case) for case in OUTPUT_SIZE_CASES]

INPUT_SIZE_CASES = [
    # default hyperparameters
    {
        "output_size": 10,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "output_padding": 0,
    },
    # nontrivial hyperparameters
    {
        "output_size": 11,
        "kernel_size": 3,
        "stride": 2,
        "padding": 10,
        "dilation": 2,
        "output_padding": 1,
    },
    # nontrivial hyperparameters (non-overlapping patches)
    {
        "output_size": 11,
        "kernel_size": 4,
        "stride": 2,
        "padding": 10,
        "dilation": 5,
        "output_padding": 0,
    },
]
INPUT_SIZE_IDS = [make_id(case) for case in INPUT_SIZE_CASES]
