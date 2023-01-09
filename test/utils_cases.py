"""Contains test cases for ``einconv/utils``'s functionality."""

from test.utils import make_id

OUTPUT_SIZE_CASES = [
    # default hyperparameters
    {
        "input_size": 10,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
    },
    # nontrivial hyperparameters
    {
        "input_size": 31,
        "kernel_size": 3,
        "stride": 2,
        "padding": 10,
        "dilation": 2,
    },
    # nontrivial hyperparameters (non-overlapping patches)
    {
        "input_size": 51,
        "kernel_size": 4,
        "stride": 2,
        "padding": 10,
        "dilation": 15,
    },
]

OUTPUT_SIZE_IDS = [make_id(case) for case in OUTPUT_SIZE_CASES]
