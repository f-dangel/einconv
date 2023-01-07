"""Contains test cases for ``einconv``'s index pattern computation."""

from test.utils import make_id

INDEX_PATTERN_CASES = [
    # required arguments
    {"input_size": 13, "kernel_size": 2},
    # with nontrivial stride
    {"input_size": 15, "kernel_size": 3, "stride": 2},
    # with nontrivial padding
    {"input_size": 11, "kernel_size": 4, "padding": 2},
    # with nontrivial large padding
    {"input_size": 11, "kernel_size": 4, "padding": 6},
    # with string-valued padding
    {"input_size": 11, "kernel_size": 4, "padding": "valid"},
    # with nontrivial dilation
    {"input_size": 20, "kernel_size": 3, "stride": 2},
    # mixed non-default hyperparameters
    {"input_size": 91, "kernel_size": 4, "stride": 2, "padding": 2, "dilation": 3},
    # mixed non-default hyperparameters and large padding
    {"input_size": 91, "kernel_size": 4, "stride": 2, "padding": 20, "dilation": 3},
]

INDEX_PATTERN_IDS = [make_id(case) for case in INDEX_PATTERN_CASES]
