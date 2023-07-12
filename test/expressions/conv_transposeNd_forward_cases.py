"""Test cases for einsum expression of N-dimensional transpose convolution."""

from test.utils import make_id

from torch import rand

FORWARD_1D_CASES = [
    # no kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10),
        "weight_fn": lambda: rand(3, 4, 5),
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10),
        "weight_fn": lambda: rand(3, 4, 5),
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 10),
        "weight_fn": lambda: rand(4, 2, 5),
        "kernel_size": 5,
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10),
        "weight_fn": lambda: rand(3, 4, 5),
        "kwargs": {"padding": 2},
    },
    # kernel_size, padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10),
        "weight_fn": lambda: rand(3, 4, 5),
        "kwargs": {"padding": (2,), "stride": (3,), "dilation": (1,)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10),
        "weight_fn": lambda: rand(3, 4, 5),
        "kwargs": {"dilation": 2},
    },
]
FORWARD_1D_IDS = [make_id(case) for case in FORWARD_1D_CASES]

# TODO Add test cases for 2d transpose convolution
FORWARD_2D_CASES = []
FORWARD_2D_IDS = [make_id(case) for case in FORWARD_2D_CASES]

# TODO Add test cases for 3d transpose convolution
FORWARD_3D_CASES = []
FORWARD_3D_IDS = [make_id(case) for case in FORWARD_3D_CASES]
