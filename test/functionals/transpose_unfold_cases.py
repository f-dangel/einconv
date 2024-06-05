"""Contains cases to test transpose convolution unfolding functional."""

from test.utils import make_id

from torch import rand

TRANSPOSE_UNFOLD_1D_CASES = [
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 1,
        "kwargs": {},
    },
    {
        "seed": 0,
        "input_fn": lambda: rand(5, 2, 9),
        "kernel_size": 3,
        "kwargs": {"stride": 2, "padding": 1, "output_padding": 1, "dilation": 2},
    },
]

TRANSPOSE_UNFOLD_1D_IDS = [make_id(problem) for problem in TRANSPOSE_UNFOLD_1D_CASES]

TRANSPOSE_UNFOLD_2D_CASES = [
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10, 10),
        "kernel_size": 1,
        "kwargs": {},
    },
    {
        "seed": 0,
        "input_fn": lambda: rand(5, 2, 9, 7),
        "kernel_size": (3, 2),
        "kwargs": {
            "stride": (2, 1),
            "padding": (1, 0),
            "output_padding": (1, 0),
            "dilation": (2, 2),
        },
    },
]

TRANSPOSE_UNFOLD_2D_IDS = [make_id(problem) for problem in TRANSPOSE_UNFOLD_2D_CASES]

TRANSPOSE_UNFOLD_3D_CASES = [
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 10, 10, 10),
        "kernel_size": 1,
        "kwargs": {},
    },
    {
        "seed": 0,
        "input_fn": lambda: rand(5, 2, 9, 7, 6),
        "kernel_size": (3, 2, 4),
        "kwargs": {
            "stride": (2, 1, 3),
            "padding": (1, 0, 2),
            "output_padding": (1, 0, 2),
            "dilation": (2, 2, 1),
        },
    },
]

TRANSPOSE_UNFOLD_3D_IDS = [make_id(problem) for problem in TRANSPOSE_UNFOLD_3D_CASES]
