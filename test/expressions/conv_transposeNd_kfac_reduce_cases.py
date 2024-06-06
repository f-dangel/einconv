"""Test cases for einsum expression of input-based TRANSPOSE_KFAC-reduce factor of convolution."""

from test.utils import make_id

from torch import rand

TRANSPOSE_KFAC_REDUCE_1D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 8),
        "kernel_size": 5,
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8),
        "kernel_size": 5,
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 8),
        "kernel_size": 5,
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8),
        "kernel_size": 5,
        "kwargs": {"padding": 2},
    },
    # non-default output padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8),
        "kernel_size": 5,
        "kwargs": {"output_padding": 1, "stride": 2},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8),
        "kernel_size": 5,
        "kwargs": {"dilation": 2},
    },
    # non-default arguments supplied as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8),
        "kernel_size": (3,),
        "kwargs": {
            "padding": (1,),
            "stride": (2,),
            "dilation": (1,),
            "output_padding": (1,),
        },
    },
]
TRANSPOSE_KFAC_REDUCE_1D_IDS = [
    make_id(case) for case in TRANSPOSE_KFAC_REDUCE_1D_CASES
]

TRANSPOSE_KFAC_REDUCE_2D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 8, 7),
        "kernel_size": 5,
        "kwargs": {},
    },
    # non-default kwargs supplied as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8, 7),
        "kernel_size": (5, 3),
        "kwargs": {
            "padding": (1, 2),
            "stride": (2, 3),
            "dilation": (2, 1),
            "output_padding": (1, 2),
        },
    },
]
TRANSPOSE_KFAC_REDUCE_2D_IDS = [
    make_id(case) for case in TRANSPOSE_KFAC_REDUCE_2D_CASES
]

TRANSPOSE_KFAC_REDUCE_3D_CASES = [
    # no kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8, 7, 6),
        "kernel_size": 5,
        "kwargs": {},
    },
    # non-default kwargs supplied as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 8, 7, 6),
        "kernel_size": (5, 3, 2),
        "kwargs": {
            "padding": (0, 1, 2),
            "stride": (3, 2, 1),
            "dilation": (1, 2, 3),
            "output_padding": (2, 0, 0),
        },
    },
]
TRANSPOSE_KFAC_REDUCE_3D_IDS = [
    make_id(case) for case in TRANSPOSE_KFAC_REDUCE_3D_CASES
]
