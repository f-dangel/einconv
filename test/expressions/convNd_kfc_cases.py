"""Test cases for einsum expression of input-based KFC factor for convolution."""

from test.utils import make_id

from torch import rand

KFC_1D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 5,
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 5,
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50),
        "kernel_size": 5,
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 5,
        "kwargs": {"padding": 2},
    },
    # kernel_size, padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": (5,),
        "kwargs": {"padding": (2,), "stride": (3,), "dilation": (1,)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 5,
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 20),
        "kernel_size": 5,
        "kwargs": {"padding": "same"},
    },
]
KFC_1D_IDS = [make_id(case) for case in KFC_1D_CASES]

KFC_2D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": 5,
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": 5,
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50, 40),
        "kernel_size": 5,
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": 6,
        "kwargs": {"padding": 2},
    },
    # kernel_size, padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (5, 3),
        "kwargs": {"padding": (2, 1), "stride": (3, 2), "dilation": (1, 2)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (5, 6),
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (5, 3),
        "kwargs": {"padding": "same", "dilation": (1, 2)},
    },
]
KFC_2D_IDS = [make_id(case) for case in KFC_2D_CASES]

KFC_3D_CASES = [
    # no kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": 5,
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": 5,
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50, 40, 30),
        "kernel_size": 5,
        "kwargs": {"stride": 3, "groups": 2},
    },
    # tuple-valued kernel_size and non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (5, 6, 7),
        "kwargs": {"padding": 2},
    },
    # kernel_size, padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (5, 3, 2),
        "kwargs": {"padding": (2, 1, 0), "stride": (3, 2, 1), "dilation": (1, 2, 3)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 40),
        "kernel_size": (5, 6, 7),
        "weight_fn": lambda: rand(4, 3, 5, 6, 7),
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (5, 3, 2),
        "kwargs": {"padding": "same", "dilation": (1, 2, 2)},
    },
]
KFC_3D_IDS = [make_id(case) for case in KFC_3D_CASES]
