"""Test cases for functional of convolution."""

from test.utils import make_id

from torch import rand

CONV_1D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 50),
        # (out_channels, in_channels // groups, kernel_size)
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: None,
        # stride, padding, dilation, groups
        "kwargs": {},
    },
    # non-default stride, bias
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "kwargs": {"stride": 2},
    },
    # non-default stride, bias, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50),
        "weight_fn": lambda: rand(6, 2, 5),
        "bias_fn": lambda: rand(6),
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default bias, padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": (2,), "stride": (3,), "dilation": (1,)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: None,
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 20),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": "same"},
    },
]
CONV_1D_IDS = [make_id(case) for case in CONV_1D_CASES]

CONV_2D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 50, 40),
        # (out_channels, in_channels // groups, kernel_size_h, kernel_size_w)
        "weight_fn": lambda: rand(4, 3, 5, 5),
        "bias_fn": lambda: None,
        # stride, padding, dilation, groups
        "kwargs": {},
    },
    # non-default stride, bias
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 5),
        "bias_fn": lambda: rand(4),
        "kwargs": {"stride": 2},
    },
    # non-default stride, bias, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50, 40),
        "weight_fn": lambda: rand(6, 2, 5, 5),
        "bias_fn": lambda: rand(6),
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default bias, padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 6),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 3),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": (2, 1), "stride": (3, 2), "dilation": (1, 2)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 6),
        "bias_fn": lambda: None,
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 3),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": "same", "dilation": (1, 2)},
    },
]
CONV_2D_IDS = [make_id(case) for case in CONV_2D_CASES]

CONV_3D_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        # (out_channels, in_channels // groups, kernel_size_h, kernel_size_w)
        "weight_fn": lambda: rand(4, 3, 5, 5, 5),
        "bias_fn": lambda: None,
        # stride, padding, dilation, groups
        "kwargs": {},
    },
    # non-default stride, bias
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 5, 5),
        "bias_fn": lambda: rand(4),
        "kwargs": {"stride": 2},
    },
    # non-default stride, bias, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50, 40, 30),
        "weight_fn": lambda: rand(6, 2, 5, 5, 5),
        "bias_fn": lambda: rand(6),
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default bias, padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 6, 7),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 3, 2),
        "bias_fn": lambda: rand(4),
        "kwargs": {
            "padding": (2, 1, 0),
            "stride": (3, 2, 1),
            "dilation": (1, 2, 3),
        },
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 40),
        "weight_fn": lambda: rand(4, 3, 5, 6, 7),
        "bias_fn": lambda: None,
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 3, 2),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": "same", "dilation": (1, 2, 2)},
    },
]
CONV_3D_IDS = [make_id(case) for case in CONV_3D_CASES]

CONV_4D_CASES = [
    # no kwargs, non-trivial bias
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 25, 20, 15, 10),
        # (out_channels, in_channels // groups, kernel_size_h, kernel_size_w)
        "weight_fn": lambda: rand(4, 3, 5, 5, 5, 5),
        "bias_fn": lambda: rand(4),
        # stride, padding, dilation, groups
        "kwargs": {},
    },
    # non-trivial kwargs, non-trivial bias
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 40, 35, 30, 25),
        "weight_fn": lambda: rand(4, 1, 5, 5, 4, 3),
        "bias_fn": lambda: rand(4),
        "kwargs": {
            "padding": (2, 1, 0, 1),
            "stride": (3, 2, 1, 2),
            "dilation": (3, 2, 3, 1),
            "groups": 2,
        },
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 10, 8, 6, 5),
        "weight_fn": lambda: rand(4, 1, 5, 5, 4, 3),
        "bias_fn": lambda: rand(4),
        "kwargs": {"padding": "same", "dilation": (2, 2, 2, 1), "groups": 2},
    },
]
CONV_4D_IDS = [make_id(case) for case in CONV_4D_CASES]
