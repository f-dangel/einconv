"""Contains test cases for ``einconv``'s module implementation of convolution."""

from test.utils import make_id

from torch import rand

CONV_1D_MODULE_CASES = [
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 50),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "conv_kwargs": {
            "bias": False,
        },
    },
    # non-default kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 50),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "conv_kwargs": {
            "stride": 2,
            "padding": 1,
            "dilation": 2,
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
    # non-default kwargs as tuples
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 6, 99),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5,),
        # stride, padding, dilation, groups, padding_mode, bias
        "conv_kwargs": {
            "stride": (2,),
            "padding": (1,),
            "dilation": (2,),
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
]

CONV_1D_MODULE_IDS = [make_id(case) for case in CONV_1D_MODULE_CASES]
