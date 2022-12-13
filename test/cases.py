"""Contains test cases for ``einconv``."""

from test.utils import get_available_devices, make_id

from torch import rand

DEVICES = get_available_devices()
DEVICE_IDS = [f"device_{dev}" for dev in DEVICES]

CONV_1D_FUNCTIONAL_CASES = [
    # no kwargs
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 50),
        # (out_channels, in_channels // groups, kernel_size)
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: None,
        # stride, padding, dilation, groups
        "conv_kwargs": {},
    },
    # non-default stride, bias
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "conv_kwargs": {"stride": 2},
    },
    # non-default stride, bias, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50),
        "weight_fn": lambda: rand(6, 2, 5),
        "bias_fn": lambda: rand(6),
        "conv_kwargs": {"stride": 3, "groups": 2},
    },
    # non-default bias, padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "conv_kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "conv_kwargs": {"padding": (2,), "stride": (3,), "dilation": (1,)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: None,
        "conv_kwargs": {"dilation": 2},
    },
]
CONV_1D_FUNCTIONAL_IDS = [make_id(case) for case in CONV_1D_FUNCTIONAL_CASES]
