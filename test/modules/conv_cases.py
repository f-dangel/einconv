"""Test cases for convolution module."""

from test.utils import make_id
from typing import Dict, Union

import torch
from torch import rand
from torch.nn import Conv1d, Conv2d, Conv3d

from einconv.modules import ConvNd

CONV_1D_CASES = [
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels)
        "input_fn": lambda: rand(2, 3, 50),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {"bias": False},
    },
    # non-default kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
        "kwargs": {
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
        "input_fn": lambda: rand(2, 6, 99),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5,),
        "kwargs": {
            "stride": (2,),
            "padding": (1,),
            "dilation": (2,),
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
    # padding specified as str
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 6, 71),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5,),
        "kwargs": {
            "padding": "same",
            "dilation": (2,),
            "groups": 3,
            "bias": True,
        },
    },
]

CONV_1D_IDS = [make_id(case) for case in CONV_1D_CASES]


CONV_2D_CASES = [
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 50, 40),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {
            "bias": False,
        },
    },
    # non-default kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 45),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
        "kwargs": {
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
        "input_fn": lambda: rand(2, 6, 99, 88),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5, 4),
        "kwargs": {
            "stride": (2, 3),
            "padding": (1, 0),
            "dilation": (2, 1),
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
    # padding specified as str
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 6, 63, 52),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5, 4),
        "kwargs": {
            "padding": "same",
            "dilation": (1, 2),
            "groups": 3,
            "bias": True,
        },
    },
]
CONV_2D_IDS = [make_id(case) for case in CONV_2D_CASES]

CONV_3D_CASES = [
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_d, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 20, 15, 10),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {
            "bias": False,
        },
    },
    # non-default kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 45, 40),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
        "kwargs": {
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
        "input_fn": lambda: rand(2, 6, 99, 88, 77),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5, 4, 3),
        "kwargs": {
            "stride": (2, 3, 2),
            "padding": (1, 0, 2),
            "dilation": (2, 1, 3),
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
    # padding specified as str
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 6, 37, 41, 28),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5, 4, 3),
        "kwargs": {
            "padding": "same",
            "dilation": (2, 1, 3),
            "groups": 3,
            "bias": True,
        },
    },
]
CONV_3D_IDS = [make_id(case) for case in CONV_3D_CASES]

CONV_4D_CASES = [
    # no kwargs (bias enabled)
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 25, 20, 15, 10),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {},
    },
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 25, 20, 15, 10),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {
            "bias": False,
        },
    },
    # non-default kwargs as tuples
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 40, 35, 30, 25),
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": (5, 5, 4, 3),
        "kwargs": {
            "stride": (2, 3, 2, 1),
            "padding": (2, 0, 2, 1),
            "dilation": (2, 1, 3, 2),
            "groups": 2,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
    # padding specified as str
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 21, 26, 33, 17),
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": (5, 5, 4, 3),
        "kwargs": {
            "padding": "same",
            "dilation": (2, 1, 3, 2),
            "groups": 2,
            "bias": True,
        },
    },
]
CONV_4D_IDS = [make_id(case) for case in CONV_4D_CASES]

CONV_5D_CASES = [
    # no kwargs (bias enabled)
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 15, 10, 8, 6, 5),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 3,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {
            "bias": False,
        },
    },
    # non-default kwargs as tuples
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 20, 15, 10, 15, 20),
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": (5, 4, 2, 3, 4),
        "kwargs": {
            "stride": (4, 3, 1, 2, 4),
            "padding": (2, 0, 1, 1, 3),
            "groups": 2,
            "padding_mode": "zeros",
            "bias": False,
        },
    },
    # non-default kwargs as tuples, output of third-party PyTorch
    # implementation (https://github.com/pvjosue/pytorch_convNd) disagrees.
    # This test case is evidence that there might be a bug in the third-party
    # implementation, because einconv and JAX agree, but einconv and
    # third-party disagree.
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 20, 15, 10, 15, 20),
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": (5, 4, 2, 3, 4),
        "kwargs": {
            "stride": (4, 3, 1, 2, 4),
            "padding": (0, 1, 0, 0, 0),
            "groups": 2,
            "padding_mode": "zeros",
            "bias": False,
        },
    },
]
CONV_5D_IDS = [make_id(case) for case in CONV_5D_CASES]

CONV_6D_CASES = [
    # no kwargs (bias enabled)
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 15, 10, 8, 6, 5, 5),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 3,
        # stride, padding, dilation, groups, padding_mode, bias
        "kwargs": {
            "bias": False,
        },
    },
    # non-default kwargs as tuples
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 2, 10, 10, 10, 10, 5, 5),
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": (3, 2, 3, 4, 2, 2),
        "kwargs": {
            "stride": (3, 2, 2, 3, 3, 2),
            "padding": (2, 0, 0, 2, 1, 1),
            "groups": 2,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
]
CONV_6D_IDS = [make_id(case) for case in CONV_6D_CASES]


def conv_module_from_case(
    N: int, case: Dict, device: torch.device, dtype: Union[torch.dtype, None] = None
) -> Union[Conv1d, Conv2d, Conv3d]:
    """Create PyTorch convolution module from a case.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the module.
        device: Device to load the module to.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).

    Returns:
        PyTorch convolution module.
    """
    conv_cls = {1: Conv1d, 2: Conv2d, 3: Conv3d}[N]
    return conv_cls(
        case["in_channels"],
        case["out_channels"],
        case["kernel_size"],
        **case["kwargs"],
        device=device,
        dtype=dtype,
    )


def convNd_module_from_case(
    N: int, case: Dict, device: torch.device, dtype: Union[torch.dtype, None] = None
) -> ConvNd:
    """Create einconv convolution module from a case.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the module.
        device: Device to load the module to.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).

    Returns:
        einconv convolution module.
    """
    return ConvNd(
        N,
        case["in_channels"],
        case["out_channels"],
        case["kernel_size"],
        **case["kwargs"],
        device=device,
        dtype=dtype,
    )
