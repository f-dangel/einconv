"""Contains test cases for ``einconv``'s module implementation of convolution."""

from test.utils import make_id
from typing import Dict, Union

import torch
from torch import rand
from torch.nn import Conv1d, Conv2d, Conv3d

from einconv.einconvnd import EinconvNd

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
        "input_fn": lambda: rand(2, 3, 50),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
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
        "input_fn": lambda: rand(2, 6, 99),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5,),
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


CONV_2D_MODULE_CASES = [
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 50, 40),
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
        "input_fn": lambda: rand(2, 3, 50, 45),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
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
        "input_fn": lambda: rand(2, 6, 99, 88),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5, 4),
        "conv_kwargs": {
            "stride": (2, 3),
            "padding": (1, 0),
            "dilation": (2, 1),
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
]
CONV_2D_MODULE_IDS = [make_id(case) for case in CONV_2D_MODULE_CASES]

CONV_3D_MODULE_CASES = [
    # no kwargs except for bias disabled
    {
        "seed": 0,
        # (batch_size, in_channels, num_pixels_d, num_pixels_h, num_pixels_w)
        "input_fn": lambda: rand(2, 3, 20, 15, 10),
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
        "input_fn": lambda: rand(2, 3, 50, 45, 40),
        "in_channels": 3,
        "out_channels": 9,
        "kernel_size": 5,
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
        "input_fn": lambda: rand(2, 6, 99, 88, 77),
        "in_channels": 6,
        "out_channels": 9,
        "kernel_size": (5, 4, 3),
        "conv_kwargs": {
            "stride": (2, 3, 2),
            "padding": (1, 0, 2),
            "dilation": (2, 1, 3),
            "groups": 3,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
]
CONV_3D_MODULE_IDS = [make_id(case) for case in CONV_3D_MODULE_CASES]

CONV_4D_MODULE_CASES = [
    # no kwargs (bias enabled)
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 25, 20, 15, 10),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 5,
        # stride, padding, dilation, groups, padding_mode, bias
        "conv_kwargs": {},
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
        "conv_kwargs": {
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
        "conv_kwargs": {
            "stride": (2, 3, 2, 1),
            "padding": (2, 0, 2, 1),
            "dilation": (2, 1, 3, 2),
            "groups": 2,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
]
CONV_4D_MODULE_IDS = [make_id(case) for case in CONV_4D_MODULE_CASES]

CONV_5D_MODULE_CASES = [
    # no kwargs (bias enabled)
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 15, 10, 8, 6, 5),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 3,
        # stride, padding, dilation, groups, padding_mode, bias
        "conv_kwargs": {
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
        "conv_kwargs": {
            "stride": (4, 3, 1, 2, 4),
            # TODO the following does not work:
            # (1, 0, 0, 0, 0)
            # But the following works:
            # (3, 0, 0, 0, 0)
            # (2, 0, 1, 1, 3)
            # (0, 1, 0, 0, 0)
            # (0, 0, 1, 0, 0)
            # (0, 0, 0, 1, 0)
            # (0, 0, 0, 0, 1)
            # Something seems to be wrong with the first padding axes, but
            # I don't know why. Could the bug be in the 3rd-party implementation?
            "padding": (2, 0, 1, 1, 3),
            # "dilation": (2, 1, 3, 2, 1), # not supported by 3rd party implementation
            "groups": 2,
            "padding_mode": "zeros",
            "bias": False,
        },
    },
]
CONV_5D_MODULE_IDS = [make_id(case) for case in CONV_5D_MODULE_CASES]

CONV_6D_MODULE_CASES = [
    # no kwargs (bias enabled)
    {
        "seed": 0,
        # (batch_size, in_channels, *num_pixels)
        "input_fn": lambda: rand(2, 3, 15, 10, 8, 6, 5, 5),
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 3,
        # stride, padding, dilation, groups, padding_mode, bias
        "conv_kwargs": {
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
        "conv_kwargs": {
            "stride": (3, 2, 2, 3, 3, 2),
            # TODO the following does not work:
            # (0, 1, 0, 0, 0, 0),
            # (0, 0, 1, 0, 0, 0),
            # 1,
            # But the following works:
            # (1, 0, 0, 0, 0, 0),
            # (0, 0, 0, 1, 0, 0),
            # (0, 0, 0, 0, 1, 0),
            # (0, 0, 0, 0, 0, 1),
            # (2, 0, 0, 2, 1, 1),
            # Something seems to be wrong with the second and third padding axes, but
            # I don't know why. Could the bug be in the 3rd-party implementation?
            "padding": (2, 0, 0, 2, 1, 1),
            # "dilation": (2, 1, 3, 2, 1, 1), # not supported by 3rd party implem.
            "groups": 2,
            "padding_mode": "zeros",
            "bias": True,
        },
    },
]
CONV_6D_MODULE_IDS = [make_id(case) for case in CONV_6D_MODULE_CASES]


def conv_module_from_case(
    N: int, case: Dict, device: torch.device, dtype: Union[torch.dtype, None] = None
) -> Union[Conv1d, Conv2d, Conv3d]:
    """Create PyTorch convolution module from a case.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the module.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).

    Returns:
        PyTorch convolution module.
    """
    conv_cls = {1: Conv1d, 2: Conv2d, 3: Conv3d}[N]
    return conv_cls(
        case["in_channels"],
        case["out_channels"],
        case["kernel_size"],
        **case["conv_kwargs"],
        device=device,
        dtype=dtype,
    )


def einconv_module_from_case(
    N: int, case: Dict, device: torch.device, dtype: Union[torch.dtype, None] = None
) -> EinconvNd:
    """Create einconv convolution module from a case.

    Args:
        N: Convolution dimension.
        case: Dictionary describing the module.
        dtype: Data type assumed by the layer. Default: ``None`` (``torch.float32``).

    Returns:
        einconv convolution module.
    """
    return EinconvNd(
        N,
        case["in_channels"],
        case["out_channels"],
        case["kernel_size"],
        **case["conv_kwargs"],
        device=device,
        dtype=dtype,
    )
