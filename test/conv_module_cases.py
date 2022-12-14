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
        dtype=dtype
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
        dtype=dtype
    )
