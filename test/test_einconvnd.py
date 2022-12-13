"""Tests for ``einconv/einconvnd``."""

from test.utils import report_nonclose

from pytest import mark
from torch import manual_seed, rand
from torch.nn.functional import conv1d

from einconv.einconvnd import einconv1d

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
    # non-default bias, padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "bias_fn": lambda: rand(4),
        "conv_kwargs": {"padding": (2,), "stride": (3,), "dilation": (1,)},
    },
]


@mark.parametrize("case", CONV_1D_CASES)
def test_einconv1d(case):
    """Compare PyTorch's conv1d with einconv's einconv1d."""
    manual_seed(case["seed"])
    x = case["input_fn"]()
    weight = case["weight_fn"]()
    bias = case["bias_fn"]()

    conv1d_output = conv1d(x, weight, bias=bias, **case["conv_kwargs"])
    einconv1d_output = einconv1d(x, weight, bias=bias, **case["conv_kwargs"])

    report_nonclose(conv1d_output, einconv1d_output)
