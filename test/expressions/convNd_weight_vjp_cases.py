"""Test cases for ``einconv.expressions.convNd_weight_vjp``"""

from test.utils import make_id

from torch import rand

WEIGHT_VJP_1D_CASES = [
    # no kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50),
        "weight_fn": lambda: rand(6, 2, 5),
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "kwargs": {"padding": (2,), "stride": (3,), "dilation": (1,)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "weight_fn": lambda: rand(4, 3, 5),
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 20),
        "weight_fn": lambda: rand(4, 3, 5),
        "kwargs": {"padding": "same"},
    },
]
WEIGHT_VJP_1D_IDS = [make_id(case) for case in WEIGHT_VJP_1D_CASES]

WEIGHT_VJP_2D_CASES = [
    # no kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 5),
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 5),
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50, 40),
        "weight_fn": lambda: rand(6, 2, 5, 5),
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 6),
        "kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 3),
        "kwargs": {"padding": (2, 1), "stride": (3, 2), "dilation": (1, 2)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 6),
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "weight_fn": lambda: rand(4, 3, 5, 3),
        "kwargs": {"padding": "same", "dilation": (1, 2)},
    },
]
WEIGHT_VJP_2D_IDS = [make_id(case) for case in WEIGHT_VJP_2D_CASES]

WEIGHT_VJP_3D_CASES = [
    # no kwargs
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 5, 5),
        "kwargs": {},
    },
    # non-default stride
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 5, 5),
        "kwargs": {"stride": 2},
    },
    # non-default stride, groups
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 4, 50, 40, 30),
        "weight_fn": lambda: rand(6, 2, 5, 5, 5),
        "kwargs": {"stride": 3, "groups": 2},
    },
    # non-default padding
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 6, 7),
        "kwargs": {"padding": 2},
    },
    # padding, stride, dilation specified as tuple
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 3, 2),
        "kwargs": {"padding": (2, 1, 0), "stride": (3, 2, 1), "dilation": (1, 2, 3)},
    },
    # non-default dilation
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 40),
        "weight_fn": lambda: rand(4, 3, 5, 6, 7),
        "kwargs": {"dilation": 2},
    },
    # padding supplied as string
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "weight_fn": lambda: rand(4, 3, 5, 3, 2),
        "kwargs": {"padding": "same", "dilation": (1, 2, 2)},
    },
]
WEIGHT_VJP_3D_IDS = [make_id(case) for case in WEIGHT_VJP_3D_CASES]
