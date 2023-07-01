"""Contains test cases for the unfold functional.

The test cases were modified from the ``unfoldNd`` package's tests at
https://github.com/f-dangel/unfoldNd/blob/fbee3a4b1d51e4c7c9969745f8bd08ffdb9300f1/test/unfold_settings.py.
"""

from test.utils import make_id

from torch import float64, rand

UNFOLD_1D_CASES = [
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 1,
        "unfold_kwargs": {},
    },
    {
        "seed": 1,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 2,
        "unfold_kwargs": {},
    },
    {
        "seed": 2,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 3,
        "unfold_kwargs": {},
    },
    {
        "seed": 3,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 3,
        "unfold_kwargs": {"dilation": 2},
    },
    {
        "seed": 4,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 3,
        "unfold_kwargs": {"dilation": 2, "padding": 1},
    },
    {
        "seed": 5,
        "input_fn": lambda: rand(2, 3, 50),
        "kernel_size": 3,
        "unfold_kwargs": {"dilation": 2, "padding": 1, "stride": 2},
    },
]
UNFOLD_1D_IDS = [make_id(problem) for problem in UNFOLD_1D_CASES]


UNFOLD_2D_CASES = [
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": 1,
        "unfold_kwargs": {},
    },
    {
        "seed": 1,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": 2,
        "unfold_kwargs": {},
    },
    {
        "seed": 2,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (3, 2),
        "unfold_kwargs": {},
    },
    {
        "seed": 3,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (3, 2),
        "unfold_kwargs": {"dilation": 2},
    },
    {
        "seed": 4,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (3, 2),
        "unfold_kwargs": {"dilation": 2, "padding": 1},
    },
    {
        "seed": 5,
        "input_fn": lambda: rand(2, 3, 50, 40),
        "kernel_size": (3, 2),
        "unfold_kwargs": {"dilation": 2, "padding": 1, "stride": 2},
    },
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, dtype=float64),
        "kernel_size": 1,
        "unfold_kwargs": {},
        "id": "bug-float-64-input",
    },
]
UNFOLD_2D_IDS = [make_id(problem) for problem in UNFOLD_2D_CASES]


UNFOLD_3D_CASES = [
    {
        "seed": 0,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": 1,
        "unfold_kwargs": {},
    },
    {
        "seed": 1,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": 2,
        "unfold_kwargs": {},
    },
    {
        "seed": 2,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (4, 3, 2),
        "unfold_kwargs": {},
    },
    {
        "seed": 3,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (4, 3, 2),
        "unfold_kwargs": {"dilation": 2},
    },
    {
        "seed": 4,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (4, 3, 2),
        "unfold_kwargs": {"dilation": 2, "padding": 1},
    },
    {
        "seed": 5,
        "input_fn": lambda: rand(2, 3, 50, 40, 30),
        "kernel_size": (4, 3, 2),
        "unfold_kwargs": {"dilation": 2, "padding": 1, "stride": 2},
    },
]
UNFOLD_3D_IDS = [make_id(problem) for problem in UNFOLD_3D_CASES]
