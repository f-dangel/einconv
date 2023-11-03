"""Test symbolic representation and manipulation of tensors."""

from test.utils import DEVICE_IDS, DEVICES, report_nonclose

import torch
from pytest import mark, raises
from torch import bfloat16, float16, float32, manual_seed, rand

from einconv.simplifications.symbolic import SymbolicTensor

DTYPES = [float32, float16, bfloat16]
DTYPE_IDS = [f"dtype={dtype}" for dtype in DTYPES]


@mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_instantiate_no_history(device: torch.device, dtype: torch.dtype):
    """Test instantiation of a symbolic tensor without history.

    Args:
        device: Device to instantiate on.
        dtype: Data type to instantiate with.
    """
    manual_seed(0)

    name = "weight"
    shape = (4, 3, 5, 2)
    indices = ("c_out", "c_in", "k1", "k2")
    weight_symbolic = SymbolicTensor(name, shape, indices)

    # check that construction fails if shape and indices have different lengths
    with raises(ValueError):
        other_indices = ("c_out", "c_in", "k1")
        SymbolicTensor(name, shape, other_indices)

    # check that instantiation fails if tensor is unspecified
    with raises(ValueError):
        weight_symbolic.instantiate(device=device, dtype=dtype)

    # check that instantiation fails if shape is incorrect
    with raises(ValueError):
        other_shape = (4, 3, 5, 1)
        weight_symbolic.instantiate(rand(*other_shape, device=device, dtype=dtype))

    # check if properly converted to dtype and device
    tensor = rand(*shape)
    weight_tensor = weight_symbolic.instantiate(tensor, device=device, dtype=dtype)
    assert weight_tensor.dtype == dtype
    assert weight_tensor.device == device
    report_nonclose(tensor.to(device, dtype), weight_tensor)
