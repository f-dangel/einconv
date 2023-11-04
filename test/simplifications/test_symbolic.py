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

    # instantiation fails if indices are not unique
    with raises(ValueError):
        SymbolicTensor(name, shape, ("c_out", "c_in", "c_out", "k2"))

    # check that construction fails if shape and indices have different lengths
    other_indices = ("c_out", "c_in", "k1")
    with raises(ValueError):
        SymbolicTensor(name, shape, other_indices)

    # check that instantiation fails if tensor is unspecified
    with raises(ValueError):
        weight_symbolic.instantiate(device=device, dtype=dtype)

    # check that instantiation fails if shape is incorrect
    other_shape = (4, 3, 5, 1)
    with raises(ValueError):
        weight_symbolic.instantiate(rand(*other_shape, device=device, dtype=dtype))

    # check if properly converted to dtype and device
    tensor = rand(*shape)
    weight_tensor = weight_symbolic.instantiate(tensor, device=device, dtype=dtype)
    assert weight_tensor.dtype == dtype
    assert weight_tensor.device == device
    report_nonclose(tensor.to(device, dtype), weight_tensor)


@mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_group(device: torch.device, dtype: torch.dtype):
    """Test grouping multiple indices together.

    Args:
        device: Device to instantiate on after grouping.
        dtype: Data type to instantiate with after grouping.
    """
    manual_seed(0)

    name = "weight"
    shape = (4, 3, 5, 2)
    indices = ("c_out", "c_in", "k1", "k2")
    weight_symbolic = SymbolicTensor(name, shape, indices)
    weight_tensor = rand(*shape, device=device, dtype=dtype)

    # grouping non-consecutive indices is not supported
    with raises(NotImplementedError):
        weight_symbolic.group(("c_out", "k1"))

    # grouping fails if grouped axis name already exists
    poor_naming = ("(c_out c_in)", "c_out", "c_in", "a")
    with raises(ValueError):
        SymbolicTensor(name, shape, poor_naming).group(("c_out", "c_in"))

    # grouping correctly transforms a tensor when instantiating
    weight_symbolic.group(("c_out", "c_in"))
    grouped_indices = ("(c_out c_in)", "k1", "k2")
    grouped_shape = (12, 5, 2)
    assert weight_symbolic.indices == grouped_indices
    assert weight_symbolic.shape == grouped_shape
    grouped_weight_tensor = weight_symbolic.instantiate(weight_tensor)
    report_nonclose(weight_tensor.flatten(end_dim=1), grouped_weight_tensor)


@mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_narrow(device: torch.device, dtype: torch.dtype):
    """Test narrowing along an index.

    Args:
        device: Device to instantiate on after narrowing.
        dtype: Data type to instantiate with after narrowing.
    """
    manual_seed(0)

    name = "weight"
    shape = (4, 3, 5, 2)
    indices = ("c_out", "c_in", "k1", "k2")
    weight_symbolic = SymbolicTensor(name, shape, indices)
    weight_tensor = rand(*shape, device=device, dtype=dtype)

    # start must be non-negative
    with raises(ValueError):
        weight_symbolic.narrow("c_out", -1, 1)

    # length must be positive
    with raises(ValueError):
        weight_symbolic.narrow("c_out", 1, 0)

    # range must not exceed dimension
    with raises(ValueError):
        weight_symbolic.narrow("c_out", 0, 5)

    # range must not exceed dimension
    with raises(ValueError):
        weight_symbolic.narrow("c_out", 2, 3)

    # narrowing correctly transforms a tensor when instantiating
    weight_symbolic.narrow("k1", 1, 2)
    narrowed_indices = indices
    narrowed_shape = (4, 3, 2, 2)
    assert weight_symbolic.indices == narrowed_indices
    assert weight_symbolic.shape == narrowed_shape
    narrowed_weight_tensor = weight_symbolic.instantiate(weight_tensor)
    report_nonclose(weight_tensor.narrow(2, 1, 2), narrowed_weight_tensor)


@mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_rename(device: torch.device, dtype: torch.dtype):
    """Test renaming an index.

    Args:
        device: Device to instantiate on after renaming.
        dtype: Data type to instantiate with after renaming.
    """
    manual_seed(0)

    name = "weight"
    shape = (4, 3, 5, 2)
    indices = ("c_out", "c_in", "k1", "k2")
    weight_symbolic = SymbolicTensor(name, shape, indices)
    weight_tensor = rand(*shape, device=device, dtype=dtype)

    # new name cannot be in use
    with raises(ValueError):
        weight_symbolic.rename("c_out", "k1")

    # renaming does nothing to a tensor when instantiating
    weight_symbolic.rename("k1", "k1_new")
    renamed_indices = ("c_out", "c_in", "k1_new", "k2")
    assert weight_symbolic.indices == renamed_indices
    assert weight_symbolic.shape == shape
    renamed_weight_tensor = weight_symbolic.instantiate(weight_tensor)
    report_nonclose(weight_tensor, renamed_weight_tensor)


@mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_ungroup(device: torch.device, dtype: torch.dtype):
    """Test un-grouping an index.

    Args:
        device: Device to instantiate on after un-grouping.
        dtype: Data type to instantiate with after un-grouping.
    """
    manual_seed(0)

    name = "weight"
    shape = (4, 3, 20, 2)
    indices = ("c_out", "c_in", "k1", "k2")
    weight_symbolic = SymbolicTensor(name, shape, indices)
    weight_tensor = rand(*shape, device=device, dtype=dtype)

    # sizes must preserve the ungrouped dimension
    with raises(ValueError):
        weight_symbolic.ungroup("k1", (9, 2))

    # can't rename to an existing name
    with raises(ValueError):
        weight_symbolic.ungroup("k1", (10, 2), new_names=("k2", "d"))

    # un-grouping applies the correct transformation
    weight_symbolic.ungroup("k1", (5, 2, 2))
    ungrouped_shape = (4, 3, 5, 2, 2, 2)
    assert weight_symbolic.shape == ungrouped_shape
    ungrouped_weight_tensor = weight_symbolic.instantiate(weight_tensor)
    report_nonclose(weight_tensor.reshape(*ungrouped_shape), ungrouped_weight_tensor)
