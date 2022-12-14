"""Utility functions for testing."""

from types import LambdaType
from typing import Any, Dict, List

from torch import Tensor, allclose, cuda, device, isclose
from torch.nn import Module


def report_nonclose(
    tensor1: Tensor,
    tensor2: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
):
    """Compare two tensors, raise exception if nonclose values and print them.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        rtol: Relative tolerance (see ``torch.allclose``). Default: ``1e-5``.
        atol: Absolute tolerance (see ``torch.allclose``). Default: ``1e-8``.
        equal_nan: Whether comparing two NaNs should be considered as ``True``
            (see ``torch.allclose``). Default: ``False``.

    Raises:
        ValueError: If the two tensors don't match in shape or have nonclose values.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Arrays shapes don't match.")

    if allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan):
        print("Compared tensors match.")
    else:
        for a1, a2 in zip(tensor1.flatten(), tensor2.flatten()):
            if not isclose(a1, a2, atol=atol, rtol=rtol, equal_nan=equal_nan):
                print(f"{a1} ≠ {a2}")
        raise ValueError("Compared tensors don't match.")


def get_available_devices() -> List[device]:
    """Return CPU and, if present, GPU device.

    Returns:
        List of available PyTorch devices.
    """
    devices = [device("cpu")]

    if cuda.is_available():
        devices.append(device("cuda"))

    return devices


DEVICES = get_available_devices()
DEVICE_IDS = [f"device_{dev}" for dev in DEVICES]


def make_id(case: Dict) -> str:
    """Create human-readable ID for a test case.

    Args:
        case: A dictionary.

    Returns:
        Human-readable string describing the dictionary's items.
    """
    parts = []

    for key, value in case.items():
        key_str = str(key)

        if isinstance(value, Dict):
            parts.append("_".join([key_str, make_id(value)]))
        else:
            if isinstance(value, LambdaType):
                output = value()
                if isinstance(output, Tensor):
                    value_str = "x".join(str(s) for s in value().shape)
                else:
                    value_str = str(output)
            else:
                value_str = str(value)

            parts.append("_".join([key_str, value_str]))

    return "_".join(parts)


def compare_attributes(obj1: Any, obj2: Any, attributes: List[str]):
    """Compare multiple attributes of two objects.

    Args:
        obj1: First object.
        obj2: Second object.
        attributes: Attribute names to compare.

    Raises:
        ValueError: If two attributes don't match.
    """
    for attr in attributes:
        attr1 = getattr(obj1, attr)
        attr2 = getattr(obj2, attr)
        if attr1 != attr2:
            raise ValueError(f"'{attr}' attribute does not match: {attr1} ≠ {attr2}")


def sync_parameters(module1: Module, module2: Module) -> None:
    """Copy the values of ``module1``'s parameters to ``module2``'s parameters

    Args:
        module1: Module whose parameters will be used for syncing.
        module2: Module whose parameters will be synced to those of ``module1``.

    Raises:
        ValueError: If module parameter names don't match.
    """
    param_names1 = {n for (n, _) in module1.named_parameters()}
    param_names2 = {n for (n, _) in module2.named_parameters()}

    if param_names1 != param_names2:
        raise ValueError(
            f"Module parameters have different names: {param_names1} ≠ {param_names2}"
        )

    for param_name in param_names1:
        param1 = getattr(module1, param_name)
        param2 = getattr(module2, param_name)

        if param1 is not None or param2 is not None:
            attributes = ["shape", "dtype", "device"]
            compare_attributes(param1, param2, attributes)
            param2.data = param1.data
