"""Utility functions for ``einconv``."""

from typing import Any, List, Tuple, Union

from torch.nn import Module


def _tuple(conv_hyperparameter: Union[int, Tuple[int, ...]], N: int) -> Tuple[int, ...]:
    """Convert a convolution hyperparameter specified as int or tuple to tuple format.

    This function is a generalization of
    ``torch.nn.modules.utils.{_single, _pair, _triple}``.

    Args:
        conv_hyperparameter: kernel size, padding, dilation, or stride, accepted as
            input by PyTorch's convolution functionals or modules.
        N: Convolution dimension.

    Returns:
        A tuple with ``N`` entries containing the hyperparameter to be used for each
        dimension.

    Raises:
        ValueError: If the hyperparameter is not specified correctly.
    """
    if isinstance(conv_hyperparameter, int):
        return N * (conv_hyperparameter,)

    if not isinstance(conv_hyperparameter, tuple):
        raise ValueError(
            "Convolution hyperparameters must be specified as int or tuple of int."
            + f" Got {conv_hyperparameter}."
        )
    elif len(conv_hyperparameter) != N:
        raise ValueError(
            f"Convolution hyperparameters must be specified as tuple of length {N}."
            + f" Got tuple of length {len(conv_hyperparameter)}."
        )
    elif any(not isinstance(h, int) for h in conv_hyperparameter):
        raise ValueError(
            f"All hyperparameters must be integers. Got {conv_hyperparameter}."
        )
    else:
        return conv_hyperparameter


def sync_parameters(module1: Module, module2: Module) -> None:
    """Copy the values of ``module1``'s parameters to ``module2``'s parameters.

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
            param2.data = param1.data.clone()


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
