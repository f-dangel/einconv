"""Utility functions for ``einconv``."""

from typing import Tuple, Union


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
