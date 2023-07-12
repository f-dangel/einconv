"""Utility functions for ``einconv``."""

from math import floor
from typing import Any, List, Tuple, Union

import torch
from torch.nn import Module

cpu = torch.device("cpu")


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
            raise ValueError(f"{attr!r} attribute does not match: {attr1} ≠ {attr2}")


def get_conv_paddings(
    kernel_size: int, stride: int, padding: Union[int, str], dilation: int
) -> Tuple[int, int]:
    """Get left and right padding as of a convolution as integers.

    Args:
        kernel_size: Kernel size along dimension.
        stride: Stride along dimension.
        padding: Padding along dimension. Can be an integer or a string. Allowed
            strings are ``'same'`` and ``'valid'``.
        dilation: Dilation along dimension.

    Returns:
        Left and right padding.

    Raises:
        ValueError: If ``padding='same'`` and the convolution is strided.
        ValueError: For unknown convolution strings.
    """
    if isinstance(padding, str):
        if padding == "valid":
            padding_left, padding_right = 0, 0
        elif padding == "same":
            if stride != 1:
                raise ValueError(
                    "padding='same' is not supported for strided convolutions."
                )
            total_padding = dilation * (kernel_size - 1)
            padding_left = total_padding // 2
            padding_right = total_padding - padding_left
        else:
            raise ValueError(f"Unknown string-value for padding: {padding!r}.")
    else:
        padding_left, padding_right = padding, padding

    return padding_left, padding_right


def get_conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: Union[int, str],
    dilation: int,
) -> int:
    """Compute the output dimension of a convolution.

    Args:
        input_size: Number of pixels along dimension.
        kernel_size: Kernel size along dimension.
        stride: Stride along dimension.
        padding: Padding along dimension. Can be an integer or a string. Allowed
            strings are ``'same'`` and ``'valid'``.
        dilation: Dilation along dimension.

    Returns:
        Convolution output dimension.
    """
    padding_left, padding_right = get_conv_paddings(
        kernel_size, stride, padding, dilation
    )
    kernel_span = kernel_size + (kernel_size - 1) * (dilation - 1)

    return 1 + floor((input_size + padding_left + padding_right - kernel_span) / stride)


def get_conv_input_size(
    output_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """Compute the input dimension of a convolution (output of a transpose convolution).

    Args:
        output_size: Spatial output dimensions of a convolution (spatial input
            dimensions of a transpose convolution).
        kernel_size: Kernel size of the convolution.
        stride: Stride of the convolution.
        padding: Padding of the convolution.
        output_padding: Number of pixels at the right edge of the convolution's input
            that do not overlap with the kernel and hence must be added as padding when
            considering a transpose convolution.
        dilation: Dilation of the convolution.

    Returns:
        Input dimension of the convolution (output dimension of transpose convolution.)
    """
    kernel_span = kernel_size + (kernel_size - 1) * (dilation - 1)
    input_size_padded = (output_size - 1) * stride + kernel_span + output_padding

    return input_size_padded - 2 * padding
