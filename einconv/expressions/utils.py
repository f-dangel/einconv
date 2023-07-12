"""Utility functions for creating einsum expressions."""

from typing import List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from einconv import index_pattern
from einconv.utils import _tuple, cpu, get_conv_input_size


def create_conv_transpose_index_patterns(
    N: int,
    output_size: Union[int, Tuple[int, ...]],
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    output_padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = cpu,
    dtype: torch.dtype = torch.bool,
) -> List[Tensor]:
    """Create the index pattern tensors for all dimensions of a transpose convolution.

    Args:
        N: Transpose convolution dimension.
        output_size: Spatial input dimensions of the transpose convolution (output
            dimensions of the associated convolution). Can be a single integer
            (shared along all spatial dimensions), or an ``N``-tuple of integers.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        output_padding: Number of pixels at the right edge of the convolution's input
            that do not overlap with the kernel and hence must be added as padding when
            considering a transpose convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``0``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        device: Device to create the tensors on. Default: ``'cpu'``.
        dtype: Data type of the pattern tensor. Default: ``torch.bool``.

    Returns:
        List of index pattern tensors for dimensions ``1, ..., N``.
    """
    # determine associated convolution's input size
    t_output_size = _tuple(output_size, N)
    t_kernel_size = _tuple(kernel_size, N)
    t_stride = _tuple(stride, N)
    t_padding = _tuple(padding, N)
    t_output_padding = _tuple(output_padding, N)
    t_dilation = _tuple(dilation, N)
    t_input_size = tuple(
        get_conv_input_size(
            t_output_size[n],
            t_kernel_size[n],
            t_stride[n],
            t_padding[n],
            t_output_padding[n],
            t_dilation[n],
        )
        for n in range(N)
    )

    return create_conv_index_patterns(
        N,
        t_input_size,
        t_kernel_size,
        stride=t_stride,
        padding=t_padding,
        dilation=t_dilation,
        device=device,
        dtype=dtype,
    )


def create_conv_index_patterns(
    N: int,
    input_size: Union[int, Tuple[int, ...]],
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = cpu,
    dtype: torch.dtype = torch.bool,
) -> List[Tensor]:
    """Create the index pattern tensors for all dimensions of a convolution.

    Args:
        N: Convolution dimension.
        input_size: Spatial dimensions of the convolution. Can be a single integer
            (shared along all spatial dimensions), or an ``N``-tuple of integers.
        kernel_size: Kernel dimensions. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along
            all spatial dimensions), an ``N``-tuple of integers, or a string.
            Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
        dilation: Dilation of the convolution. Can be a single integer (shared along
            all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        device: Device to create the tensors on. Default: ``'cpu'``.
        dtype: Data type of the pattern tensor. Default: ``torch.bool``.

    Returns:
        List of index pattern tensors for dimensions ``1, ..., N``.
    """
    # convert into tuple format
    t_input_size = _tuple(input_size, N)
    t_kernel_size = _tuple(kernel_size, N)
    t_stride = _tuple(stride, N)
    t_padding = padding if isinstance(padding, str) else _tuple(padding, N)
    t_dilation = _tuple(dilation, N)

    return [
        index_pattern(
            t_input_size[n],
            t_kernel_size[n],
            stride=t_stride[n],
            padding=t_padding if isinstance(t_padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=device,
            dtype=dtype,
        )
        for n in range(N)
    ]


def translate_to_torch(einops_equation: str) -> str:
    """Translate an einsum equation in einops convention to PyTorch convention.

    An einops equation can use multi-letter indices, and separates indices with a
    white space. The arrow can be separated by spaces too. A PyTorch equation uses
    single-letter indices and does not separate them with a white space.

    For instance, a valid einops equation might be ``'row j, j col -> row col'``.
    A valid PyTorch version would be ``'ij,jk->ik'``.

    Args:
        einops_equation: Einsum equation in einops syntax.

    Returns:
        Einsum equation in PyTorch syntax.

    Raises:
        ValueError: If parsing the indices failed.
    """
    # get all indices
    indices = []
    for operand_indices in einops_equation.split("->")[0].strip().split(","):
        for operand_idx in operand_indices.strip().split(" "):
            if operand_idx not in indices:
                indices.append(operand_idx)

    if "" in indices:
        raise ValueError(f"Index parsing failed. Equation {einops_equation}: {indices}")

    # figure out which ones need to be renamed
    rename = [idx for idx in indices if len(idx) > 1 or not idx.islower()]
    keep = {idx for idx in indices if idx not in rename}

    # rename them (make sure the index to be renamed is not contained in others)
    old_to_new = dict(zip(rename, get_letters(len(rename), blocked=keep)))

    torch_equation = einops_equation
    while rename:
        idx = rename.pop(0)
        # cannot rename if idx is a sub-string of other indices, e.g. idx has the value
        # `'row'` but there is another index named `'row2'`.
        if any(idx in other_idx for other_idx in rename):
            rename.append(idx)
        else:
            torch_equation = torch_equation.replace(idx, old_to_new[idx])

    # clean white spaces
    return torch_equation.replace(" ", "")


def get_letters(num_letters: int, blocked: Optional[Set] = None) -> List[str]:
    """Return a list of ``num_letters`` unique letters for an einsum equation.

    Args:
        num_letters: Number of letters to return.
        blocked: Set of letters that should not be used.

    Returns:
        List of ``num_letters`` unique letters.

    Raises:
        ValueError: If ``num_letters`` cannot be satisfies with einsum-supported
            letters.
    """
    if num_letters == 0:
        return []

    max_letters = 26
    blocked = set() if blocked is None else blocked
    letters = []

    for i in range(max_letters):
        letter = chr(ord("a") + i)
        if letter not in blocked:
            letters.append(letter)
            if len(letters) == num_letters:
                return letters

    raise ValueError(
        f"Ran out of letters. PyTorch's einsum supports {max_letters} letters."
        + f" Requested {num_letters}, blocked: {len(blocked)}.)"
    )
