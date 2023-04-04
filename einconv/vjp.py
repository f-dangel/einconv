"""Contains functionality for tensor network VJPs of convolutions."""

from typing import List, Tuple, Union

from torch import Tensor

from einconv.einconvnd import _conv_einsum_equation, _conv_einsum_operands


def _conv_weight_vjp_einsum_equation(N: int) -> str:
    """Return the einsum equation for a weight VJP.

    Args:
        N: Convolution dimensionality

    Returns:
        Einsum equation for the weight VJP. Argument order is assumed to
        be ``input, *index_patterns, grad_output``.
    """
    forward_equation = _conv_einsum_equation(N)

    # swap the kernel and the output indices
    operands_idxs, out_idxs = forward_equation.split("->")
    operands_idxs = operands_idxs.split(",")
    # kernel position is last
    kernel_idxs = operands_idxs.pop()
    operands_idxs.append(out_idxs)

    return "->".join([",".join(operands_idxs), kernel_idxs])


def _conv_weight_vjp_einsum_operands(
    input: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> List[Tensor]:
    """Prepare the tensor contraction operands for the VJP.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Returns:
        Tensor list containing the operands. Convention: reshaped input, followed by
        index pattern tensors, followed by reshaped grad_output.
    """
    operands = _conv_einsum_operands(input, weight, stride, padding, dilation, groups)
    # drop kernel (last)
    operands.pop()

    # separate groups
    batch_size = grad_output.shape[0]
    out_channels = grad_output.shape[1]
    output_spatial_dims = grad_output.shape[2:]
    operands.append(
        grad_output.reshape(
            batch_size, groups, out_channels // groups, *output_spatial_dims
        )
    )

    return operands


def _conv_input_vjp_einsum_equation(N: int) -> str:
    """Return the einsum equation for an input VJP.

    Args:
        N: Convolution dimensionality

    Returns:
        Einsum equation for the input VJP. Argument order is assumed to
        be ``grad_output, *index_patterns, kernel``.
    """
    forward_equation = _conv_einsum_equation(N)

    # swap the input and the output indices
    operands_idxs, out_idxs = forward_equation.split("->")
    operands_idxs = operands_idxs.split(",")
    # input position is first
    input_idxs = operands_idxs.pop(0)
    operands_idxs = [out_idxs] + operands_idxs

    return "->".join([",".join(operands_idxs), input_idxs])


def _conv_input_vjp_einsum_operands(
    input: Tensor,
    weight: Tensor,
    grad_output: Tensor,
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> List[Tensor]:
    """Prepare the tensor contraction operands for the VJP.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Returns:
        Tensor list containing the operands. Convention: reshaped grad_output, followed
        by index pattern tensors, followed by reshaped kernel.
    """
    operands = _conv_einsum_operands(input, weight, stride, padding, dilation, groups)
    # drop input (first)
    operands.pop(0)

    # separate groups
    batch_size = grad_output.shape[0]
    out_channels = grad_output.shape[1]
    output_spatial_dims = grad_output.shape[2:]
    operands = [
        grad_output.reshape(
            batch_size, groups, out_channels // groups, *output_spatial_dims
        )
    ] + operands

    return operands
