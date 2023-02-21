"""PyTorch modules and functionals implementing N-dimensional convolution."""

from __future__ import annotations

from math import sqrt
from typing import List, Tuple, Union

import torch
from torch import Tensor, einsum, empty
from torch.nn import Conv1d, Conv2d, Conv3d, Module, Parameter, init

from einconv.index_pattern import conv_index_pattern
from einconv.utils import _tuple, sync_parameters


class EinconvNd(Module):
    """Module for N-dimensional convolution using tensor contractions."""

    def __init__(
        self,
        N: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], str] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Union[None, torch.device] = None,
        dtype: Union[None, torch.dtype] = None,
    ) -> None:
        """Initialize convolution layer.

        The weight has shape ``[out_channels, in_channels // groups, *kernel_size]``
        with ``len(kernel_size)==N``. The bias has shape ``[out_channels]``.

        Parameters are initialized using the same convention as PyTorch's convolutions.

        Args:
            N: Convolution dimension. For ``N=1,2,3`` the layer behaves like PyTorch's
                ``nn.Conv{N=1,2,3}d`` layers. However, this layer generalizes
                convolution and therefore also supports ``N>3``.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel dimensions. Can be a single integer (shared along all
                spatial dimensions), or an ``N``-tuple of integers.
            stride: Stride of the convolution. Can be a single integer (shared along all
                spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
            padding: Padding of the convolution. Can be a single integer (shared along
                all spatial dimensions), an ``N``-tuple of integers, or a string.
                Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
            dilation: Dilation of the convolution. Can be a single integer (shared along
                all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
            groups: How to split the input into groups. Default: ``1``.
            bias: Whether to use a non-zero bias vector. Default: ``True``.
            padding_mode: How to perform padding. Default: ``'zeros'``. No other modes
                are supported at the moment.
            device: Device on which the module is initialized.
            dtype: Data type assumed by the module.

        Raises:
            NotImplementedError: For unsupported padding modes.
            ValueError: For invalid combinations of ``in_channels``, ``out_channels``,
                and ``groups``.
        """
        super().__init__()

        if padding_mode != "zeros":
            raise NotImplementedError("Only padding_mode='zeros' supported.")

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError(
                f"groups ({groups}) must divide in_channels ({in_channels})."
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"groups ({groups}) must divide out_channels ({out_channels})."
            )

        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuple(kernel_size, N)
        self.stride = _tuple(stride, N)
        self.padding = padding if isinstance(padding, str) else _tuple(padding, N)
        self.padding_mode = padding_mode
        self.dilation = _tuple(dilation, N)
        self.groups = groups

        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = Parameter(empty(weight_shape, device=device, dtype=dtype))

        if bias:
            bias_shape = (out_channels,)
            self.bias = Parameter(empty(bias_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @classmethod
    def from_nn_Conv(cls, conv_module: Union[Conv1d, Conv2d, Conv3d]) -> EinconvNd:
        """Convert a ``torch.nn.Conv{1,2,3}d`` module to a ``EinconvNd`` layer.

        Args:
            conv_module: Convolution module.

        Returns:
            EinconvNd module.
        """
        N = {Conv1d: 1, Conv2d: 2, Conv3d: 3}[conv_module.__class__]
        einconv_module = cls(
            N,
            conv_module.in_channels,
            conv_module.out_channels,
            conv_module.kernel_size,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=conv_module.bias is not None,
            padding_mode=conv_module.padding_mode,
            device=conv_module.weight.device,
            dtype=conv_module.weight.dtype,
        )
        sync_parameters(conv_module, einconv_module)

        return einconv_module

    def reset_parameters(self):
        """Initialize the parameters.

        Follows the initialization scheme for convolutions in PyTorch, see
        https://github.com/pytorch/pytorch/blob/orig/release/1.13/torch/nn/modules/conv.py#L146-L155
        """
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Perform convolution on the input.

        Args:
            input: Convolution input. Has shape
                ``[batch_size, in_channels, *spatial_in_dims]`` where
                ``len(spatial_in_dims)==N``.

        Returns:
            Result of the convolution. Has shape
                ``[batch_size, out_channels, *spatial_out_dims]`` where
                ``len(spatial_out_dims)==N``.
        """
        return einconvNd(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        """Generate representation of extra arguments.

        Returns:
            String describing the module's arguments.
        """
        s = (
            "{N}, {in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if (self.padding != (0,) * len(self.padding)) or isinstance(self.padding, str):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


def einconvNd(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, str, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
):
    """Generalization of ``torch.nn.functional.conv{1,2,3}d`` to ``N``d.

    ``N`` is determined from the input tensor: It's first axis is the batch dimension,
    the second axis the channel dimension, and the remaining number of dimensions is
    interpreted as spatial dimension (with number of spatial dimensions ``N``)

    Args:
        input: Input of the convolution. Has shape ``[batch_size,
            in_channels, *]`` where ``*`` can be an arbitrary shape. The
            convolution dimension is ``len(*)``.
        weight: Kernel of the convolution. Has shape ``[out_channels,
            in_channels / groups, *]`` where ``*`` contains the kernel sizes and has
            length ``N``.
        bias: Optional bias vector of the convolution. Has shape ``[out_channels]``.
            Default: ``None``.
        stride: Stride of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        padding: Padding of the convolution. Can be a single integer (shared along all
            spatial dimensions), an ``N``-tuple of integers, or a string. Allowed
            strings are ``'same'`` and ``'valid'``. Default: ``0``.
        dilation: Dilation of the convolution. Can be a single integer (shared along all
            spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
        groups: How to split the input into groups. Default: ``1``.

    Returns:
        Result of the convolution. Has shape ``[batch_size, out_channels, *]`` where
        ``*`` is the the spatial output dimension shape.
    """
    _conv_check_args(input, weight, bias, groups)

    N = input.dim() - 2
    equation = _conv_einsum_equation(N)
    operands: List[Tensor] = _conv_einsum_operands(
        input, weight, stride, padding, dilation, groups
    )
    output = einsum(equation, *operands)
    output = output.flatten(start_dim=1, end_dim=2)

    if bias is not None:
        out_channels = weight.shape[0]
        shape_before_expand = (1, out_channels) + N * (1,)
        output += bias.reshape(*shape_before_expand).expand_as(output)

    return output


def _conv_check_args(
    input: Tensor, weight: Tensor, bias: Union[Tensor, None], groups: int
):
    """Check the input arguments to ``einconvNd``.

    Args:
        See ``einconvNd``.
        # noqa: DAR101

    Raises:
        ValueError: If bias has incorrect shape.
        ValueError: If weight dimension is incorrect.
        ValueError: If weight shape is invalid.
    """
    N = input.dim() - 2
    in_channels = input.shape[1]
    out_channels = weight.shape[0]

    if weight.dim() != N + 2:
        raise ValueError(
            f"For Conv(N={N})d, the kernel must be {N+2}d. Got {weight.dim()}d."
        )

    if weight.shape[0] % groups != 0:
        raise ValueError(
            f"Groups ({groups}) must divide out_channels ({weight.shape[0]})."
        )

    if weight.shape[1] * groups != in_channels:
        raise ValueError(
            f"Kernel dimension 1 ({weight.shape[1]}) multiplied by groups ({groups})"
            + f" must equal in_channels ({in_channels})."
        )

    if bias is not None and (bias.dim() != 1 or bias.numel() != out_channels):
        raise ValueError(f"Bias should have shape [{out_channels}]. Got {bias.shape}.")


def _conv_einsum_equation(N: int) -> str:
    """Generate einsum equation for convolution.

    The arguments are ``input, *index_patterns, weight -> output``.

    See https://arxiv.org/pdf/1908.04471.pdf, figure 2a for a visualization of the 3d
    case (neglecting the groups). The Nd case follows identically, and groups can be
    supported by a separate axis in the input, weight, and output.

    Args:
        N: Convolution dimension.
        # noqa: DAR101

    Raises:
        ValueError: If the equation cannot be realized without exceeding the alphabet.

    Returns:
        Einsum equation for N-dimensional convolution.
    """
    input_str = ""
    output_str = ""
    pattern_strs: List[str] = []
    kernel_str = ""

    # requires 4 + 3 * N letters
    # einsum can deal with the 26 lowercase letters of the alphabet
    max_letters, required_letters = 26, 4 + 3 * N
    if required_letters > max_letters:
        raise ValueError(
            f"Cannot form einsum equation. Need {required_letters} letters."
            + f" But einsum only supports {max_letters}."
        )
    letters = [chr(ord("a") + i) for i in range(required_letters)]

    # batch dimension
    batch_letter = letters.pop()
    input_str += batch_letter
    output_str += batch_letter

    # group dimension
    group_letter = letters.pop()
    input_str += group_letter
    output_str += group_letter
    kernel_str += group_letter

    # input and output channel dimensions
    in_channel_letter = letters.pop()
    out_channel_letter = letters.pop()
    input_str += in_channel_letter
    output_str += out_channel_letter
    kernel_str += out_channel_letter + in_channel_letter

    # coupling of input, output via kernel
    for _ in range(N):
        input_letter = letters.pop()
        kernel_letter = letters.pop()
        output_letter = letters.pop()

        input_str += input_letter
        output_str += output_letter
        kernel_str += kernel_letter
        pattern_strs.append(kernel_letter + output_letter + input_letter)

    input_equation = ",".join([input_str] + pattern_strs + [kernel_str])

    return "->".join([input_equation, output_str])


def _conv_einsum_operands(
    input: Tensor,
    weight: Tensor,
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, str, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
) -> List[Tensor]:
    """Prepare the tensor contraction operands.

    Args:
        See ``einconvNd``.

    Returns:
        Tensor list containing the operands. Convention: reshaped input, followed by
        index pattern tensors, followed by reshaped weight.
    """
    N = input.dim() - 2

    (batch_size, in_channels), input_sizes = input.shape[:2], input.shape[2:]
    (out_channels, _), kernel_sizes = weight.shape[:2], weight.shape[2:]

    operands = [
        input.reshape(batch_size, groups, in_channels // groups, *input_sizes),
    ]

    # convert into tuple format
    t_stride: Tuple[int, ...] = _tuple(stride, N)
    t_padding: Union[Tuple[int, ...], str] = (
        padding if isinstance(padding, str) else _tuple(padding, N)
    )
    t_dilation: Tuple[int, ...] = _tuple(dilation, N)

    operands.extend(
        conv_index_pattern(
            input_sizes[n],
            kernel_sizes[n],
            stride=t_stride[n],
            padding=padding if isinstance(padding, str) else t_padding[n],
            dilation=t_dilation[n],
            device=input.device,
        ).to(input.dtype)
        for n in range(N)
    )

    operands.append(
        weight.reshape(
            groups, out_channels // groups, in_channels // groups, *kernel_sizes
        )
    )

    return operands
