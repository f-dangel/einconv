"""PyTorch equivalent of ``nn.Conv{1,2,3}d`` implemented as einsum."""

from __future__ import annotations

from math import sqrt
from typing import Tuple, Union

import torch
from torch import Tensor, empty
from torch.nn import Conv1d, Conv2d, Conv3d, Module, Parameter, init

from einconv.functionals import convNd
from einconv.utils import _tuple, sync_parameters


class ConvNd(Module):
    """PyTorch module for N-dimensional convolution that uses einsum."""

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
    def from_nn_Conv(cls, conv_module: Union[Conv1d, Conv2d, Conv3d]) -> ConvNd:
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

    def forward(self, x: Tensor) -> Tensor:
        """Perform convolution on the input.

        Args:
            x: Convolution input. Has shape
                ``[batch_size, in_channels, *spatial_in_dims]`` where
                ``len(spatial_in_dims)==N``.

        Returns:
            Result of the convolution. Has shape
                ``[batch_size, out_channels, *spatial_out_dims]`` where
                ``len(spatial_out_dims)==N``.
        """
        return convNd(
            x,
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