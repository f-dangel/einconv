"""PyTorch equivalent of ``nn.Unfold`` implemented as einsum."""
from typing import Tuple, Union

from torch import Tensor
from torch.nn import Module

from einconv.functionals import unfoldNd


class UnfoldNd(Module):
    """PyTorch module for N-dimensional input unfolding (im2col) that uses einsum."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        stride: Union[int, Tuple[int, ...]] = 1,
        simplify: bool = True,
    ):
        """Extracts sliding local blocks from a batched input tensor (``im2col``).

        This module accepts batched tensors with ``N`` spatial dimensions. It acts like
        ``torch.nn.Unfold`` for a 4d input (batched images), but works for arbitrary
        ``N``. See https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.

        Args:
            kernel_size: Kernel dimensions. Can be a single integer (shared along all
                spatial dimensions), or an ``N``-tuple of integers.
            dilation: Dilation of the convolution. Can be a single integer (shared along
                all spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
            padding: Padding of the convolution. Can be a single integer (shared along
                all spatial dimensions), an ``N``-tuple of integers, or a string.
                Default: ``0``. Allowed strings are ``'same'`` and ``'valid'``.
            stride: Stride of the convolution. Can be a single integer (shared along all
                spatial dimensions), or an ``N``-tuple of integers. Default: ``1``.
            simplify: Whether to use a simplified einsum expression. Default: ``True``.
        """
        super().__init__()

        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride
        self._simplify = simplify

    def forward(self, x: Tensor) -> Tensor:
        """Compute the unfolded input.

        Args:
            x: Convolution input. Has shape ``[batch_size, in_channels, *input_sizes]``
                where ``len(input_sizes) == N``.

        Returns:
            Unfolded input. Has shape \
            ``[batch_size, in_channels * tot_kernel_size, tot_output_size]`` where \
            ``tot_kernel_size`` is the kernel dimension product and \
            ``tot_output_size`` is the product of the output spatial dimensions. In \
            ``einops`` notation, the index structure is \
            ``n (c_in k1 k2 ...) (o1 o2 ...)``.
        """
        return unfoldNd(
            x,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
            simplify=self._simplify,
        )
