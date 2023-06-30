from typing import Tuple, Union

from torch import Tensor
from torch.nn import Module

from einconv.functionals import unfoldNd


class UnfoldNd(Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    PyTorch module that accepts Nd tensors. Acts like ``torch.nn.Unfold``
    for a 4d input. Uses tensor networks under the hood.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        stride: Union[int, Tuple[int, ...]] = 1,
    ):  # noqa: D107
        super().__init__()

        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input: Tensor) -> Tensor:  # noqa: D102
        return unfoldNd(
            input,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )
