"""Utility functions for dealing with third-party implementation of convolution."""

from os.path import abspath, dirname, join
from sys import path
from typing import List, Union

from torch import Tensor
from torch.nn import Conv3d, Module, Parameter, Sequential

from einconv.einconvnd import EinconvNd

# make third-party implementation importable
HERE = abspath(__file__)
REPO_ROOT_DIR = dirname(dirname(HERE))
THIRD_PARTY_DIR = join(REPO_ROOT_DIR, "third_party", "pytorch_convNd")
path.append(THIRD_PARTY_DIR)
from convNd import convNd as Conv_Nd_third_party  # noqa: E402

path.remove(THIRD_PARTY_DIR)


def to_ConvNd_third_party(einconv_module: EinconvNd) -> Sequential:
    """Create third-party convolution layer for high-dimensional convolutions.

    Args:
        einconv_module: Einconv layer.

    Returns:
        Convolution layer for high-dimensional convolutions. Uses a third-
        party implementation under the hood.

    Raises:
        NotImplementedError: For unsupported convolution dimensions.
    """
    if einconv_module.N <= 3:
        raise NotImplementedError("Parameter syncing not implemented for N<=3.")

    module = Conv_Nd_third_party(
        einconv_module.in_channels,
        einconv_module.out_channels,
        einconv_module.N,
        einconv_module.kernel_size,
        einconv_module.stride,
        einconv_module.padding,
        padding_mode=einconv_module.padding_mode,
        dilation=einconv_module.dilation,
        groups=einconv_module.groups,
        use_bias=False,
    )

    # sync weights
    weight_flat = einconv_module.weight
    weight_flat = weight_flat.reshape(
        *weight_flat.shape[:2], -1, *weight_flat.shape[-3:]
    )
    conv_layers = _flattened_conv_layers(module)

    for idx, layer in enumerate(conv_layers):
        chunk = weight_flat[:, :, idx, :, :]
        if chunk.shape != layer.weight.shape:
            raise ValueError("Attempt of invalid chunking.")
        if chunk.device != layer.weight.device:
            raise ValueError("Inconsistent devices.")
        if chunk.dtype != layer.weight.dtype:
            raise ValueError("Inconsistent dtypes.")
        layer.weight.data = chunk.data.clone()

    sequential = Sequential(module)

    # add bias if necessary
    if einconv_module.bias is not None:
        bias_layer = ConvBias(einconv_module.out_channels).to(weight_flat.device)
        bias_layer.bias.data = einconv_module.bias.data.clone()
        sequential.extend([bias_layer])

    return sequential


def _flattened_conv_layers(
    third_party: Union[Conv_Nd_third_party, Conv3d]
) -> List[Conv3d]:
    """Flatten the 3d convolutions called by this implementation into a list.

    Args:
        third_party: Layer of third-party convolution layer or torch 3d conv layer.

    Returns:
        Flattened 3d convolutions that are recursively called by this
        implementation.
    """
    conv_layers = []
    if isinstance(third_party, Conv_Nd_third_party):
        for layer in third_party.conv_layers:
            conv_layers += _flattened_conv_layers(layer)
    else:
        conv_layers.append(third_party)

    return conv_layers


class ConvBias(Module):
    """Adds a bias to the channel (second) dimension of an input."""

    def __init__(self, num_channels: int) -> None:
        """Initialize the bias layer.

        Args:
            num_channels: Number of channels.
        """
        super().__init__()
        self.register_parameter("bias", Parameter(Tensor(num_channels)))

    def forward(self, x: Tensor) -> Tensor:
        """Add bias to the channel dimension.

        Args:
            x: Input to the layer. Must have a batch axis.

        Returns:
            Input, added with a bias.
        """
        bias_expanded = self.bias.unsqueeze(0)
        for _ in range(x.dim() - 2):
            bias_expanded = bias_expanded.unsqueeze(-1)

        return x + bias_expanded.expand_as(x)
