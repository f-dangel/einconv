"""Utility functions for dealing with third-party implementation of convolution."""

from os.path import abspath, dirname, join
from sys import path
from typing import Callable, Union

from torch import Tensor
from torch.nn import Parameter

from einconv.einconvnd import EinconvNd

# make third-party implementation importable
HERE = abspath(__file__)
REPO_ROOT_DIR = dirname(dirname(HERE))
THIRD_PARTY_DIR = join(REPO_ROOT_DIR, "third_party", "pytorch_convNd")
path.append(THIRD_PARTY_DIR)
from convNd import convNd as Conv_Nd_third_party  # noqa: E402

path.remove(THIRD_PARTY_DIR)


def to_ConvNd_third_party(einconv_module: EinconvNd) -> Conv_Nd_third_party:
    """Create third-party convolution layer for high-dimensional convolutions.

    Args:
        einconv_module: Einconv layer.

    Returns:
        Third-party convolution layer for high-dimensional convolutions. Should
        produce the same result in the forward pass.

    Raises:
        NotImplementedError: For unsupported convolution dimensions.
    """
    bias_initializer: Union[None, Callable[[Parameter], None]] = None
    if einconv_module.bias is not None:

        def bias_initializer(bias: Parameter) -> None:
            bias.data = einconv_module.bias.data.clone()

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
        use_bias=einconv_module.bias is not None,
        bias_initializer=bias_initializer,
    )

    if einconv_module.N <= 3:
        raise NotImplementedError
    if einconv_module.N != 4:
        raise NotImplementedError

    for idx, layer in enumerate(module.conv_layers):
        chunk = einconv_module.weight[:, :, idx, :, :]
        if chunk.shape != layer.weight.shape:
            raise ValueError("Attempt of invalid chunking.")
        if chunk.device != layer.weight.device:
            raise ValueError("Inconsistent devices.")
        if chunk.dtype != layer.weight.dtype:
            raise ValueError("Inconsistent dtypes.")
        layer.weight.data = chunk.data.clone()

    return module
