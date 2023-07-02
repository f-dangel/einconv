"""Compare forward pass of a 2d convolution layer."""

from torch import allclose, manual_seed, rand
from torch.nn import Conv2d

from einconv.modules import ConvNd

manual_seed(0)  # make deterministic

x = rand(10, 4, 28, 28)  # random input
conv_params = {
    "in_channels": 4,
    "out_channels": 8,
    "kernel_size": 4,  # can also use tuple
    "padding": 1,  # can also use tuple, or string
    "stride": 3,  # can also use tuple
    "dilation": 2,  # can also use tuple
    "groups": 2,
    "bias": True,
}
N = 2  # convolution dimension

torch_layer = Conv2d(**conv_params)
ein_layer = ConvNd(N, **conv_params)
ein_layer.weight.data = torch_layer.weight.data
ein_layer.bias.data = torch_layer.bias.data

assert allclose(torch_layer(x), ein_layer(x), atol=1e-7)
