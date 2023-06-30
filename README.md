# <img alt="Einconv:" src="./docs/logo.png" height="90"> Convolutions Through the Lens of Tensor Networks

This package offers `einsum`-based implementations of convolutions and related
operations in PyTorch.

**Disclaimer:** The package name is inspired by
[this](https://github.com/pfnet-research/einconv) Github repository which
represented the starting point for our work.

## Installation
Install from PyPI via `pip`

```sh
pip install einconv
```

## Example

Try running the [basic
example](https://github.com/f-dangel/einconv/blob/master/docs/tutorials/basic_conv2d.py).

More tutorials are [TODO here]().

## Features & Usage

In general, `einconv`'s goals are:

- Full hyper-parameter support (stride, padding, dilation, groups, etc.)
- Support for any dimension (e.g. 5d-convolution)
- Optimizations via symbolic simplification

### Modules

`einconv` provides `einsum`-based implementations of the following PyTorch modules:

| `torch` module    | `einconv` module   |
|-------------------|--------------------|
| `nn.Conv{1,2,3}d` | `modules.ConvNd`   |
| `nn.Unfold`       | `modules.UnfoldNd` |

They work in exactly the same way as their PyTorch equivalents.

### Functionals

`einconv` provides `einsum`-based implementations of the following PyTorch functionals:

| `torch` functional           | `einconv` functional   |
|------------------------------|------------------------|
| `nn.functional.conv{1,2,3}d` | `functionals.convNd`   |
| `nn.functional.unfold`       | `functionals.unfoldNd` |

They work in exactly the same way as their PyTorch equivalents.

### Einsum Expressions
`einconv` can generate `einsum` expressions (equation, operands, and output
shape) for the following operations:

- Forward pass of `N`-dimensional convolution
- Backward pass (input and weight VJPs) of `N`-dimensional convolution
- Input unfolding (`im2col/unfold`) for inputs of `N`-dimensional convolution

These can then be evaluated with `einsum`:
```py
from torch import einsum
from einconv.expressions import conv_forward

equation, operands, final_shape = conv_forward.einsum_expression(...)
result = einsum(equation, *operands).reshape(final_shape)
```

### Symbolic Simplification

Some operations (e.g. dense convolutions) can be optimized via symbolic simplifications:
```py
from einconv import simplify
from torch import allclose

equation_opt, operands_opt, final_shape = simplify(equation, operands)
# alternatively:
# equation_opt, operands_opt, final_shape = conv_forward.einsum_expression(..., simplify=True)
result_opt = einsum(equation_opt, *operands_opt).reshape(final_shape)

allclose(result, result_opt) # True
```

## Citation

If you find the `einconv` package useful for your research, consider mentioning
the accompanying article

```bib

@article{dangel2023convolutions,
  title =        {Convolutions Through the Lens of Tensor Networks},
  author =       {Dangel, Felix},
  year =         2023,
}

```
## Limitations

Under preparation
