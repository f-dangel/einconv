"""Symbolic TN simplification. Does not require tensor representation.

This allows to simplify contractions once and cache them.
"""

from typing import Callable, List, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, get_default_dtype

from einconv.utils import cpu


class SymbolicTensor:
    """Class for symbolic representation and manipulation of tensors."""

    def __init__(self, name: str, shape: Tuple[int, ...], indices: Tuple[str, ...]):
        """Store tensor name, dimensions, and axes names.

        Args:
            name: Tensor name.
            shape: Tensor dimensions.
            indices: Name of each axis.

        Raises:
            ValueError: If the number of dimensions does not match the number of axes.
        """
        if len(shape) != len(indices):
            raise ValueError(
                f"Shape {shape} of length {len(shape)} must have same length as "
                + f" indices {indices} of length ({len(indices)})."
            )
        self.name = name
        self.shape = shape
        self.indices = indices

        # human-readable record of transformations
        self.history: List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]] = [
            ("initial", shape, indices)
        ]
        # record of functional transformations
        self.transforms: List[Callable[[Tensor], Tensor]] = []

    def instantiate(
        self,
        tensor: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Instantiate the symbolic tensor with a concrete tensor.

        Args:
            tensor: Concrete tensor. Can be None if the symbolic tensor represents
                a special object that can be instantiated with calls to an external
                library (for instance an identity matrix).
            device: Device to instantiate the tensor on. If ``None`` and ``tensor``
                is specified, the device of ``tensor`` is used. If specified, the passed
                device will be used. Otherwise, the tensor will be instantiated on CPU.
            dtype: Data type to instantiate the tensor with. If ``None`` and ``tensor``
                is specified, the data type of ``tensor`` is used. If specified, the
                passed data type will be used. Otherwise, the tensor will be
                instantiated with PyTorch's current default data type.

        Returns:
            The instantiated tensor.

        Raises:
            ValueError: If tensor is unspecified.
        """
        if tensor is None:
            raise ValueError("Require tensor data to instantiate.")
        self._check_transformed_tensor(tensor)

        target_device = cpu if tensor is None else tensor.device
        target_device = device if device is not None else target_device

        target_dtype = get_default_dtype() if tensor is None else tensor.dtype
        target_dtype = dtype if dtype is not None else target_dtype

        if tensor.device != target_device or tensor.dtype != target_dtype:
            tensor = tensor.to(target_device, target_dtype)

        for stage_idx, transform in enumerate(self.transforms, start=1):
            tensor = transform(tensor)
            self._check_transformed_tensor(tensor, stage_idx=stage_idx)

        return tensor

    def _check_transformed_tensor(self, tensor: Tensor, stage_idx: int = 0):
        """Verify that a tensor has the expected shape at an intermediate stage.

        Args:
            tensor: Tensor after processing of stage `stage_idx`.
            stage_idx: Transformation stage. `0` means initial stage.  Default: `0`.

        Raises:
            ValueError: If the tensor has an unexpected shape.
        """
        stage, shape, indices = self.history[stage_idx]
        if tensor.shape != shape:
            raise ValueError(
                f"Tensor after stage {stage_idx} ({stage}) expected to have shape "
                + f"{shape} (got {tuple(tensor.shape)}) with axes named {indices}."
            )

    def group(self, indices: Tuple[str, ...]):
        """Combine multiple indices into a single index.

        Args:
            indices: Indices to group.

        Raises:
            NotImplementedError: If the indices are not consecutive.
            ValueError: If the new index name already exists.
        """
        pos = self.indices.index(indices[0])
        if indices != self.indices[pos : pos + len(indices)]:
            raise NotImplementedError(
                f"Only consecutive indices can be grouped. Got {indices} but axes "
                + f"are {self.indices}."
            )

        group_name = "(" + " ".join(indices) + ")"
        if group_name in self.indices:
            raise ValueError(f"Index {group_name} already exists.")

        # determine dimension and indices of grouped tensor
        group_dim = 1
        for dim in self.shape[pos : pos + len(indices)]:
            group_dim *= dim

        new_indices = (
            self.indices[:pos] + (group_name,) + self.indices[pos + len(indices) :]
        )
        new_shape = self.shape[:pos] + (group_dim,) + self.shape[pos + len(indices) :]

        # construct transform and update internal state
        equation = f"{' '.join(self.indices)} -> {' '.join(new_indices)}"

        def apply_grouping(tensor: Tensor) -> Tensor:
            """Group the specified axes into a single one.

            Args:
                tensor: Tensor to group axes of.

            Returns:
                Tensor with grouped axes.
            """
            return rearrange(tensor, equation)

        self.history.append(
            (f"group {indices} into {group_name!r}", new_shape, new_indices)
        )
        self.transforms.append(apply_grouping)
        self.indices = new_indices
        self.shape = new_shape

    def __repr__(self) -> str:
        """Return a string representation of the symbolic tensor.

        Returns:
            String representation of the symbolic tensor, including its transformation
            history.
        """
        as_str = f"SymbolicTensor({self.name!r}, {self.shape}, {self.indices})"

        as_str += "\nTransformations:"
        for idx, (info, shape, indices) in enumerate(self.history):
            as_str += "\n\t- "
            as_str += f"({idx}) {info}: shape {shape}, indices {indices}"
        return as_str
