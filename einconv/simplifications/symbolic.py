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
            indices: Name of each axis. Each name must be unique.
        """
        self.name = name
        self.shape = shape
        self.indices = indices

        # human-readable record of transformations
        self.history: List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]] = [
            ("initial", shape, indices)
        ]
        # record of functional transformations
        self.transforms: List[Callable[[Tensor], Tensor]] = []

        self._check_state_valid()

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

        def apply_group(tensor: Tensor) -> Tensor:
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
        self.transforms.append(apply_group)
        self.indices = new_indices
        self.shape = new_shape

        self._check_state_valid()

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

    def narrow(self, index: str, start: int, length: int):
        """Narrow an index to a specified range.

        Args:
            index: Name of the index to narrow.
            start: Start of the range.
            length: Length of the range.

        Raises:
            ValueError: If the start or length are invalid.
        """
        if start < 0:
            raise ValueError(f"Start of range must be non-negative. Got {start}.")
        if length <= 0:
            raise ValueError(f"Length of range must be positive. Got {length}.")

        pos = self.indices.index(index)
        end = start + length - 1

        if end >= self.shape[pos]:
            raise ValueError(
                f"Range [{start}, {end}] exceeds the length of index {index} "
                + f"({self.shape[pos]})."
            )

        new_shape = self.shape[:pos] + (length,) + self.shape[pos + 1 :]

        # construct transform and update internal state
        def apply_narrow(tensor: Tensor) -> Tensor:
            """Narrow the specified index to the specified range.

            Args:
                tensor: Tensor to narrow.

            Returns:
                Narrowed tensor.
            """
            return tensor.narrow(pos, start, length)

        self.history.append(
            (
                f"narrow {index!r} from {start} to including {end}",
                new_shape,
                self.indices,
            )
        )
        self.transforms.append(apply_narrow)
        self.shape = new_shape

        self._check_state_valid()

    def rename(self, old: str, new: str):
        """Rename an index.

        Args:
            old: Name of the index to rename.
            new: New name of the index.

        Raises:
            ValueError: If the new index name already exists.
        """
        pos = self.indices.index(old)

        if new in self.indices:
            raise ValueError(f"New index name {new!r} already in use.")

        new_indices = self.indices[:pos] + (new,) + self.indices[pos + 1 :]

        # create transformation and update internal state
        def apply_rename(tensor: Tensor) -> Tensor:
            """Rename the specified index.

            Renaming leave a tensor unchanged.

            Args:
                tensor: Tensor to rename.

            Returns:
                The original tensor.
            """
            return tensor

        self.history.append((f"rename {old!r} to {new!r}", self.shape, new_indices))
        self.indices = new_indices

        self._check_state_valid()

    def _check_state_valid(self):
        """Verify that the internal state is valid.

        Indices must have unique names, and have the same length as shape.

        Raises:
            ValueError: If the internal state is invalid.
        """
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"Indices must be unique. Got {self.indices}.")
        if len(self.shape) != len(self.indices):
            raise ValueError(
                f"Shape {self.shape} of length {len(self.shape)} must have same length "
                + f"as indices {self.indices} of length ({len(self.indices)})."
            )

    def ungroup(
        self,
        index: str,
        sizes: Tuple[int, ...],
        new_names: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[str, ...]:
        """Un-group an index into multiple indices.

        Args:
            index: Index to ungroup.
            size: Sizes of the new indices.
            new_names: Names of the new indices. If `None`, new names are generated
                automatically. Defaults to `None`.

        Returns:
            Names of the new indices.

        Raises:
            ValueError: If the supplied or auto-generated new index names are already
                in use.
            ValueError: If the supplied sizes do not match the size of the index.
        """
        if new_names is None:
            new_names = tuple(f"{index}_{i}" for i in range(len(sizes)))

        if any(new_name in self.indices for new_name in new_names):
            raise ValueError(
                f"One of the new index names {new_names} is already in use. "
                + f"Indices: {self.indices}."
            )

        dim = 1
        for size in sizes:
            dim *= size

        pos = self.indices.index(index)
        if dim != self.shape[pos]:
            raise ValueError(
                f"Size of new indices {sizes} ({dim}) must match size of index "
                + f"{index} ({self.shape[pos]})."
            )

        # create transform and update inner state
        shape_info = dict(zip(new_names, sizes))
        equation = (
            f"{' '.join(self.indices[:pos])} ({' '.join(new_names)}) "
            + f"{' '.join(self.indices[pos + 1:])} -> "
            + f"{' '.join(self.indices[:pos] + new_names + self.indices[pos + 1:])}"
        )

        def apply_ungroup(tensor: Tensor) -> Tensor:
            """Un-group the specified index.

            Args:
                tensor: Tensor to un-group.

            Returns:
                Ungrouped tensor.
            """
            return rearrange(tensor, equation, **shape_info)

        new_indices = self.indices[:pos] + new_names + self.indices[pos + 1 :]
        new_shape = self.shape[:pos] + sizes + self.shape[pos + 1 :]

        self.history.append(
            (
                f"ungroup {index!r} into {sizes} {new_names}",
                new_shape,
                new_indices,
            )
        )
        self.transforms.append(apply_ungroup)
        self.indices = new_indices
        self.shape = new_shape

        self._check_state_valid()

        return new_names
