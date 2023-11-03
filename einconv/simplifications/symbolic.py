"""Symbolic TN simplification. Does not require tensor representation.

This allows to simplify contractions once and cache them.
"""

from typing import Callable, List, Optional, Tuple

import torch
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
        """
        if len(shape) != len(indices):
            raise ValueError(
                f"Shape {shape} of length {len(shape)} must have same length as "
                + f" indices {indices} of length ({len(indices)})."
            )
        self.name = name
        self.shape = shape
        self.indices = indices

        self.history: List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]] = [
            ("initial", shape, indices)
        ]
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
                passed data type will be used. Otherwise, the tensor will be instantiated
                with PyTorch's current default data type.

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
