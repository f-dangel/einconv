"""Simplifications for tensor networks represented as einsum equation+operands."""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import Tensor, eye
from torch.nn import Parameter

from einconv import index_pattern
from einconv.expressions.utils import get_letters
from einconv.utils import cpu, get_conv_paddings


class Identity:
    """Represents a 2d Kronecker delta. Can be simplified by ``TensorNetwork``."""

    def __init__(
        self, dim: int, device: torch.device = cpu, dtype: torch.dtype = torch.float32
    ) -> None:
        """Store parameters required to create a tensor.

        Args:
            dim: Index dimension
            device: Device that will be used if a tensor representation is requested.
                Default: CPU
            dtype: Data type that will be used if a tensor representation is requested.
                Default: ``float32``.
        """
        self._dim = dim
        self._device = device
        self._dtype = dtype

    def to_tensor(self) -> Tensor:
        """Create a PyTorch tensor representing the identity matrix.

        Returns:
            Identity matrix of stored shape, device, and data type.
        """
        return eye(self._dim, device=self._device, dtype=self._dtype)

    def __repr__(self) -> str:
        """Create a human-readable representation of the object.

        Returns:
            Human-readable description.
        """
        return f"Identity({self._dim}, device={self._device}, dtype={self._dtype})"


class TensorNetwork:
    """Class for transforming tensor networks represented by einsum equations+operands.

    Intended usage:

    1) Construct an instance (``__init__``)

    2) Simplify the expression using ``simplify``. This applies simplifications based on
      the connectivity structure of sub-sampling and dense convolutions. This may slice
      some operands and introduce Kronecker deltas. Kronecker deltas will be used to
      further simplify the contraction.

    3) Request the final ``einsum`` equation and operands. The output may have different
      shape than the original contraction, but contains the same result tensor up to a
      reshape.
    """

    def __init__(
        self, equation: str, operands: List[Union[Tensor, Parameter, Identity]]
    ) -> None:
        """Store einsum equation and operands.

        Args:
            equation: A valid ``torch.einsum`` equation.
            operands: List of tensors acting as input for the einsum equation.
                Can also contain ``Identity`` objects that serve as placeholders for
                Kronecker deltas
        """
        self.input_indices = equation.split("->")[0].split(",")
        self.output_indices = equation.split("->")[1]
        self.operands = operands

        # set up mapping from indices to their dimension
        self.dims = {}
        for input_idxs, op in zip(self.input_indices, operands):
            if isinstance(op, Tensor):
                for idx, dim in zip(input_idxs, op.shape):
                    self.dims[idx] = dim
            if isinstance(op, Identity):
                for idx in input_idxs:
                    self.dims[idx] = op._dim

        self.output_shape = [self.dims[idx] for idx in self.output_indices]

    def simplify(self) -> None:
        """Analyze the operands and try to simplify the ``einsum`` expression."""
        self._simplify_downsampling()
        self._simplify_dense()
        self.prune_identities()
        self.merge_parallel()
        self.squeeze()

    def merge_parallel(self):
        """Merge parallel dimensions into a single dimension.

        This should allow using more specialized BLAS routines, according to
        https://github.com/google/TensorNetwork#optimized-contractions.
        """

        def contains_pair_or_neither(indices: str, pair: str) -> bool:
            """Check if a string contains pair-sequences or none of the pair entries.

            Args:
                indices: String to check.
                pair: Pair sequence of characters to check for.

            Returns:
                True if the string contains the pair sequence or neither of the pair.
            """
            assert len(pair) == 2
            return pair in indices or all(
                p not in indices.replace(pair, "") for p in pair
            )

        def next_mergable() -> str:
            """Find next pair of merge-able indices.

            Returns:
                Index pair that can be merged or empty string if none can be merged.
            """
            for indices in [idx for idx in self.input_indices if len(idx) > 1]:
                for i in range(len(indices) - 1):
                    pair = indices[i : i + 2]
                    can_merge = all(
                        contains_pair_or_neither(idxs, pair)
                        for idxs in self.input_indices + [self.output_indices]
                    )
                    if can_merge:
                        return pair
            return ""

        merge = next_mergable()
        while merge != "":
            self.group(list(merge))
            merge = next_mergable()

    def squeeze(self) -> None:
        """Try eliminating dimensions of size 1."""
        maybe_squeeze = [idx for idx, size in self.dims.items() if size == 1]

        for pos in range(len(self.input_indices)):
            for idx in maybe_squeeze:
                if (
                    idx in self.input_indices[pos]
                    and len(self.input_indices[pos]) > 1
                    and not isinstance(self.operands[pos], Identity)
                ):
                    idx_pos = self.input_indices[pos].index(idx)

                    self.input_indices[pos] = self.input_indices[pos].replace(idx, "")
                    self.operands[pos] = self.operands[pos].squeeze(idx_pos)

        for idx in maybe_squeeze:
            if idx in self.output_indices:
                self.output_indices = self.output_indices.replace(idx, "")

        indices = set("".join(self.input_indices) + self.output_indices)
        for idx in list(self.dims.keys()):
            if idx not in indices:
                self.dims.pop(idx)

    def generate_expression(self) -> Tuple[str, List[Union[Tensor, Parameter]]]:
        """Return (potentially simplified) information to carry out the contraction.

        Returns:
            Einsum equation
            Einsum operands
        """
        equation = "->".join([",".join(self.input_indices), self.output_indices])
        operands = []
        for op in self.operands:
            if isinstance(op, Identity):
                operands.append(op.to_tensor())
            else:
                operands.append(op)

        return equation, operands

    def prune_identities(self):
        """Remove Kronecker deltas that act as identity during the contraction."""
        self._remove_duplicate_identities()
        self._replace_self_contracting_identities()
        self._remove_contracting_identities()

    def _simplify_downsampling(self):
        def downsampling_positions():
            positions = []
            for pos, op in enumerate(self.operands):
                if hasattr(op, "_pattern_hyperparams"):
                    hyperparams = op._pattern_hyperparams
                    I = hyperparams["input_size"]  # noqa: E741
                    K = hyperparams["kernel_size"]
                    S = hyperparams["stride"]
                    P = hyperparams["padding"]
                    D = hyperparams["dilation"]

                    is_downsampling = False
                    # for string-valued padding, we currently only support cases where
                    # left and right padding coincide
                    if isinstance(P, str):
                        P_left, P_right = get_conv_paddings(K, S, P, D)
                        if P_left == P_right:
                            P = P_left
                            is_downsampling = S > K and P == 0 and D == 1 and I % S == 0
                    else:
                        is_downsampling = S > K and P == 0 and D == 1 and I % S == 0

                    if is_downsampling:
                        _, _, i_idx = self.input_indices[pos]
                        if i_idx not in self.output_indices:
                            positions.append(pos)
            return positions

        while downsampling_positions():
            pos = downsampling_positions()[0]
            _, _, i_idx = self.input_indices[pos]

            op = self.operands[pos]
            I = op._pattern_hyperparams["input_size"]  # noqa: E741
            S = op._pattern_hyperparams["stride"]
            K = op._pattern_hyperparams["kernel_size"]
            P = op._pattern_hyperparams["padding"]
            D = op._pattern_hyperparams["dilation"]

            i_prime, k_prime = self.ungroup(i_idx, [I // S, S])
            self.narrow(k_prime, 0, K)
            self.group([i_prime, k_prime])

            self.operands[pos] = index_pattern(
                (K * I) // S,
                K,
                stride=K,
                padding=P,
                dilation=D,
                device=op.device,
                dtype=op.dtype,
            )

    def _simplify_dense(self):
        def is_dense_pattern(op: Union[Tensor, Parameter, Identity]) -> bool:
            if not hasattr(op, "_pattern_hyperparams"):
                return False
            hyperparams = op._pattern_hyperparams
            I = hyperparams["input_size"]  # noqa: E741
            K = hyperparams["kernel_size"]
            S = hyperparams["stride"]
            P = hyperparams["padding"]
            D = hyperparams["dilation"]

            is_dense = False
            # for string-valued padding, we currently only support cases where
            # left and right padding coincide
            if isinstance(P, str):
                P_left, P_right = get_conv_paddings(K, S, P, D)
                if P_left == P_right:
                    P = P_left
                    is_dense = (
                        (I + 2 * P - (K + (K - 1) * (D - 1))) % S == 0
                        and K == S
                        and P == 0
                        and D == 1
                    )
            else:
                is_dense = (
                    (I + 2 * P - (K + (K - 1) * (D - 1))) % S == 0
                    and K == S
                    and P == 0
                    and D == 1
                )

            return is_dense

        while any(is_dense_pattern(op) for op in self.operands):
            idx = [is_dense_pattern(op) for op in self.operands].index(True)

            op = self.operands[idx]
            assert isinstance(op, Tensor)
            I, K = (
                op._pattern_hyperparams["input_size"],
                op._pattern_hyperparams["kernel_size"],
            )
            K_idx, O_idx, I_idx = self.input_indices[idx]
            I_tilde_idx, K_tilde_idx = self.ungroup(I_idx, [I // K, K])

            self.operands[idx] = Identity(I // K, device=op.device, dtype=op.dtype)
            self.operands.insert(idx + 1, Identity(K, device=op.device, dtype=op.dtype))

            self.input_indices[idx] = O_idx + I_tilde_idx
            self.input_indices.insert(idx + 1, K_idx + K_tilde_idx)

    def group(self, idxs: List[str]) -> str:
        """Group multiple indices into one.

        Args:
            idxs: List of characters representing the indices that will be grouped.
                Must be in the order they will be grouped.

        Returns:
            Name of the grouped index.
        """
        joined = "".join(idxs)
        group_dim = Tensor([self.dims[i] for i in idxs]).int().prod().item()
        group_idx = idxs[0]

        for indices in self.input_indices + [self.output_indices]:
            if joined in indices:
                assert all(i not in indices.replace(joined, "") for i in idxs)

        for i, indices in enumerate(self.input_indices):
            self.input_indices[i] = self.input_indices[i].replace(joined, group_idx)
            if joined in indices:
                pos = indices.index(joined)
                self.operands[i] = self.operands[i].flatten(
                    start_dim=pos, end_dim=pos + len(idxs) - 1
                )

        self.output_indices = self.output_indices.replace(joined, group_idx)

        self.dims[group_idx] = group_dim
        for idx in idxs[1:]:
            self.dims.pop(idx)

        return group_idx

    def narrow(self, idx: str, start: int, length: int):
        """Narrow the range of an index in the tensor network.

        Args:
            idx: Name of the index to be narrowed.
            start: Starting position of the index range that will be kept.
            length: Length of the range that will be kept.
        """
        assert start >= 0

        for i, indices in enumerate(self.input_indices):
            if idx in indices:
                dim = indices.index(idx)
                self.operands[i] = self.operands[i].narrow(dim, start, length)

        self.dims[idx] = length

    def ungroup(self, idx: str, size: List[int]) -> List[str]:
        """Ungroup an index into multiple indices.

        Args:
            idx: Name of the index to be ungrouped.
            size: List of sizes of the new indices. Must multiply to size of ``idx``.

        Returns:
            List of names of the ungrouped indices.
        """
        assert self.dims[idx] == Tensor(size).prod()

        # reshape operands
        for i, input_idx in enumerate(self.input_indices):
            if idx in input_idx:
                pos = input_idx.index(idx)
                current_shape = [self.dims[idx] for idx in input_idx]
                new_shape = current_shape[:pos] + list(size) + current_shape[pos + 1 :]
                self.operands[i] = self.operands[i].reshape(new_shape)

        # rename in equation
        new_idxs = [idx] + get_letters(
            len(size) - 1, blocked=set("".join(self.input_indices))
        )

        self.input_indices = [
            input_idx.replace(idx, "".join(new_idxs))
            for input_idx in self.input_indices
        ]
        self.output_indices = self.output_indices.replace(idx, "".join(new_idxs))

        # update dimensions
        for i, s in zip(new_idxs, size):
            self.dims[i] = s

        return new_idxs

    def _replace_self_contracting_identities(self):
        def prunable_positions():
            positions = [
                pos for pos, op in enumerate(self.operands) if isinstance(op, Identity)
            ]
            prunable = []
            for pos in positions:
                i, j = self.input_indices[pos]
                without = [idx for p, idx in enumerate(self.input_indices) if p != pos]
                self_contracts = not any(
                    i in idx or j in idx for idx in without + [self.output_indices]
                )
                if self_contracts:
                    prunable.append(pos)

            return prunable

        while prunable_positions():
            pos = prunable_positions()[0]
            i, j = self.input_indices[pos]
            I = self.operands[pos]  # noqa: E741

            self.input_indices[pos] = i
            self.operands[pos] = Tensor([I._dim], device=I._device).to(I._device)
            self.dims[i] = 1
            self.dims.pop(j)

    def _remove_contracting_identities(self):
        def prunable_positions():
            positions = [
                pos for pos, op in enumerate(self.operands) if isinstance(op, Identity)
            ]
            prunable = []
            for pos in positions:
                i, j = self.input_indices[pos]
                without = [idx for p, idx in enumerate(self.input_indices) if p != pos]
                contracts_other = any(i in idx or j in idx for idx in without)
                if contracts_other:
                    prunable.append(pos)

            return prunable

        while prunable_positions():
            pos = prunable_positions()[0]
            i, j = self.input_indices[pos]

            self.operands.pop(pos)
            self.input_indices.pop(pos)

            self.input_indices = [idx.replace(i, j) for idx in self.input_indices]
            self.output_indices = self.output_indices.replace(i, j)
            self.dims.pop(i)

    def _remove_duplicate_identities(self):
        positions = [
            pos for (pos, op) in enumerate(self.operands) if isinstance(op, Identity)
        ]
        indices = [set(self.input_indices[pos]) for pos in positions]
        occurences = {tuple(sorted(idxs)): indices.count(idxs) for idxs in indices}

        drop_positions = []

        for key, occ in occurences.items():
            if occ > 1:
                duplicate_positions = [
                    pos for pos, idxs in enumerate(indices) if set(key) == idxs
                ]
                drop_positions += duplicate_positions[1:]

        drop_positions.sort()
        for drop_pos in drop_positions:
            self.operands.pop(drop_pos)
            self.input_indices.pop(drop_pos)

    def __repr__(self):
        """Create a human-readable representation of the object.

        Returns:
            Human-readable description.
        """
        return (
            f"TensorNetwork('{','.join(self.input_indices)}->{self.output_indices}',"
            + f" {self.operands}, dims={self.dims})"
        )
