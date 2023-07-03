"""Test ``einconv.simplifications.opt``."""


from torch import allclose, einsum, float32, manual_seed, rand

from einconv import index_pattern
from einconv.simplifications.opt import Identity, TensorNetwork


def test_prune_one_identity():
    """Test pruning of an expression with one identity."""
    manual_seed(0)
    I = Identity(10)  # noqa: E741
    v = rand(10)
    equation = "ij,j->i"
    truth = einsum(equation, I.to_tensor(), v)

    tn = TensorNetwork(equation, [I, v])
    tn.prune_identities()
    equ, ops = tn.generate_expression()
    assert equ == "j->j"
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_prune_two_identities():
    """Test pruning of an expression with two different identities."""
    manual_seed(0)
    I1, I2 = Identity(10), Identity(10)
    v = rand(10)
    equation = "ij,jk,k->i"
    truth = einsum(equation, I1.to_tensor(), I2.to_tensor(), v)

    tn = TensorNetwork(equation, [I1, I2, v])
    tn.prune_identities()
    equ, ops = tn.generate_expression()
    assert equ == "k->k"
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_remove_duplicate_identities():
    """Test removal of two identical identities."""
    manual_seed(0)
    I1, I2 = Identity(10), Identity(10)
    v = rand(10)
    equation = "ij,ij,j->i"
    truth = einsum(equation, I1.to_tensor(), I2.to_tensor(), v)

    tn = TensorNetwork(equation, [I1, I2, v])
    tn._remove_duplicate_identities()
    equ, ops = tn.generate_expression()
    assert equ == "ij,j->i"
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_prune_unprunable_identity():
    """Test if identities that cannot be removed are kept."""
    manual_seed(0)
    I1, I2 = Identity(10), Identity(10)
    equation = "ij,kl->ijkl"
    # contracts with nothing
    truth = einsum(equation, I1.to_tensor(), I2.to_tensor())

    tn = TensorNetwork(equation, [I1, I2])
    tn.prune_identities()
    equ, ops = tn.generate_expression()
    assert equ == equation
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_replace_self_contracting_identities():
    """Test replacement of identities that contract with nothing but themselves."""
    manual_seed(0)
    I1, I2 = Identity(10), Identity(10)
    # second identity contracts to one
    equation = "ij,kl->ij"
    truth = einsum(equation, I1.to_tensor(), I2.to_tensor())

    tn = TensorNetwork(equation, [I1, I2])
    tn._replace_self_contracting_identities()
    equ, ops = tn.generate_expression()
    assert equ == "ij,k->ij"
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_ungroup():
    """Test ungrouping an index into multiple ones."""
    manual_seed(0)
    A = rand((3, 10))
    B = rand((10, 4))
    equation = "ij,jk->ik"
    truth = einsum(equation, A, B)

    tn = TensorNetwork(equation, [A, B])
    tn.ungroup("j", [2, 5])
    equ, ops = tn.generate_expression()

    assert ops[0].shape == (3, 2, 5)
    assert ops[1].shape == (2, 5, 4)
    assert equ != equation
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))

    tn = TensorNetwork(equation, [A, B])
    tn.ungroup("j", [2, 5])
    tn.ungroup("k", [2, 2])
    equ, ops = tn.generate_expression()

    assert ops[0].shape == (3, 2, 5)
    assert ops[1].shape == (2, 5, 2, 2)
    assert equ != equation
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_simplify_dense():
    """Test simplification of a TN with an index pattern from a dense convolution."""
    pattern = index_pattern(
        input_size=10, kernel_size=2, stride=2, padding=0, dilation=1
    )
    equation = "koi->koi"
    operands = [pattern]
    truth = einsum(equation, *operands)

    tn = TensorNetwork(equation, operands)
    tn._simplify_dense()
    assert len(tn.operands) == 2
    assert all(isinstance(op, Identity) for op in tn.operands)

    equ, ops = tn.generate_expression()

    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_simplify_downsampling():
    """Test simplification of TN with index pattern from down-sampling convolution."""
    manual_seed(0)
    pattern = index_pattern(
        input_size=9, kernel_size=2, stride=3, padding=0, dilation=1, dtype=float32
    )
    v = rand(9)
    equation = "i,koi->ko"
    truth = einsum(equation, v, pattern)

    # downsampling simplification
    tn = TensorNetwork(equation, [v, pattern])
    tn._simplify_downsampling()
    equ, ops = tn.generate_expression()

    assert len(ops) == 2
    assert equ == equation
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_simplify_downsampling_dense_prune():
    """Check simplification chain of a down-sampling convolution."""
    manual_seed(0)
    pattern = index_pattern(
        input_size=9, kernel_size=2, stride=3, padding=0, dilation=1, dtype=float32
    )
    v = rand(9)
    equation = "i,koi->ko"
    truth = einsum(equation, v, pattern)

    # downsampling + dense + zeros simplification
    tn = TensorNetwork(equation, [v, pattern])
    tn._simplify_downsampling()
    tn._simplify_dense()
    tn.prune_identities()
    equ, ops = tn.generate_expression()

    assert len(ops) == 1
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_squeeze():
    """Test squeezing of dimensions with unit dimension."""
    manual_seed(0)
    A = rand(1, 1)
    B = rand(1, 9)
    v = rand(9)
    equation = "ij,jk,k->i"
    truth = einsum(equation, A, B, v)

    tn = TensorNetwork(equation, [A, B, v])
    tn.squeeze()
    equ, ops = tn.generate_expression()

    assert len(ops) == 3
    assert ops[0].shape == (1,)
    assert ops[1].shape == (9,)
    assert ops[2].shape == (9,)
    assert equ == "j,k,k->"
    assert "i" not in tn.dims
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_merge_parallel_in_output():
    """Test merging of parallel legs that occur in output."""
    manual_seed(0)
    A = rand(5, 9, 8)
    v = rand(5)
    B = rand(5, 9, 8)
    equation = "ijk,i,ijk->ijk"
    truth = einsum(equation, A, v, B)

    tn = TensorNetwork(equation, [A, v, B])
    tn.merge_parallel()
    equ, ops = tn.generate_expression()

    assert len(ops) == 3
    assert ops[0].shape == (5, 72)
    assert ops[1].shape == (5,)
    assert ops[2].shape == (5, 72)
    assert equ == "ij,i,ij->ij"
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))


def test_merge_parallel():
    """Test merging of parallel legs."""
    manual_seed(0)
    A = rand(5, 9, 8)
    v = rand(5)
    B = rand(5, 9, 8)
    equation = "ijk,i,ijk->i"
    truth = einsum(equation, A, v, B)

    tn = TensorNetwork(equation, [A, v, B])
    tn.merge_parallel()
    equ, ops = tn.generate_expression()

    assert len(ops) == 3
    assert ops[0].shape == (5, 72)
    assert ops[1].shape == (5,)
    assert ops[2].shape == (5, 72)
    assert equ == "ij,i,ij->i"
    assert allclose(truth, einsum(equ, *ops).reshape_as(truth))
