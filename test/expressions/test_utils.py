"""Tests ``einconv.expressions.utils``."""

from einconv.expressions.utils import get_letters, translate_to_torch


def test_get_letters():
    """Test get_letters function."""
    blocked = {"a", "o"}
    num_letters = 24
    letters = get_letters(num_letters, blocked=blocked)

    assert len(letters) == num_letters
    assert all(b not in letters for b in blocked)


def test_translate_to_torch():
    """Test translation from einops to PyTorch einsum syntax."""
    # no renaming
    assert translate_to_torch("i j,j k->i k") == "ij,jk->ik"
    # arrow has white space
    assert translate_to_torch("i j,j k -> i k") == "ij,jk->ik"
    # one rename, white space before comma
    assert translate_to_torch("row1 j , j k -> i k") == "aj,jk->ik"
    # 'idx' is a sub-string of 'idx2' -> renaming order crucial
    assert translate_to_torch("idx j , j idx2 -> idx idx2") == "aj,jb->ab"
