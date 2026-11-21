"""Trace solute changes must not disappear beside unchanged particle mass."""
import numpy as np
import pytest

from ld_chem.systems import account_for_implicit_aqueous_water


def test_trace_solute_increase_cannot_bypass_available_water_check():
    before = np.array([[1.0e-15, 0.0, 1.0e-35]])
    after = np.array([[1.0e-15, 1.0e-33, 1.0e-35]])
    with pytest.raises(ValueError, match='more water than is available'):
        account_for_implicit_aqueous_water(before, after, water_idx=2)


def test_trace_solute_change_is_independent_of_unchanged_inert_mass():
    # Enough water to represent the correction, but a much larger unchanged
    # component hides it in a difference of whole-row totals.
    before = np.array([[1.0e-15, 0.0, 1.0e-25]])
    after = np.array([[1.0e-15, 1.0e-30, 1.0e-25]])
    balanced = account_for_implicit_aqueous_water(before, after, water_idx=2)
    assert balanced[0, 2] == pytest.approx(1.0e-25 - 1.0e-30, rel=1.0e-12, abs=0.0)
    np.testing.assert_array_equal(balanced[:, :2], after[:, :2])
    np.testing.assert_array_equal(before, [[1.0e-15, 0.0, 1.0e-25]])
    np.testing.assert_array_equal(after, [[1.0e-15, 1.0e-30, 1.0e-25]])


def test_trace_solute_loss_returns_water_for_each_particle():
    before = np.array([[1.0e-15, 1.0e-33, 0.0], [1.0e-15, 2.0e-33, 0.0]])
    after = np.array([[1.0e-15, 0.0, 0.0], [1.0e-15, 0.0, 0.0]])
    balanced = account_for_implicit_aqueous_water(before, after, water_idx=2)
    np.testing.assert_allclose(balanced[:, 2], [1.0e-33, 2.0e-33], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("reference, updated", [
    (np.zeros((1, 3)), np.zeros((2, 3))),
    (np.zeros((2, 3)), np.zeros((1, 3))),
    (np.zeros((2, 1)), np.zeros((2, 3))),
    (np.zeros(3), np.zeros(3)),
])
def test_water_balance_rejects_broadcasting_or_nonmatrix_inputs(reference, updated):
    with pytest.raises(ValueError, match="shape|two-dimensional"):
        account_for_implicit_aqueous_water(reference, updated, water_idx=0)


@pytest.mark.parametrize("which", ["reference", "updated"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_water_balance_rejects_nonfinite_mass_inputs(which, bad):
    reference = np.array([[0.0, 1.0]])
    updated = reference.copy()
    (reference if which == "reference" else updated)[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        account_for_implicit_aqueous_water(reference, updated, water_idx=1)


def test_unsigned_mass_decrease_does_not_wrap():
    reference = np.array([[2, 10]], dtype=np.uint64)
    updated = np.array([[1, 10]], dtype=np.uint64)
    result = account_for_implicit_aqueous_water(reference, updated, water_idx=1)
    np.testing.assert_array_equal(result, [[1.0, 11.0]])
    np.testing.assert_array_equal(reference, [[2, 10]])
    np.testing.assert_array_equal(updated, [[1, 10]])


def test_finite_inputs_cannot_return_infinite_water_after_overflow():
    reference = np.array([[1e308, 1e308, 0.0]])
    updated = np.zeros((1, 3))
    with pytest.raises(ValueError, match="finite"):
        account_for_implicit_aqueous_water(reference, updated, water_idx=2)


@pytest.mark.parametrize("water_idx", [True, 0.5, 2, -3])
def test_water_index_must_select_a_real_column(water_idx):
    masses = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="water_idx"):
        account_for_implicit_aqueous_water(masses, masses, water_idx)


def test_empty_particle_axis_and_negative_column_index_remain_supported():
    empty = np.empty((0, 2))
    assert account_for_implicit_aqueous_water(empty, empty, -1).shape == (0, 2)
    masses = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(account_for_implicit_aqueous_water(masses, masses, -1), masses)


def test_unsigned_water_cannot_wrap_past_insufficient_water_guard():
    reference = np.array([[1, 0]], dtype=np.uint64)
    updated = np.array([[2, 0]], dtype=np.uint64)
    with pytest.raises(ValueError, match='more water than is available'):
        account_for_implicit_aqueous_water(reference, updated, water_idx=1)
