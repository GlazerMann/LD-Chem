"""Water feedback checks must retain changes near floating-point resolution."""
from decimal import Decimal, localcontext
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from ld_chem.utilities import check_water_condensation


def _state(water, number):
    return SimpleNamespace(particles=SimpleNamespace(
        spec_masses=np.array([[water]]), num_concs=np.array([number]),
        get_species_idx=lambda name: 0,
    ))


def _exact_mass_change(before, after, n_before, n_after):
    # Independent high-precision reference for the represented float inputs.
    with localcontext() as context:
        context.prec = 100
        return float(Decimal.from_float(after) * Decimal.from_float(n_after)
                     - Decimal.from_float(before) * Decimal.from_float(n_before))


def test_tiny_condensation_change_passes_mass_check():
    before = 1.0e-25
    after = float(np.nextafter(before, 0.0))
    change = _exact_mass_change(before, after, 1.0e6, 1.0e6)
    assert change != 0.0
    check_water_condensation(_state(before, 1.0e6), _state(after, 1.0e6), change)


def test_incorrect_tiny_condensation_change_is_still_rejected():
    before = 1.0e-25
    after = float(np.nextafter(before, 0.0))
    change = _exact_mass_change(before, after, 1.0e6, 1.0e6)
    with pytest.raises(ValueError, match='does not match'):
        check_water_condensation(_state(before, 1.0e6), _state(after, 1.0e6), 2.0 * change)


def test_number_concentration_change_is_included_in_water_budget():
    before, after = _state(1.0e-25, 1.0e6), _state(1.0e-25, 2.0e6)
    change = _exact_mass_change(1.0e-25, 1.0e-25, 1.0e6, 2.0e6)
    check_water_condensation(before, after, change)
    with pytest.raises(ValueError, match='does not match'):
        check_water_condensation(before, after, 0.0)


def test_zero_water_change_does_not_emit_a_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        check_water_condensation(_state(1.0e-25, 1.0e6), _state(1.0e-25, 1.0e6), 0.0)
