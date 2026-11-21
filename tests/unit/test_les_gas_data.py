from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from ld_chem.processes.air_thermo import H2O_mole_fraction
from ld_chem.systems import air_from_les


class FakeState:
    def __init__(self, **values):
        self.__dict__.update(values)

    def clone_detached(self):
        return deepcopy(self)


def test_instantaneous_les_mixing_without_trace_gas_data_uses_background():
    state = FakeState(
        x=0.0, y=0.0, z=0.0, P=101325.0, T=298.0, S=0.5,
        gas=SimpleNamespace(
            gases=[SimpleNamespace(name=name) for name in ("N2", "O2", "HO2")],
            concs=np.array([1.0, 2.0, 3.0]),
        ),
    )
    driver = SimpleNamespace(
        t_data=np.array([0.0, 1.0]),
        x_data=np.zeros(2), y_data=np.zeros(2), z_data=np.ones(2),
        P_data=np.full(2, 101325.0), T_data=np.full(2, 298.0),
        S_data=np.full(2, 0.5), TraceGas_data=None,
    )

    result = air_from_les(state, processes=None, t2=1.0,
                          relaxation_time=0.0, dt=1.0, driver=driver)

    water_fraction = H2O_mole_fraction(0.5, 298.0, 101325.0)
    np.testing.assert_allclose(
        result.gas.concs,
        [1e9 * 0.7808 * (1 - water_fraction),
         1e9 * 0.2095 * (1 - water_fraction), 0.0],
    )


def test_instantaneous_les_mixing_fills_missing_entries_in_partial_mapping():
    state = FakeState(
        x=0.0, y=0.0, z=0.0, P=101325.0, T=298.0, S=0.5,
        gas=SimpleNamespace(
            gases=[SimpleNamespace(name=name) for name in ("N2", "O2", "HO2")],
            concs=np.array([1.0, 2.0, 3.0]),
        ),
    )
    driver = SimpleNamespace(
        t_data=np.array([0.0, 1.0]),
        x_data=np.zeros(2), y_data=np.zeros(2), z_data=np.ones(2),
        P_data=np.full(2, 101325.0), T_data=np.full(2, 298.0),
        S_data=np.full(2, 0.5),
        TraceGas_data={"HO2": np.array([7.0, 11.0])},
    )

    result = air_from_les(state, processes=None, t2=1.0,
                          relaxation_time=0.0, dt=1.0, driver=driver)

    water_fraction = H2O_mole_fraction(0.5, 298.0, 101325.0)
    np.testing.assert_allclose(
        result.gas.concs,
        [1e9 * 0.7808 * (1 - water_fraction),
         1e9 * 0.2095 * (1 - water_fraction), 11.0],
    )
