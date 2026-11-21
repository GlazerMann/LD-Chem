#!/usr/bin/env python3
"""Run a small self-contained LD-Chem LES-style trajectory case."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def _numpy():
    import numpy as np

    return np


def build_aerosol_population() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
    """Build a simple two-mode aerosol population for the trajectory case."""
    np = _numpy()
    from part2pop.population import build_population

    pop_cfg = {
        "type": "binned_lognormals",
        "N": [1.0e9, 1.0e9],
        "GMD": [150e-9, 150e-9],
        "GSD": [1.6, 1.6],
        "aero_spec_names": [["SO4"], ["OC"]],
        "aero_spec_fracs": [[1.0], [1.0]],
        "N_bins": 10,
        "N_sigmas": 5,
        "species_modifications": {"OC": {"density": 1200}},
    }

    aerosol_population = build_population(pop_cfg)
    aero_spec_names = np.array([species.name for species in aerosol_population.species])
    aero_spec_masses = np.array(aerosol_population.spec_masses)
    num_concs = np.array(aerosol_population.num_concs)
    return (
        aero_spec_names,
        aero_spec_masses,
        num_concs,
        tuple(aerosol_population.species),
    )


def build_trajectory() -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Build a synthetic LES-style trajectory for a short tutorial run."""
    np = _numpy()
    t = np.arange(0.0, 1800.0 + 5.0, 5.0)
    z = 500.0 + 300.0 * np.sin(np.pi * t / t[-1])
    T = 293.0 - 0.0065 * z
    P = 101325.0 * np.exp(-z / 8500.0)
    s = 0.98 + 0.04 * np.sin(np.pi * t / t[-1])
    x = np.full_like(t, -97.5)
    y = np.full_like(t, 36.6)
    gas = {
        "IEPOX": np.full_like(t, 0.4),
        "SO2": np.full_like(t, 1.0),
    }

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "T": T,
        "P": P,
        "s": s,
        "gas": gas,
    }


def _numeric_array(value: Any) -> np.ndarray | None:
    np = _numpy()
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None

    if array.size == 0 or not np.isfinite(array).any():
        return None
    return np.squeeze(array)


def _find_series(data: Any, names: tuple[str, ...]) -> np.ndarray | None:
    if isinstance(data, dict):
        lowered = {str(key).lower(): key for key in data}
        for name in names:
            key = lowered.get(name.lower())
            if key is not None:
                array = _numeric_array(data[key])
                if array is not None:
                    return array
        for value in data.values():
            array = _find_series(value, names)
            if array is not None:
                return array

    for name in names:
        if hasattr(data, name):
            array = _numeric_array(getattr(data, name))
            if array is not None:
                return array
    return None

def _build_particle_gas_trajectory(data, names):
    if names is not None:
        out_data = {}
        for ii, (name) in enumerate(names):
            out_data[name]=data[:,:,ii]
        return out_data
    return None

def _reduce_to_axis(values: np.ndarray, axis_length: int) -> np.ndarray:
    np = _numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    if values.shape[0] == axis_length:
        return np.nanmedian(values, axis=tuple(range(1, values.ndim)))
    if values.shape[-1] == axis_length:
        return np.nanmedian(values, axis=tuple(range(values.ndim - 1)))
    return np.nanmedian(values.reshape(values.shape[0], -1), axis=1)


def run_case(output_dir: Path) -> Path:
    """Run the synthetic LES-style trajectory case."""
    np = _numpy()
    from ld_chem.run import simulate_les_trajectory

    output_dir.mkdir(parents=True, exist_ok=True)

    (
        aero_spec_names,
        aero_spec_masses,
        num_concs,
        aero_species,
    ) = build_aerosol_population()
    pHs = np.full(num_concs.shape[0], 4.5)
    trajectory_data = build_trajectory()

    output_filename = output_dir / "trajectory.pkl"
    restart_filename = output_dir / "trajectory_restart.pkl"

    simulate_les_trajectory(
        aero_spec_names,
        aero_spec_masses,
        num_concs,
        pHs,
        trajectory_data,
        dt=1.0,
        restart_filename=str(restart_filename),
        radius_scale="log",
        output_filename=str(output_filename),
        aq_chemistry=None,
        write_every=10.0,
        condensation=True,
        gas_chemistry=False,
        print_to_screen=True,
        cocondensation=False,
        relaxation_time=25.0,
        aero_species=aero_species,
    )

    return output_filename


def plot_outputs(output_dir: Path) -> None:
    """Plot saturation ratio and median wet diameter against time."""
    np = _numpy()
    import matplotlib.pyplot as plt

    output_filename = output_dir / "trajectory.pkl"
    with output_filename.open("rb") as handle:
        data = pickle.load(handle)
    data["particles"]=_build_particle_gas_trajectory(data["particles"], data["particle species"])
    data["gases"]=_build_particle_gas_trajectory(data["gases"], data["gas species"])

    time = _find_series(data, ("times", "time", "t"))
    saturation = _find_series(data, ("s", "S", "saturation ratio"))
    wet_diameter = _find_series(data["particles"], ("Dwet", "wet_diameter"))

    if time is None or saturation is None or wet_diameter is None:
        raise KeyError(
            "Could not find time, saturation ratio, and wet diameter "
            f"series in {output_filename}."
        )

    time = np.ravel(time)
    saturation = _reduce_to_axis(saturation, time.size)
    wet_diameter = _reduce_to_axis(wet_diameter, time.size)

    n_points = min(time.size, saturation.size, wet_diameter.size)
    time_minutes = time[:n_points] / 60.0
    saturation = saturation[:n_points]
    wet_diameter = wet_diameter[:n_points] * 1e6

    _, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(time_minutes, saturation)
    axes[0].set_xlabel("Time [min]")
    axes[0].set_ylabel("Saturation ratio [-]")

    axes[1].plot(time_minutes, wet_diameter)
    axes[1].set_xlabel("Time [min]")
    axes[1].set_ylabel("Median wet diameter [um]")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated LD-Chem output files.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show simple diagnostic plots after the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_case(args.output_dir)
    print(f"Wrote {output_path}")

    if args.plot:
        plot_outputs(args.output_dir)


if __name__ == "__main__":
    main()
