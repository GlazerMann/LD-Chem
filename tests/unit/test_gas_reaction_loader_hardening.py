"""Additional gas-reaction loader regressions."""

from pathlib import Path
import pytest
from ld_chem.reactions import make_GasReactions

def test_make_gas_reactions_accepts_path_and_fortran_exponents(tmp_path: Path):
    (tmp_path / "gas_reactions.dat").write_text(
        "reactants products rate high_P_limit T_dependence form\n"
        "A,B C 1.5d3 2.0D2 -3.0d0 power\n",
        encoding="utf-8",
    )

    result = make_GasReactions(mechanism_data_path=tmp_path)

    assert result.ids == (0,)
    assert result.reactions[0].rate0 == pytest.approx(1500.0)
    assert result.reactions[0].high_P_limit == pytest.approx(200.0)
    assert result.reactions[0].T_dependence == pytest.approx(-3.0)
