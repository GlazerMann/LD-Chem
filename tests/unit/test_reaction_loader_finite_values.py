"""Regression tests for non-finite reaction-rate parameters."""

from pathlib import Path

import pytest

from ld_chem.reactions import make_AqReactions, make_GasReactions


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_aqueous_loader_rejects_non_finite_rates(tmp_path: Path, value: str):
    (tmp_path / "aq_reactions.dat").write_text(
        "reactants products rate neg_Ea_R group\n"
        f"A B {value} 0 test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid numeric value in aqueous reaction"):
        make_AqReactions(chemistry="test", mechanism_data_path=tmp_path)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_gas_loader_rejects_non_finite_rates(tmp_path: Path, value: str):
    (tmp_path / "gas_reactions.dat").write_text(
        "reactants products rate high_P_limit T_dependence form\n"
        f"A B {value} 0 0 exp\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid numeric value in gas reaction"):
        make_GasReactions(mechanism_data_path=tmp_path)
