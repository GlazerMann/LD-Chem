import pytest
import numpy as np
from pathlib import Path
from ld_chem.reactions import (
    AqReaction, GasReaction, AqueousReactions, GasReactions,
    make_AqReactions, make_GasReactions
)
from ld_chem.processes.air_thermo import H2O_gas_conc, es


def test_aq_reaction_creation():
    """Test AqReaction dataclass creation."""
    reaction = AqReaction(
        reactants=["SO2"],
        products=["HSO3", "H+"],
        rate0=5.0e5,
        neg_Ea_R=1960.0
    )
    assert reaction.reactants == ["SO2"]
    assert reaction.products == ["HSO3", "H+"]
    assert reaction.rate0 == 5.0e5
    assert reaction.neg_Ea_R == 1960.0


def test_aq_reaction_get_rate():
    """Test aqueous reaction rate calculation."""
    reaction = AqReaction(
        reactants=["SO2"],
        products=["HSO3", "H+"],
        rate0=5.0e5,
        neg_Ea_R=1960.0
    )
    T = 298  # 25°C
    rate = reaction.get_rate(T)

    # At reference temperature, should equal rate0
    assert np.isclose(rate, reaction.rate0)

    # Should change with temperature
    T_higher = 308  # 35°C
    rate_higher = reaction.get_rate(T_higher)
    assert rate_higher != rate


def test_gas_reaction_creation():
    """Test GasReaction dataclass creation."""
    reaction = GasReaction(
        reactants=["H2O2"],
        products=["OH", "OH"],
        rate0=7.66e-6,
        high_P_limit=0.0,
        T_dependence=0.0,
        form="exp"
    )
    assert reaction.reactants == ["H2O2"]
    assert reaction.products == ["OH", "OH"]
    assert reaction.rate0 == 7.66e-6
    assert reaction.high_P_limit == 0.0
    assert reaction.T_dependence == 0.0
    assert reaction.form == "exp"


def test_gas_reaction_get_rate_power():
    """Test gas reaction rate calculation with power law."""
    reaction = GasReaction(
        reactants=["A"],
        products=["B"],
        rate0=1e-10,
        high_P_limit=0.0,
        T_dependence=2.0,
        form="power"
    )
    S = 1.0
    T = 300.0
    P = 101325.0
    rate = reaction.get_rate(S, T, P)
    expected = 1e-10 * (T/300)**2.0
    assert np.isclose(rate, expected)


def test_gas_reaction_get_rate_exp():
    """Test gas reaction rate calculation with exponential."""
    reaction = GasReaction(
        reactants=["A","B"],
        products=["C"],
        rate0=7.66e-6,
        high_P_limit=0.0,
        T_dependence=0.0,
        form="exp"
    )
    S = 1.0
    T = 298.15
    P = 101325.0
    rate = reaction.get_rate(S, T, P)
    expected = 7.66e-6 * np.exp(0.0/T)
    assert np.isclose(rate, expected)

def test_gas_reaction_get_rate_troe():
    """Test gas reaction rate calculation with troe form."""
    reaction = GasReaction(
        reactants=["A","B"],
        products=["C"],
        rate0=7.66e-6,
        high_P_limit=1.2e6,
        T_dependence=-4.0,
        form="troe"
    )
    S = 1.0
    T = 298.15
    P = 101325.0
    rate = reaction.get_rate(S, T, P)

    X_H2O = (S*es(T-273.15))/P
    k0_N2 = reaction.rate0*(T/300)**reaction.T_dependence
    k0_H2O = 1.65e-32*3.63e35*(T/300)**(-4.9)
    k0_mix = (1-X_H2O)*k0_N2+X_H2O*k0_H2O
    k_inf = reaction.high_P_limit
    M = P/(8.314*T)
    Pr = (k0_mix*M)/k_inf
    logFc = np.log10(0.58)
    N = 0.75 - 1.27 * logFc
    logPr = np.log10(Pr)
    denom = 1.0 + (logPr / N)**2
    F = 10.0**(logFc / denom)
    expected = (F*k0_mix*M)/(1+((k0_mix*M)/k_inf))
    
    assert np.isclose(rate, expected)

def test_gas_reaction_get_rate_H2O_enhancement():
    """Test gas reaction rate calculation with H2O enhancement form."""
    reaction = GasReaction(
        reactants=["A","B"],
        products=["C"],
        rate0=0.0,
        high_P_limit=0.0,
        T_dependence=0.0,
        form="HO2_water_enhancement"
    )
    S = 1.0
    T = 298.15
    P = 101325.0
    rate = reaction.get_rate(S, T, P)

    H2O_conc = H2O_gas_conc(S,T,P)
    N2_conc = 0.7808*((P/(8.314*T))-H2O_conc)
    k1 = 1.32e5*np.exp(600/T)
    k2 = 6.9e2*N2_conc*np.exp(980/T)
    expected = (k1+k2)*(1.0+8.4e-4*H2O_conc*np.exp(2200/T))
    
    assert np.isclose(rate, expected)

def test_aqueous_reactions_creation():
    """Test AqueousReactions dataclass creation."""
    reaction1 = AqReaction(["A", "B"], ["C"], 5.0e5, 1960.0)
    reaction2 = AqReaction(["D"], ["E", "F"], 5.0e5, 1500.0)
    reactions = (reaction1, reaction2)
    ids = (0, 1)
    aq_reactions = AqueousReactions(reactions=reactions, ids=ids)
    assert len(aq_reactions.reactions) == 2
    assert len(aq_reactions.ids) == 2
    assert aq_reactions.reactions[0].reactants == ["A", "B"]


def test_gas_reactions_creation():
    """Test GasReactions dataclass creation."""
    reaction1 = GasReaction(["A", "B"], ["C"], 7.66e-6, 0.0, 0.0, "exp")
    reaction2 = GasReaction(["D"], ["E", "F"], 3.55e-5, 0.0, 0.0, "exp")
    reactions = (reaction1, reaction2)
    ids = (0, 1)
    gas_reactions = GasReactions(reactions=reactions, ids=ids)
    assert len(gas_reactions.reactions) == 2
    assert len(gas_reactions.ids) == 2
    assert gas_reactions.reactions[0].reactants == ["A", "B"]


def test_make_aq_reactions():
    """Test creation of AqueousReactions from file."""
    # Path to mechanisms directory from test file location
    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"

    chemistry = ["sulfate"]
    aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=str(mechanisms_path) + "/")

    assert aq_reactions.reactions is not None
    assert aq_reactions.ids is not None
    assert len(aq_reactions.reactions) > 0
    assert len(aq_reactions.ids) > 0

    # Check that reactions have the expected structure
    for reaction in aq_reactions.reactions:
        assert isinstance(reaction, AqReaction)
        assert isinstance(reaction.reactants, list)
        assert isinstance(reaction.products, list)
        assert isinstance(reaction.rate0, float)
        assert isinstance(reaction.neg_Ea_R, float)

def test_make_aq_reactions_empty():
    """Test creation of empty AqueousReactions."""
    # Path to mechanisms directory from test file location
    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    chemistry = []
    aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=str(mechanisms_path) + "/")
    assert aq_reactions.reactions is None
    assert aq_reactions.ids is None

def test_make_gas_reactions():
    """Test creation of GasReactions from file."""
    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"

    # make_GasReactions doesn't use chemistry parameter like make_AqReactions
    gas_reactions = make_GasReactions(mechanism_data_path=str(mechanisms_path) + "/")

    assert gas_reactions.reactions is not None
    assert gas_reactions.ids is not None
    assert len(gas_reactions.reactions) > 0
    assert len(gas_reactions.ids) > 0

    # Check that reactions have the expected structure
    for reaction in gas_reactions.reactions:
        assert isinstance(reaction, GasReaction)
        assert isinstance(reaction.reactants, list)
        assert isinstance(reaction.products, list)
        assert isinstance(reaction.rate0, float)
        assert isinstance(reaction.high_P_limit, float)
        assert isinstance(reaction.T_dependence, float)
        assert reaction.form in ["power", "exp", "troe", "HO2_water_enhancement"]

def test_make_gas_reactions_rejects_chemistry_argument(tmp_path):
    with pytest.raises(NotImplementedError, match="group-based filtering"):
        make_GasReactions(
            chemistry=["sulfate"],
            mechanism_data_path=str(tmp_path) + "/",
        )


def test_make_gas_reactions_rejects_unknown_form(tmp_path):
    (tmp_path / "gas_reactions.dat").write_text(
        "reactants products rate high_P_limit T_dependence form\n"
        "A B 1.0 0.0 0.0 unsupported\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported gas reaction form"):
        make_GasReactions(mechanism_data_path=str(tmp_path) + "/")


def test_make_gas_reactions_rejects_malformed_rows(tmp_path):
    (tmp_path / "gas_reactions.dat").write_text(
        "reactants products rate high_P_limit T_dependence form\n"
        "A B 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 6 fields"):
        make_GasReactions(mechanism_data_path=str(tmp_path) + "/")


def test_make_gas_reactions_header_only_returns_empty(tmp_path):
    (tmp_path / "gas_reactions.dat").write_text(
        "reactants products rate high_P_limit T_dependence form\n",
        encoding="utf-8",
    )

    gas_reactions = make_GasReactions(
        mechanism_data_path=str(tmp_path) + "/")

    assert gas_reactions.reactions == ()
    assert gas_reactions.ids == ()

def test_make_gas_reactions_parses_valid_file(tmp_path):
    (tmp_path / "gas_reactions.dat").write_text(
        "reactants products rate high_P_limit T_dependence form\n"
        "A,B C 1.0 2.0 3.0 power\n",
        encoding="utf-8",
    )

    result = make_GasReactions(
        mechanism_data_path=str(tmp_path) + "/"
    )

    assert len(result.reactions) == 1
    assert result.ids == (0,)
    assert result.reactions[0].reactants == ["A", "B"]
    assert result.reactions[0].products == ["C"]
    assert result.reactions[0].rate0 == pytest.approx(1.0)
