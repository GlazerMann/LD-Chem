#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:58:24 2024

@author: beel083
"""

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import numpy as np
import sys
import ld_chem.constants as c
from .processes.air_thermo import H2O_gas_conc, es

@dataclass(frozen=True)
class AqReaction:
    """Reaction: the definition of an aqueous
    phase reaction. Aqueous reaction rates have units
    of M^(1-n)/s."""
    reactants: list          # name of the species
    products: list
    rate0: float
    neg_Ea_R: float
    
    def get_rate(self, T):
        # returns rate at given temperature
        return self.rate0*np.exp(self.neg_Ea_R*((1/T)-(1/298)))  # (mol/m^3^(1-n)/s)

@dataclass(frozen=True)
class GasReaction:
    """Reaction: the definition of a gas
    phase reaction. Gas phase reaction rates have units
   of (molec/cm^3)^(1-n)/s."""
    reactants: list          # name of the species
    products: list
    rate0: float
    high_P_limit: float
    T_dependence: float
    form: str
    
    def get_rate(self, S, T, P):
        # returns rate at a given temperature
        if self.form == 'power':
            return self.rate0*(T/300)**self.T_dependence
        elif self.form == 'exp':
            return self.rate0*np.exp(self.T_dependence/T)
        elif self.form == 'troe':
            X_H2O = (S*es(T-273.15))/P
            k0_N2 = self.rate0*(T/300)**self.T_dependence
            k0_H2O = 1.65e-32*3.63e35*(T/300)**(-4.9)
            k0_mix = (1-X_H2O)*k0_N2+X_H2O*k0_H2O
            k_inf = self.high_P_limit
            M = P/(c.R*T)
            Pr = (k0_mix*M)/k_inf
            logFc = np.log10(0.58)
            N = 0.75 - 1.27 * logFc
            logPr = np.log10(Pr)
            denom = 1.0 + (logPr / N)**2
            F = 10.0**(logFc / denom)
            return (F*k0_mix*M)/(1+((k0_mix*M)/k_inf))
        elif self.form == 'HO2_water_enhancement':
            H2O_conc = H2O_gas_conc(S,T,P)
            N2_conc = 0.7808*((P/(c.R*T))-H2O_conc)
            k1 = 1.32e5*np.exp(600/T)
            k2 = 6.9e2*N2_conc*np.exp(980/T)
            return (k1+k2)*(1.0+8.4e-4*H2O_conc*np.exp(2200/T))
            
@dataclass
class AqueousReactions:
    """EquilibriumReactions: the definition of which aqueous reactions
    are accounted for in the model"""
    reactions: Tuple[AqReaction, ...]
    ids: Tuple[int, ...]
    
@dataclass
class GasReactions:
    reactions: Tuple[GasReaction, ...]
    ids: Tuple[int, ...]
    
def make_AqReactions(chemistry=None, mechanism_data_path="mechanisms/"):
    reaction_datafile = Path(mechanism_data_path) / "aq_reactions.dat"
    selected_groups = None if chemistry is None else set(chemistry)

    def _fortran_float(value: str) -> float:
        return float(value.replace("D", "E").replace("d", "e"))

    reactions = []
    ids = []

    with reaction_datafile.open(encoding="utf-8") as data_file:
        # aq_reactions.dat contains a one-line header.
        next(data_file, None)

        for line_number, line in enumerate(data_file, start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"Malformed aqueous reaction in {reaction_datafile} at "
                    f"line {line_number}: expected 5 fields, found "
                    f"{len(fields)}."
                )

            reactants, products, rate, neg_Ea_R, group = fields
            if selected_groups is not None and group not in selected_groups:
                continue

            try:
                one_reaction = AqReaction(
                    reactants=reactants.split(","),
                    products=products.split(","),
                    rate0=_fortran_float(rate),
                    neg_Ea_R=_fortran_float(neg_Ea_R),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in aqueous reaction at "
                    f"{reaction_datafile}:{line_number}."
                ) from exc

            ids.append(len(reactions))
            reactions.append(one_reaction)

    return AqueousReactions(
        reactions=tuple(reactions),
        ids=tuple(ids),
    )

def make_GasReactions(chemistry=None, mechanism_data_path="mechanisms/"):
    if chemistry is not None:
        raise NotImplementedError(
            "make_GasReactions() received chemistry=%r, but group-based "
            "filtering of gas-phase reactions is not implemented: "
            "gas_reactions.dat has no 'group' column (unlike aq_reactions.dat, "
            "which make_AqReactions() filters on), so there is currently no way "
            "to select a subset of gas-phase reactions."
            % (chemistry,)
        )

    reaction_datafile = mechanism_data_path + "gas_reactions.dat"
    valid_forms = {
        "power",
        "exp",
        "troe",
        "HO2_water_enhancement",
    }

    def _fortran_float(value: str) -> float:
        return float(value.replace("D", "E").replace("d", "e"))

    reactions = []
    ids = []

    with reaction_datafile.open() as data_file:
        # gas_reactions.dat contains a one-line header.
        next(data_file, None)

        for line_number, line in enumerate(data_file, start=2):
            fields = line.split()

            if not fields:
                continue

            if len(fields) != 6:
                raise ValueError(
                    f"Malformed gas reaction in {reaction_datafile} at "
                    f"line {line_number}: expected 6 fields, found "
                    f"{len(fields)}."
                )

            (
                reactants,
                products,
                rate,
                highP_limit,
                T_dependence,
                form,
            ) = fields

            if form not in valid_forms:
                raise ValueError(
                    f"Unsupported gas reaction form '{form}' in "
                    f"{reaction_datafile} at line {line_number}."
                )

            try:
                one_reaction = GasReaction(
                    reactants=reactants.split(","),
                    products=products.split(","),
                    rate0=_fortran_float(rate),
                    high_P_limit=_fortran_float(highP_limit),
                    T_dependence=_fortran_float(T_dependence),
                    form=form,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in gas reaction at "
                    f"{reaction_datafile}:{line_number}."
                ) from exc

            ids.append(len(reactions))
            reactions.append(one_reaction)
    
    return GasReactions(
        reactions=tuple(reactions),
        ids=tuple(ids),
    )
