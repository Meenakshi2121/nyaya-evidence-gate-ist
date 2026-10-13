"""Utilities for the Nyāya-inspired IST evidence-gate analysis."""

from .adequacy import data_adequacy_check
from .crossfit import make_stratified_folds
from .estimators import EffectPredictions, fit_effect_estimators
from .evaluation import aipw_policy_risk, group_policy_gain
from .gate import build_gate
from .policies import gated_policy, individualised_policy, matched_uncertainty_acceptance
from .preprocessing import FoldPreprocessor
from .support import GowerReferenceSupport

__all__ = [
    "aipw_policy_risk",
    "build_gate",
    "data_adequacy_check",
    "EffectPredictions",
    "fit_effect_estimators",
    "FoldPreprocessor",
    "gated_policy",
    "GowerReferenceSupport",
    "group_policy_gain",
    "individualised_policy",
    "make_stratified_folds",
    "matched_uncertainty_acceptance",
]

