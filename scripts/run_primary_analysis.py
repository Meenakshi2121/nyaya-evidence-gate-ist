"""Run the prespecified five-fold IST evidence-gate point analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_gate.adequacy import data_adequacy_check  # noqa: E402
from nyaya_gate.analysis import assemble_primary_analysis  # noqa: E402
from nyaya_gate.config import load_config  # noqa: E402
from nyaya_gate.crossfit import make_stratified_folds  # noqa: E402
from nyaya_gate.estimators import fit_effect_estimators  # noqa: E402
from nyaya_gate.support import GowerReferenceSupport  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "analysis.yml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ist_analysis.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not args.input.exists():
        raise FileNotFoundError(
            f"Prepared analysis file not found: {args.input}. "
            "Run `python scripts/prepare_ist.py` first."
        )
    data = pd.read_csv(args.input, low_memory=False)

    modelling = config["data"]["modelling_variables"]
    numeric = tuple(modelling["numeric"])
    categorical = tuple(modelling["categorical"])
    required = {
        "study_row_id",
        "aspirin",
        "outcome",
        *numeric,
        *categorical,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Prepared analysis columns are missing: {missing}")

    treatment = data["aspirin"].to_numpy(dtype=np.int8)
    outcome = data["outcome"].to_numpy(dtype=np.int8)
    participant_id = data["study_row_id"].to_numpy()
    crossfit = config["cross_fitting"]
    folds = make_stratified_folds(
        treatment,
        outcome,
        n_splits=int(crossfit["folds"]),
        random_state=int(crossfit["primary_seed"]),
        participant_id=participant_id,
    )

    n = len(data)
    reference_supported = np.empty(n, dtype=np.int8)
    reference_score = np.empty(n, dtype=float)
    causal_forest_effect = np.empty(n, dtype=float)
    causal_forest_standard_error = np.empty(n, dtype=float)
    penalised_logistic_effect = np.empty(n, dtype=float)
    t_learner_effect = np.empty(n, dtype=float)
    outcome_risk_no_aspirin = np.empty(n, dtype=float)
    outcome_risk_aspirin = np.empty(n, dtype=float)
    support_thresholds: dict[int, float] = {}

    support_config = config["gate"]["reference_support"]
    estimator_config = config["estimators"]
    unique_folds = sorted(np.unique(folds))
    for fold in unique_folds:
        held_out = folds == fold
        training = ~held_out
        print(
            f"Fold {int(fold) + 1}/{len(unique_folds)}: "
            f"training={int(training.sum())}, held_out={int(held_out.sum())}",
            flush=True,
        )

        support = GowerReferenceSupport(
            numeric_columns=numeric,
            categorical_columns=categorical,
            n_neighbors=int(support_config["neighbours_per_treatment_arm"]),
            quantile=0.95,
            n_jobs=-1,
        ).fit(data.loc[training], treatment[training])
        reference_score[held_out] = support.score(data.loc[held_out])
        reference_supported[held_out] = (
            reference_score[held_out] <= support.threshold_
        ).astype(np.int8)
        support_thresholds[int(fold)] = support.threshold_

        predictions = fit_effect_estimators(
            data.loc[training],
            data.loc[held_out],
            treatment[training],
            outcome[training],
            numeric_columns=numeric,
            categorical_columns=categorical,
            estimator_config=estimator_config,
        )
        causal_forest_effect[held_out] = predictions.causal_forest
        causal_forest_standard_error[held_out] = (
            predictions.causal_forest_standard_error
        )
        penalised_logistic_effect[held_out] = predictions.penalised_logistic
        t_learner_effect[held_out] = predictions.gradient_boosted_t_learner
        outcome_risk_no_aspirin[held_out] = predictions.outcome_risk_no_aspirin
        outcome_risk_aspirin[held_out] = predictions.outcome_risk_aspirin

    data_adequate = data_adequacy_check(
        data, config["gate"]["data_adequacy"]["essential_variables"]
    )
    analysis = assemble_primary_analysis(
        data,
        folds=folds,
        data_adequate=data_adequate,
        reference_supported=reference_supported,
        reference_score=reference_score,
        causal_forest_effect=causal_forest_effect,
        causal_forest_standard_error=causal_forest_standard_error,
        penalised_logistic_effect=penalised_logistic_effect,
        t_learner_effect=t_learner_effect,
        outcome_risk_no_aspirin=outcome_risk_no_aspirin,
        outcome_risk_aspirin=outcome_risk_aspirin,
        support_thresholds=support_thresholds,
        propensity=float(
            config["data"]["aspirin"]["nominal_randomisation_probability"]
        ),
    )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"
    table_dir = results_dir / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    analysis.patient_results.to_csv(
        processed_dir / "primary_analysis_patient_level.csv", index=False
    )
    analysis.policy_risks.to_csv(table_dir / "primary_policy_risks.csv", index=False)
    analysis.group_gains.to_csv(table_dir / "primary_group_gains.csv", index=False)
    analysis.check_patterns.to_csv(table_dir / "gate_check_patterns.csv", index=False)
    with (results_dir / "primary_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(analysis.summary, stream, indent=2)

    print(json.dumps(analysis.summary, indent=2))
    print(
        "Point analysis complete. Patient-level output remains excluded from Git; "
        "aggregate outputs are ready for review.",
        flush=True,
    )


if __name__ == "__main__":
    main()
