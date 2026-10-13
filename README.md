# Nyāya-Inspired Evidence Gate for Clinical Treatment Recommendations

This repository contains the analysis code and reproducibility materials for a
non-compensatory evidence gate for individualised clinical treatment
recommendations. The proof-of-concept case study uses aspirin allocation in the
International Stroke Trial (IST).

The analysis compares:

1. treating all patients with aspirin;
2. ungated individualisation;
3. Nyāya-inspired evidence-gated individualisation; and
4. uncertainty-only abstention at matched coverage.

## Current stage

The repository performs reproducible data ingestion and audit, training-fold
preprocessing, the three prespecified treatment-effect estimators, Boolean gate
construction, matched-coverage policy definition and augmented
inverse-probability-weighted policy evaluation. The primary runner connects
these components in a strictly held-out five-fold point analysis. The
whole-pipeline bootstrap is implemented as independently saved, resumable
batches; final interval estimation remains pending until all prespecified
replicates have completed.

No numerical study result is hard-coded in this repository.

## Data

Download the International Stroke Trial database (version 2) from:

https://doi.org/10.7488/ds/104

Place the CSV file at `data/raw/IST_corrected.csv`. Raw and processed patient
data are excluded from version control and must not be uploaded to GitHub.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the first calculation

```bash
python scripts/inspect_ist.py --input data/raw/IST_corrected.csv
```

This creates:

- `results/tables/column_inventory.csv`;
- `results/tables/value_inventory.csv`;
- `results/tables/missingness.csv`; and
- `results/data_audit.json`.

Review these files before editing the column mappings in `config/analysis.yml`.

## Run the primary point analysis

Prepare the analysis cohort, then run the five-fold analysis:

```bash
python scripts/prepare_ist.py
python scripts/run_primary_analysis.py
```

The second command fits all three estimators in each training fold, predicts
only its held-out fold, applies the three evidence checks and evaluates every
policy using augmented inverse-probability weighting. It writes aggregate
tables under `results/` and a Git-ignored patient-level audit file under
`data/processed/`. The aggregate outputs include the overlap between gate and
uncertainty-only selections, so matched coverage is not mistaken for selecting
the same patients.

## Run bootstrap inference

First run one complete replicate as a timing and integrity check:

```bash
python scripts/run_bootstrap.py --start 0 --stop 1
```

`--stop` is exclusive. Completed replicates are saved atomically under
`results/bootstrap/`, which is excluded from Git. Rerunning an overlapping
range safely skips completed replicates. A fingerprint prevents batches made
with different data, configuration or analysis code from being combined.

After reviewing the pilot runtime, continue in manageable batches, for example:

```bash
python scripts/run_bootstrap.py --start 1 --stop 10
```

The exact models are refitted in every replicate, so 1,000 replicates are
computationally expensive. Do not start the full range until the pilot runtime
has been reviewed and a suitable compute plan has been agreed.

Only after all 1,000 replicates are complete, create the aggregate confidence
interval outputs:

```bash
python scripts/summarise_bootstrap.py
```

The summariser refuses to run when a replicate is missing. It writes the
tracked aggregate files `results/bootstrap_summary.json` and
`results/tables/bootstrap_confidence_intervals.csv`; individual replicate
files remain excluded from version control.

## Tests

```bash
pytest
```

## Reproducibility rule

The analysis configuration is read from `config/analysis.yml`. Changes to model
definitions, thresholds or estimands must be committed before the affected
results are generated.

