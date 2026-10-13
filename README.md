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
construction, matched-coverage policy definition and tested augmented
inverse-probability-weighted policy evaluation. The next implementation stage
connects these components in the primary five-fold analysis runner.

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

## Tests

```bash
pytest
```

## Reproducibility rule

The analysis configuration is read from `config/analysis.yml`. Changes to model
definitions, thresholds or estimands must be committed before the affected
results are generated.

