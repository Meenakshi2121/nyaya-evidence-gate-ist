# Data instructions

1. Download IST version 2 from https://doi.org/10.7488/ds/104.
2. Save the original CSV as `data/raw/IST_corrected.csv`.
3. Do not commit the CSV or any patient-level derivative to GitHub.
4. Run `python scripts/inspect_ist.py --input data/raw/IST_corrected.csv`.
5. Confirm the mappings and value codings in `config/analysis.yml` against the
   accompanying IST documentation.

Only aggregate, disclosure-safe tables and figures should be committed.

