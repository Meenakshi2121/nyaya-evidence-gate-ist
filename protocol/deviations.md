# Protocol Deviations

This file records changes made after the `protocol-v1.0` specification is fixed.

| Date | Original specification | Change made | Reason | Before or after viewing the affected result? |
|---|---|---|---|---|
| 2026-10-13 | The causal-forest implementation was named, but its EconML nuisance-model defaults were not written explicitly in the configuration. | Recorded the pinned EconML 0.17.0 defaults: automatic outcome and treatment nuisance models, binary outcome, two nuisance folds, subforest size four and all available processing cores. | Implementation clarification to make the declared software environment fully reproducible; the causal-forest estimand and forest hyperparameters were unchanged. | Before running the final treatment-effect or policy-performance analysis. |
| 2026-10-13 | Baseline variables and heparin adjustment were specified, but their software data types were not listed separately. | Recorded `RDELAY`, `AGE` and `RSBP` as numeric; recorded the remaining baseline variables and pooled heparin group as categorical. | Implementation clarification of the already specified covariates; no patient characteristic or estimand was changed. | Before running the final treatment-effect or policy-performance analysis. |
| 2026-10-13 | The specification required 1,000 whole-pipeline bootstrap replicates but did not state how replicate-specific random seeds would be generated or how interrupted runs would be resumed. | Derive separate resampling, fold-assignment and estimator seeds deterministically from the locked project seed and zero-indexed replicate number; save each completed replicate atomically and reject mixed analysis fingerprints. | Computational reproducibility and safe resumption of the already specified bootstrap; the resampling scheme, estimands and interval method are unchanged. | After viewing point estimates but before running bootstrap inference. |

