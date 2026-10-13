# Protocol Deviations

This file records changes made after the `protocol-v1.0` specification is fixed.

| Date | Original specification | Change made | Reason | Before or after viewing the affected result? |
|---|---|---|---|---|
| 2026-10-13 | The causal-forest implementation was named, but its EconML nuisance-model defaults were not written explicitly in the configuration. | Recorded the pinned EconML 0.17.0 defaults: automatic outcome and treatment nuisance models, binary outcome, two nuisance folds, subforest size four and all available processing cores. | Implementation clarification to make the declared software environment fully reproducible; the causal-forest estimand and forest hyperparameters were unchanged. | Before running the final treatment-effect or policy-performance analysis. |

