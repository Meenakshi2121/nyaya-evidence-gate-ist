# Technical analysis amendment: nuisance-model convergence

**Date:** 3 November 2026
**Stage:** Before production whole-pipeline bootstrap inference

## Trigger

The initial IST point analysis and one bootstrap timing replicate produced a
scikit-learn `ConvergenceWarning`. A subsequent diagnostic run promoted the
warning to an error and traced it to the linear nuisance-model candidate
created by EconML's automatic first-stage model selector.

## Amendment

EconML's original forest-versus-linear candidate set is retained. The linear
candidate remains `LogisticRegressionCV`, but it is now instantiated
explicitly with a maximum of 5,000 iterations instead of the library default
of 100. The prespecified random state and model-selection procedure are
retained.

No outcome definition, analysis population, covariate, treatment definition,
gate rule, comparator, decision threshold, bootstrap stratification rule,
replicate count, seed schedule or scientific endpoint was changed.

## Handling of earlier outputs

The earlier point estimates and bootstrap replicate 0 are treated as
diagnostic outputs only. They will not contribute to final inference. The
point analysis and all 1,000 bootstrap replicates will be recomputed from a
clean result set after verification of this amendment.

## Verification plan

1. Run the existing automated test suite.
2. Run one complete bootstrap replicate with convergence warnings treated as
   errors.
3. Freeze and commit the amended code and configuration.
4. Restart the complete 1,000-replicate bootstrap from an empty production
   result directory.

## Verification after amendment

Verification was completed on 3 November 2026.

- All 22 tests passed.
- Bootstrap replicate 0 completed in 591 seconds without a convergence failure.
- The corrected primary analysis completed in 616 seconds.
- A repeated execution completed in 619 seconds and produced byte-for-byte
  identical aggregate outputs.
- The corrected evidence-gated risk difference versus treating all was
  0.0013354764325769208.
- The matched uncertainty-only risk difference was
  -0.000007816387638026612.
- The accepted-minus-deferred principal contrast was
  -0.00024183517477516236.
- The gate accepted 8,799 of 19,285 participants (45.63%).
- Statistical inference remains pending the prespecified 1,000-replicate
  whole-pipeline bootstrap.
