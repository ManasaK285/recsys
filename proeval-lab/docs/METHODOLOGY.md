# Methodology and limitations

## Problem setup
Each case `x_i` has a costly outcome `y_i` representing error severity in [0, 1]. The finite-benchmark ground truth is `mean(y_i)`. Failure cases satisfy `y_i >= threshold`.

## Synthetic benchmark
Cases have six numeric features representing difficulty, ambiguity, compositionality, instruction complexity, safety sensitivity, and distribution shift. A nonlinear latent function plus small noise creates severity scores. Category-specific terms produce several failure modes.

## GP surrogate
A GaussianProcessRegressor with an RBF kernel plus WhiteKernel is fitted to observed cases. It returns predictive mean and standard deviation for unobserved candidates. Training is capped for CPU practicality.

## Acquisition strategies
- `random`: choose a random unobserved case.
- `uncertainty`: choose cases with largest GP standard deviation.
- `risk`: choose cases with largest predicted severity.
- `hybrid`: rank a blend of predicted severity, uncertainty, and distance from the failure threshold.

## Score estimation
For a finite benchmark, the simplest unbiased reference is uniform random sampling and its sample mean. The prototype's `gp_weighted` estimate uses GP predictions for unobserved cases and replaces predictions with observed labels. This is a plug-in estimator, not a claim of unbiased Bayesian quadrature. Compare it with random-sample estimates and report absolute error against the full-benchmark mean.

A production-quality Bayesian quadrature implementation should model the input distribution and kernel mean, and report posterior integration uncertainty. Do not call the plug-in estimate a mathematically exact BQ estimator.

## Transfer baseline
The optional historical prior is intentionally simplified: a source-task GP is trained on a separate synthetic task, and its predictions provide a prior-mean-like feature to a target model. This demonstrates the experiment design for transfer, not the paper's exact pre-trained GP algorithm.

## Metrics
- `estimate_mae`: absolute error between estimated and full-benchmark mean severity.
- `failure_precision` / `failure_recall`: discovered failures among queried examples, using the benchmark labels available in this offline simulation.
- `category_coverage`: number of failure categories represented among discovered failures.
- `evaluations_to_tolerance`: first budget checkpoint where estimation MAE <= target tolerance, or null if not reached.

## Reproducibility
Use fixed seeds, the same benchmark per seed for all strategies, and multiple seeds. The complete synthetic labels are used only to simulate an offline oracle. In real deployment, the labels for unqueried cases would not be available.
