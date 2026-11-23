# Figure captions (structure: claim | panels | statistics | n & unit)

The 'Observed' line is generated from the result tables. Write the final claim and interpretation yourself after reading the figure.

## fig0_data_quality

**Fig. 0 |** Special codes are common in some numeric fields; amounts and incomes are right-skewed.
- Panels: a, share of LAR records whose value is NA, Exempt, empty or unparsable by field. b, loan amount distribution. c, income distribution.
- Statistics: Counts only; log10 bins of width 0.1. Values outside the positive analysis domain are flagged, not called data errors or removed.
- n and unit: n = 12,259,185 LAR records with valid loan amount; unit = LAR record.
- Observed: Field with the largest non-numeric share: total_loan_costs (55.0%).

## fig1_q1_products_states

**Fig. 1 |** Denial rates differ by loan product and are broadly consistent across states.
- Panels: a, denial rate and origination share for the 12 largest product cells (different denominators). b, state x loan type denial rate (cell label = percent denied). c, Spearman correlation between each state's product-level rates and the national pattern (dashed = threshold). d, state observed/expected denial extremes after product-mix standardisation. State origination-share results are reported in the accompanying table rather than adding another crowded panel.
- Statistics: Wilson 95% intervals (a); Spearman on product cells meeting the configured size threshold (c), interpreted as agreement in product ranking, not equality of state rates; Garwood 95% intervals for O/E (d). Differences are associations, not causal effects.
- n and unit: n = 8,661,779 decided applications for denial and 10,781,992 applications for origination share; unit = application record; state NA (location not provided) excluded from b-d.

## fig2_q2_lenders

**Fig. 2 |** Lender denial rates are widely dispersed, and much of the spread persists within common product segments.
- Panels: a, distribution of lender denial rates (eligible lenders; dashed = national rate). b, observed/expected denials versus lender size (red/blue = interval excludes 1). c, observed/expected denials versus raw rate (dotted = O/E 1; raw rate shown 0-100%). d, dispersion overall and within the largest segments.
- Statistics: Lenders need at least 100 decided applications; Garwood intervals for O/E; whiskers = 5th-95th percentile. Standardisation reduces product-mix differences but does not remove confounding (no credit score, income controls or lender policy).
- n and unit: n = 3180 eligible of 4926 lenders; unit = lender.
- Observed: Lender rate p5-p95: 1.2% to 41.0% (median 11.7%); IQR changes from 16.2 to 12.9 pp, while standard deviation changes from 0.134 to 0.137; the two dispersion measures are reported separately.

## fig3_q3_finance_bands

**Fig. 3 |** Denial rates move with applicant-finance bands, and recording of the bands differs by outcome.
- Panels: a-d, denial rate among decided applications by the four proposal-core financial bands: income, loan amount, loan-to-income and DTI. Filled points are valid bands; hollow points are NA, Exempt or outside-analysis-domain statuses. Each panel states its valid n and corrected V.
- Statistics: Wilson 95% intervals; bias-corrected Cramér's V on valid bands (2 x k table: denied vs approved); with millions of records every chi-square is 'significant', so V (effect size) is the quantity to read. Association only.
- n and unit: Panel-specific valid-band n is printed above each panel; unit = decided application record. CLTV and property-value results remain in the accompanying table.
- Observed: Largest association: dti (bias-corrected V = 0.497); denial rate range across its valid bands = 77.9 percentage points.

## fig4_q4_pricing

**Fig. 4 |** Pricing differs by product, purpose and lien; observable loan characteristics explain only part of the variation.
- Panels: a-b, interest rate and rate spread in selected high-volume product-purpose-lien segments. c-d, matched OLS coefficient panels for interest rate and rate spread with 95% CI, lender-clustered (CR1).
- Statistics: Boxes = IQR, whiskers = 5th-95th percentile (exact). OLS from cell sums; panels c-d show 95% CI, lender-clustered (CR1; G = 2,758 lenders for interest rate, 2,637 for rate spread); R-squared = 0.38 for interest rate and 0.29 for rate spread. Association only: unobserved credit quality and market timing are not in the data.
- n and unit: Panel a n = 4,360,522; panel b n = 3,979,822; panel c n = 4,119,701; panel d n = 4,074,963. Unit = originated LAR record.
- Observed: Regression explains 38.0% of interest-rate variance and 29.0% of rate-spread variance.

## fig5_engine_benchmark

**Fig. 5 |** Relative speed of Pandas and PySpark depends on data size, workload and on what is timed.
- Panels: a-b, compute-only median time (bars = min-max over repeats) for the lender group-by and the exact median by state. c, speedup T_Pandas / T_Spark per workload (above 1 = Spark faster; hatched bars = Tie / inconclusive). d, peak memory in fresh processes. e, start-up costs. f, read + compute scenario for both workloads (hollow markers = Tie / inconclusive).
- Statistics: Median of repeated runs after warm-up; seeded rotated cell order; Spark cached and counted before timing in a-c. A cell is Tie / inconclusive when the min-max ranges overlap or either cell is unstable (max/min > 1.5). Valid for these two workloads and this machine only.
- n and unit: n = 12,259,199 records at 100%; unit = application record; two aggregation tasks.
- Observed: lender group-by at 100%: Pandas 0.704 s, PySpark 0.454 s, faster = PySpark; median by state at 100%: Pandas 0.928 s, PySpark 0.565 s, faster = PySpark.

## fig6_predictive

**Fig. 6 |** Application characteristics support retrospective denial-risk ranking for lenders not seen in training.
- Panels: a, ROC curves on held-out lenders. b, calibration (deciles of predicted probability). c, test ROC-AUC. d, permutation importance.
- Statistics: Lender-grouped train/validation/test split; ROC-AUC with lender-bootstrap interval; calibration by deciles; Brier score in panel b. Analytical risk ranking, not a decision rule; no demographic inputs; inputs recorded differently by outcome removed.
- n and unit: n = 150,000 test applications (sample of the test lenders); unit = decided application.
- Observed: Best test ROC-AUC: Gradient boosting = 0.854; base rate = 0.153.
