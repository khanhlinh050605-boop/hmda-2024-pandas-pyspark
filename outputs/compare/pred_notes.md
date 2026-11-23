# Predictive extension: target, inputs, evaluation and limits

**Target and intended use.** Probability that a decided application is denied (action_taken = 3 versus 1 or 2), restricted to
non-exempt, consumer-purpose, closed-end, forward-mortgage applications. Intended use: analytical risk ranking and measuring how
much lender-level variation remains after application mix. It is not a lending decision rule and not a causal claim.

**Inputs and timing.** loan_type, loan_purpose, lien_status, occupancy_type, state_code, dti_band, construction_method, total_units, log_loan_amount, log_income, cltv, term_years, log_property_value, lender_log_apps, lender_share_loan_type_1, lender_share_loan_type_2, lender_share_purpose_1, lender_share_purpose_31, lender_share_purpose_32, lender_share_purpose_4.
The lender-profile features use the lender's full-year application mix and size within its split, never outcomes. Because that
full-year profile is not necessarily available when an individual application arrives, the stated use is retrospective risk ranking,
not intake-time scoring. The no-profile ablations and product-only logistic model are reported separately.

**Excluded by design.** Applicant age, sex, race and ethnicity are never read. Post-decision fields (denial reasons, interest rate,
rate spread, loan costs, purchaser type, HOEPA status) are excluded. Inputs recorded differently by outcome (gap above
20% in missing/status share between denied and approved) or nearly empty are dropped:
none.

**Evaluation.** Lenders are assigned to train / validation / test before the train-only missingness audit. Test labels therefore do
not affect feature retention or model selection. Model selection uses validation log-loss; reported metrics are on the test lenders:
ROC-AUC (with lender-bootstrap 95% interval),
average precision (AP), log-loss, Brier score and a decile calibration table. Models: logistic regression and histogram gradient boosting,
plus no-profile ablations, a product-only logistic baseline and the training-prior baseline.

**Limitations.** No credit score and no application date are available; HMDA public data are one year; loan amount is rounded; DTI
is often banded; CLTV/property value/DTI availability may reflect the underwriting process rather than intake-time information; observed
associations are not causal. Because the lender-level split removes lender identity, differences between
lenders show up as residual O/E, which describes unexplained variation, not misconduct. A high AUC would not justify use on individual
borrowers; a low AUC would reflect missing variables, not absence of risk differences.
