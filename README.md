# HMDA 2024 – Big Data Analysis Final Project (Group 5)

Ninh Ngọc Anh 11230734 · Bùi Thị Khánh Linh 11230756 · Nguyễn Thị Thu Phương 11230770

Reference data: **HMDA 2024 One-Year National Loan-Level Dataset**, freeze date 2026-06-02,
12,259,199 LAR records. The URLs and SHA-256 hashes of the LAR/TS files are recorded in `data/raw/source_manifest.json` after preparation.
Two independent engines (Pandas/pyarrow and PySpark 3.5.3) run the same definitions; counts are compared exactly, keys as strings,
and floats within a tolerance. Correctness only PASSES when the full pinned catalogue (164 raw aggregations + 67 final tables)
is present and all 30 mandatory headline quantities have values and agree.

## Research questions (per proposal v1)
- **Main**: how do the denial rate (among decided applications) and the origination rate vary by loan type, purpose, state and lender.
- **Q1** denial rate by loan type × purpose, and whether it is consistent across states (Spearman, indirect standardisation).
- **Q2** dispersion of denial rates across lenders, and within a single product segment.
- **Q3** how income, loan amount, LTI, DTI, CLTV and property value relate to denial (Cramér's V). Association only.
- **Q4** interest rate / rate spread of originated loans by loan type/purpose/lien and loan characteristics (OLS from cross-product matrices). Association only.
- **Predictive extension (step 11)**: denial among decided applications; train/validation/test split by lender before any outcome-based audit,
  logistic regression vs gradient boosting and ablations. This is a retrospective ranking: some variables may reflect the underwriting stage,
  and the lender profiles use full-year mix and are not described as available at application time.

## Data rules (final)
- **Never read or use** demographic variables (age, sex, race, ethnicity, including co-applicants) and their derived columns,
  `tract_minority*`, `denial_reason`, `hoepa_status`, `purchaser_type`, `aus_`. A "forbidden guard" (`FORBIDDEN_RE` in `src/core/schema.py`) blocks these columns in both engines.
- No external data is used for the analysis. The TS (Transmittal Sheet) file is only used to reconcile row counts and to attach lender names/regulators;
  outside documents are only cited to verify definitions and put results in context.
- Cleaning rules are identical in both engines: see `cleaning_rules.md`.
- Synthetic data (`scripts/make_synthetic.py`) is only for testing the pipeline. No result in the report comes from it.

## Installation and building the input (portable paths)
1. Python **3.12**, Java **17+** (on Windows, `HADOOP_HOME`/`winutils.exe` if your Spark setup requires it).
   The included benchmark was measured with OpenJDK **17.0.21**; this is the measurement environment, not a requirement to use that exact Java version.
2. `python -m venv .venv`, activate the environment, then `pip install -r requirements.txt`.
3. Download exactly the two 2024 One-Year files from the official URLs below, then run the streaming command; the script never extracts the text file to disk:
   ```
   python scripts/prepare_official_2024.py --lar-zip <2024_public_lar_one_year_pipe.zip> --ts-zip <2024_public_ts_one_year_pipe.zip>
   ```
   LAR: `https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_lar_one_year_pipe.zip`
   TS: `https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_ts_one_year_pipe.zip`
4. `config.yaml` already uses the relative path `data/raw/...`; run `python scripts/check_env.py`.

## Running
```
python run.py --list                      # list the steps
python run.py --steps all --engine both   # everything (steps 00-12)
python run.py --steps 05                  # re-run a single step
python run.py --steps 04-07 --engine pandas
python run.py --steps 06 --force          # ignore the cache
```
Each step has a manifest containing the config hash and a fingerprint of all Python code, so changing code never reuses stale results.
The submission run should still use `--force`; the benchmark refuses to run if the current correctness check has not PASSED.

| Step | Module | Content |
|---|---|---|
| 00 | s00_inspect | schema, number of files/rows, reconciliation with TS |
| 01 | s01_prepare | build the analytical Parquet master (flags and bands, duplicate flag, sampling key) |
| 02 | s02_quality | missing/status/out-of-domain values, duplicates |
| 03 | s03_descriptive | descriptive overview |
| 04–07 | s04_q1 … s07_q4 | Q1–Q4 (+ Main) |
| 08 | s08_robustness | exclusion variants (query filters, no data deleted) |
| 09 | s09_correctness | compare every table between the two engines |
| 10 | s10_benchmark | two workloads (`lender_groupby`, `median_by_state`) on the same Pandas-master Parquet/layout/projection for both engines; compute-only and read+compute; Spark physical plan |
| 11 | s11_predictive | predictive extension |
| 12 | s12_report_assets | figures (cnsplots) and three-line tables (CSV + LaTeX), `figure_captions.md` |

Results: `outputs/<engine>/raw`, `outputs/<engine>/tables`, `outputs/compare/tables`, `outputs/figures`, `outputs/tables_report`.
Use both `raw/` and `tables/` to fully re-check the 231/231 claim; with `tables/` only, just the 67 final pairs can be re-checked.
Intermediate data: `data/interim`.

## Benchmark (step 10)
- Two workloads: `lender_groupby` (record count, mean loan amount, number of decided/denied applications and denial rate by lender) and `median_by_state` (exact median loan amount by state, 55 groups including the `NA` group).
- Two scenarios: compute-only (Spark cached and counted before timing) and read+compute (includes reading Parquet, Spark cache cleared before each run).
- Nested 25/50/100% samples, one untimed warm-up per workload and scenario, 7 timed repeats in seeded rotated order. A cell is labelled "Tie / inconclusive" when the min–max ranges of the two engines overlap or one engine is unstable (max/min > 1.5). The operating-system cache is not cleared.
- Outputs: `outputs/compare/tables/benchmark_summary.csv`, `benchmark_timings_raw.csv`, `benchmark_memory.csv`, `benchmark_setup_cost.csv`, `benchmark_equivalence.csv`, `outputs/tables_report/table3_benchmark.csv`, `outputs/figures/fig5_engine_benchmark.pdf`.
- Equivalence gate: every `median_by_state` cell is compared between Pandas and PySpark before timings are used; six output tables are written to `outputs/compare/benchmark_tables/` and six `kind=benchmark` rows are added to `correctness_detail.csv`. Re-running step 09 overwrites these rows, in which case step 10 must be re-run.

## Figures in the report
Figures 0–6 are produced by step 12 in `outputs/figures/` (PDF and PNG). Figure `fig6_predictive` (predictive extension) is in Appendix A of the report and numbered Figure 7.

## Tests
`pytest tests -q` (23 tests, including 4 for the `median_by_state` workload) checks the statistics, the forbidden-column guard, `income=1111/1111.0`, DTI 50/60 in both engines,
duplicate semantics, the correctness catalogue/key/count/headline checks and an end-to-end synthetic run. The synthetic fixture has 1,200 rows in repeated groups and 600 excess copies.
To run the correctness check end-to-end on synthetic data: `python scripts/make_synthetic.py`, then
`python run.py --config config.test.yaml --steps 00-09 --engine both --force`. Benchmark, predictive and report assets use the full-data outputs.

## Key conventions
- `has_reported_exempt_value` is a record-level flag, and `lenders_with_exempt_values` only means the lender has at least one such record; it is not a legal conclusion that the lender is exempt from reporting.
- `is_dup` marks every row that belongs to a repeated fingerprint. The baseline keeps them; the sensitivity analysis removes all repeated groups and reports separately the number of rows in groups and the number of excess copies.
- `~OUTSIDE_DOMAIN` means the value cannot be used for the log/ratio/statistic in question; it is not automatically wrong data.

## Limitations to state in the report
- The Q4 regressions report both classical and lender-clustered standard errors (CR1); conclusions should be read from the lender-clustered column. Interpretation is association only.
- Repeated fingerprints are defined on the permitted columns only, so they are a practical definition; they do not automatically prove that records are wrongly duplicated.
- The benchmark applies only to the two workloads and this machine; Spark start-up time is reported separately.
- Q3/Q4/predictive results draw no causal conclusion and do not assess discrimination (demographic variables were deliberately not used).
- The predictive section is a retrospective risk ranking. Lender profiles use the full-year mix within each split; the ablation without profiles is reported separately.
