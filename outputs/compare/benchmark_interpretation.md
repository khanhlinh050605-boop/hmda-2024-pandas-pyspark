# Benchmark: interpretation (numbers measured on this machine/workloads only)

Workloads: `lender_groupby` (per lender: count, mean loan amount, denial rate) and `median_by_state` (per state_code: count and exact median loan amount; the literal state "NA" is its own group). Each workload runs the full protocol (untimed warm-up, 7 repeats, seeded rotated cell order).

## 1. [lender_groupby] Which implementation was faster?

- 25% (3,066,841 records): Pandas 0.186 s, PySpark 0.366 s -> faster: **Pandas**; speedup (T_Pandas / T_Spark) = 0.51.
- 50% (6,131,779 records): Pandas 0.349 s, PySpark 0.408 s -> faster: **Tie / inconclusive** (min-max ranges overlap); speedup (T_Pandas / T_Spark) = 0.86.
- 100% (12,259,199 records): Pandas 0.704 s, PySpark 0.454 s -> faster: **PySpark**; speedup (T_Pandas / T_Spark) = 1.55.

## 2. [lender_groupby] Did relative performance change as data volume increased?

- Speedup went 0.51 -> 0.86 -> 1.55 from 25% to 100%; Pandas/Spark ratio rose with volume (Pandas time x3.79, PySpark time x1.24 from smallest to largest).
- Read + compute (separate scenario, lender_groupby): 25%: Pandas 1.60 s vs PySpark 0.73 s -> PySpark; 50%: Pandas 1.86 s vs PySpark 0.73 s -> PySpark; 100%: Pandas 2.21 s vs PySpark 0.75 s -> PySpark.

## 1. [median_by_state] Which implementation was faster?

- 25% (3,066,841 records): Pandas 0.219 s, PySpark 0.435 s -> faster: **Pandas**; speedup (T_Pandas / T_Spark) = 0.50.
- 50% (6,131,779 records): Pandas 0.447 s, PySpark 0.492 s -> faster: **Tie / inconclusive** (min-max ranges overlap); speedup (T_Pandas / T_Spark) = 0.91.
- 100% (12,259,199 records): Pandas 0.928 s, PySpark 0.565 s -> faster: **PySpark**; speedup (T_Pandas / T_Spark) = 1.64.

## 2. [median_by_state] Did relative performance change as data volume increased?

- Speedup went 0.50 -> 0.91 -> 1.64 from 25% to 100%; Pandas/Spark ratio rose with volume (Pandas time x4.23, PySpark time x1.30 from smallest to largest).
- Read + compute (separate scenario, median_by_state): 25%: Pandas 1.55 s vs PySpark 0.75 s -> PySpark; 50%: Pandas 1.84 s vs PySpark 0.82 s -> PySpark; 100%: Pandas 2.25 s vs PySpark 0.93 s -> PySpark.

## 3. What overheads affected Spark?

- import_pandas_s: 0.481 s (median of 3 fresh processes).
- spark_session_start_s: 2.647 s (median of 3 fresh processes).
- first_spark_action_s: 3.038 s (median of 3 fresh processes).
- Spark catalog/DataFrame caches were cleared before every read+compute cell and the saved physical plans show a Parquet FileScan. The operating-system file cache was not and cannot be cleared portably.
- In the primary scenario Spark data were cached and counted before timing, so task scheduling and serialisation overhead remain, but reading and session start are excluded.

## 4. Did memory limitations affect either implementation?

- [lender_groupby] pandas 25%: peak RSS 0.78 GiB (min 0.78, max 0.78; local process tree including JVM for Spark).
- [lender_groupby] pandas 50%: peak RSS 0.89 GiB (min 0.88, max 0.90; local process tree including JVM for Spark).
- [lender_groupby] pandas 100%: peak RSS 1.02 GiB (min 1.02, max 1.03; local process tree including JVM for Spark).
- [lender_groupby] spark 25%: peak RSS 1.87 GiB (min 1.75, max 1.95; local process tree including JVM for Spark).
- [lender_groupby] spark 50%: peak RSS 2.08 GiB (min 1.87, max 2.21; local process tree including JVM for Spark).
- [lender_groupby] spark 100%: peak RSS 1.83 GiB (min 1.79, max 1.92; local process tree including JVM for Spark).
- [median_by_state] pandas 25%: peak RSS 0.72 GiB (min 0.70, max 0.73; local process tree including JVM for Spark).
- [median_by_state] pandas 50%: peak RSS 0.88 GiB (min 0.86, max 0.88; local process tree including JVM for Spark).
- [median_by_state] pandas 100%: peak RSS 1.01 GiB (min 1.01, max 1.01; local process tree including JVM for Spark).
- [median_by_state] spark 25%: peak RSS 1.58 GiB (min 1.55, max 1.65; local process tree including JVM for Spark).
- [median_by_state] spark 50%: peak RSS 2.03 GiB (min 1.89, max 2.38; local process tree including JVM for Spark).
- [median_by_state] spark 100%: peak RSS 2.28 GiB (min 1.59, max 2.30; local process tree including JVM for Spark).
- Both engines completed all sizes without out-of-memory errors if this file was produced (the step would have failed otherwise).

## 5. Under what circumstances would Spark be preferable?

- This item is guidance, not a measurement: the results above come from one machine and two aggregation workloads (a lender-level group-by and an exact median by state). Spark becomes preferable when data no longer fit in one machine's RAM, when the work can be spread over several cores/nodes, or when the pipeline repeatedly reads large partitioned files; for data that fit in memory and a single aggregation, Pandas avoids Spark's start-up and scheduling costs.

## Result equivalence (required before comparing speed)

- [lender_groupby] compute_only, 25%: 4904 lenders, counts equal = True, floats within 1e-9 = True.
- [lender_groupby] compute_only, 50%: 4921 lenders, counts equal = True, floats within 1e-9 = True.
- [lender_groupby] compute_only, 100%: 4926 lenders, counts equal = True, floats within 1e-9 = True.
- [lender_groupby] read_compute, 25%: 4904 lenders, counts equal = True, floats within 1e-9 = True.
- [lender_groupby] read_compute, 50%: 4921 lenders, counts equal = True, floats within 1e-9 = True.
- [lender_groupby] read_compute, 100%: 4926 lenders, counts equal = True, floats within 1e-9 = True.
- [median_by_state] compute_only, 25%: 55 state groups, counts equal = True, floats within 1e-9 = True.
- [median_by_state] compute_only, 50%: 55 state groups, counts equal = True, floats within 1e-9 = True.
- [median_by_state] compute_only, 100%: 55 state groups, counts equal = True, floats within 1e-9 = True.
- [median_by_state] read_compute, 25%: 55 state groups, counts equal = True, floats within 1e-9 = True.
- [median_by_state] read_compute, 50%: 55 state groups, counts equal = True, floats within 1e-9 = True.
- [median_by_state] read_compute, 100%: 55 state groups, counts equal = True, floats within 1e-9 = True.
