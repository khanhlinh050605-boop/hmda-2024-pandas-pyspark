"""Benchmark workload 2: exact median loan_amount by state_code (the string 'NA' is a separate group)."""
import numpy as np
import pandas as pd
import pytest

from src.bench.workload import WORKLOADS, equivalent_state, pandas_state_workload, spark_rows_to_frame, spark_state_workload


def _df():
    rng = np.random.default_rng(7)
    states = np.array(["CA"] * 7 + ["NA"] * 4 + ["AK"] * 5 + ["TX"] * 1)
    return pd.DataFrame({"state_code": states, "loan_amount": rng.integers(10, 1000, len(states)).astype(float) * 1000})


def test_pandas_state_workload_exact_median_and_na_kept():
    d = _df()
    r = pandas_state_workload(d)
    assert r["state_code"].tolist() == ["AK", "CA", "NA", "TX"]          # sorted; 'NA' stays a group
    for _, row in r.iterrows():
        x = d.loc[d.state_code == row["state_code"], "loan_amount"].to_numpy()
        assert row["n_records"] == len(x)
        assert row["median_loan_amount"] == np.percentile(x, 50)        # linear interpolation, exact
    assert int(r["n_records"].sum()) == len(d)


def test_spark_state_workload_matches_pandas(tmp_path):
    pytest.importorskip("pyspark")
    from src.core.config import load_config
    from src.engines.spark_engine import get_spark

    d = _df()
    d.to_parquet(tmp_path / "x.parquet")
    spark = get_spark(load_config("config.test.yaml"))
    sp = spark_rows_to_frame(spark_state_workload(spark.read.parquet(tmp_path.as_posix())))
    pdres = pandas_state_workload(d)
    assert (sp["state_code"].tolist() == pdres["state_code"].tolist()) and "NA" in sp["state_code"].tolist()
    eq = equivalent_state(pdres, sp)
    assert eq == {"same_lenders": True, "counts_equal": True, "floats_close": True}


def test_equivalence_gate_detects_differences():
    a = pd.DataFrame({"state_code": ["AK", "NA"], "n_records": [3, 4], "median_loan_amount": [100.0, 200.0]})
    assert equivalent_state(a, a.copy())["floats_close"]
    b = a.copy(); b.loc[1, "median_loan_amount"] = 200.0 + 1e-6
    assert not equivalent_state(a, b)["floats_close"]
    b = a.copy(); b.loc[0, "n_records"] = 4
    assert not equivalent_state(a, b)["counts_equal"]
    b = a.copy(); b.loc[1, "state_code"] = "NaN"
    assert not equivalent_state(a, b)["same_lenders"]
    b = a.copy(); b.loc[1, "median_loan_amount"] = 200.0 * (1 + 5e-10)       # within |a-b| <= 1e-9 + 1e-9|b|
    assert equivalent_state(a, b)["floats_close"]


def test_registry_has_both_workloads():
    assert list(WORKLOADS) == ["lender_groupby", "median_by_state"]
