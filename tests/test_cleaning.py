import numpy as np
import pandas as pd

from src.core.config import load_config
from src.core.definitions import Defs
from src.engines.pandas_engine import build_master_frame, parse_numeric


def _frame():
    cfg = load_config("config.test.yaml")
    rows = []
    for income, dti, rate in [("1111", "50", "6.5"), ("1111.0", "60", "6.5"), ("100", ">60%", "Exempt")]:
        r = {c: "2" for c in [
            "lei", "action_taken", "state_code", "loan_type", "loan_purpose", "lien_status",
            "occupancy_type", "construction_method", "total_units", "open_end_line_of_credit",
            "business_or_commercial_purpose", "reverse_mortgage",
        ]}
        r.update({"lei": "L1", "action_taken": "1", "state_code": "VA", "loan_type": "1",
                  "loan_purpose": "1", "lien_status": "1", "loan_amount": "300000", "income": income,
                  "property_value": "400000", "loan_term": "360", "interest_rate": rate,
                  "rate_spread": "0.5", "total_loan_costs": "5000", "cltv": "75",
                  "debt_to_income_ratio": dti})
        rows.append(r)
    return build_master_frame(pd.DataFrame(rows), cfg, Defs(cfg), 0)


def test_income_1111_and_decimal_are_same_valid_value():
    vals, status = parse_numeric(pd.Series(["1111", "1111.0", "NA", "Exempt"]), "income")
    assert status.tolist() == ["VALUE", "VALUE", "NA", "UNPARSABLE"]
    assert vals[0] == vals[1] == 1111.0
    m = _frame()
    assert m.loc[0, "income_status"] == m.loc[1, "income_status"] == "VALUE"
    assert m.loc[0, "lti"] == m.loc[1, "lti"]
    assert m.loc[0, "has_reported_exempt_value"] == m.loc[1, "has_reported_exempt_value"] == 0
    assert m.loc[2, "has_reported_exempt_value"] == 1


def test_dti_boundaries_50_and_60_are_in_closed_50_60_band():
    m = _frame()
    assert m["dti_band"].tolist() == ["50-60", "50-60", ">60"]
    assert m["v_dti"].tolist() == [1, 1, 1]


def test_spark_field_specific_status_and_dti_boundaries():
    from pyspark.sql import functions as F
    from src.engines.spark_engine import dti_numeric_band_expr, get_spark, num_status

    cfg = load_config("config.test.yaml")
    spark = get_spark(cfg, "hmda_cleaning_unit")
    try:
        df = spark.createDataFrame([("1111", 50.0), ("1111.0", 60.0), ("Exempt", 60.1)], ["income", "dti"])
        status, _ = num_status(F, "income", "income")
        rows = (df.select(status.alias("status"), F.col("income").cast("double").alias("value"),
                          dti_numeric_band_expr(F, F.col("dti")).alias("band")).collect())
        assert [r.status for r in rows] == ["VALUE", "VALUE", "UNPARSABLE"]
        assert rows[0].value == rows[1].value == 1111.0
        assert [r.band for r in rows] == ["50-60", "50-60", ">60"]
    finally:
        spark.stop()

