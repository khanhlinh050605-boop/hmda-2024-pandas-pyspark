import pandas as pd

from src.steps.s09_correctness import _catalog_signature, compare_tables, validation_passes


def test_schema_key_01_does_not_equal_1():
    a = pd.DataFrame({"state_code": ["01"], "n": [5]})
    b = pd.DataFrame({"state_code": ["1"], "n": [5]})
    result, _ = compare_tables(a, b, 1e-9, 1e-9, "keys", key_columns={"state_code"})
    assert not result["agree"]


def test_large_integer_counts_are_compared_exactly():
    n = 2**53 + 1
    a = pd.DataFrame({"n": [str(n)]})
    b = pd.DataFrame({"n": [str(n + 1)]})
    result, _ = compare_tables(a, b, 1e-9, 1e-9, "counts")
    assert not result["agree"]


def test_large_integer_count_with_decimal_suffix_is_still_exact():
    n = 2**53 + 1
    a = pd.DataFrame({"n_decided": [str(n)]})
    b = pd.DataFrame({"n_decided": [f"{n + 1}.0"]})
    result, _ = compare_tables(a, b, 1e-9, 1e-9, "counts")
    assert not result["agree"]


def test_catalog_signature_changes_when_same_table_is_missing_from_both():
    expected = _catalog_signature(["a.csv", "b.csv"])
    assert _catalog_signature(["a.csv"]) != expected


def test_one_shared_table_is_not_enough_when_required_catalog_is_missing():
    detail = pd.DataFrame([
        {"table": "only_table", "agree": True},
        {"table": "required_missing", "agree": False},
    ])
    headline = pd.DataFrame([{"quantity": "q", "agree": True}])
    assert not validation_passes(detail, headline)


def test_headline_failure_fails_even_when_tables_pass():
    detail = pd.DataFrame([{"table": "t", "agree": True}])
    headline = pd.DataFrame([{"quantity": "q", "agree": False}])
    assert not validation_passes(detail, headline)

