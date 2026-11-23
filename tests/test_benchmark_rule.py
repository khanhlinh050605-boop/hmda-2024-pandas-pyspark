"""'Faster method' rule: conclude only when the min-max ranges do not overlap and no unstable flag is set."""
from src.steps.s10_benchmark import TIE, decide_faster


def cell(mn, mx, unstable=False):
    return {"min_s": mn, "max_s": mx, "median_s": (mn + mx) / 2, "unstable": unstable}


def test_non_overlapping_ranges_decide():
    assert decide_faster(cell(1.0, 1.2), cell(2.0, 2.4)) == "Pandas"
    assert decide_faster(cell(2.0, 2.4), cell(1.0, 1.2)) == "PySpark"


def test_overlapping_ranges_are_a_tie():
    assert decide_faster(cell(1.0, 1.6), cell(1.5, 2.0)) == TIE


def test_unstable_cell_is_never_decisive():
    assert decide_faster(cell(1.0, 1.2, unstable=True), cell(2.0, 2.4)) == TIE
    assert decide_faster(cell(1.0, 1.2), cell(2.0, 2.4, unstable=True)) == TIE
