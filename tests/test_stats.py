import numpy as np, pandas as pd
from src.core import stats
from src.core.schema import FORBIDDEN_RE


def test_wilson_contains_p():
    p, lo, hi = stats.wilson(30, 100)
    assert lo < 0.3 < hi and 0 <= lo and hi <= 1 and abs(p - 0.3) < 1e-12


def test_cramers_v_independent_is_zero():
    t = np.array([[50, 50], [50, 50]])
    assert abs(stats.cramers_v(t)["v"]) < 1e-12


def test_cramers_v_perfect():
    assert abs(stats.cramers_v(np.array([[50, 0], [0, 50]]))["v"] - 1) < 1e-9


def test_forbidden_guard():
    for c in ["applicant_age", "co_applicant_sex", "derived_race", "derived_ethnicity", "tract_minority_population_percent", "denial_reason_1"]:
        assert FORBIDDEN_RE.search(c), c
    for c in ["loan_amount", "income", "state_code", "interest_rate"]:
        assert not FORBIDDEN_RE.search(c), c


# ---------------------------------------------------------------- OLS cluster CR1 by lender
def _cells_from_rows(df, cat, cont, y, extra_keys=()):
    cols = [y] + cont
    g = df.groupby(cat + list(extra_keys), as_index=False)
    out = g.size().rename(columns={"size": "n"})
    for c in cols:
        out[f"sum__{c}"] = g[c].sum()[c].to_numpy()
    for i, a in enumerate(cols):
        for b in cols[i:]:
            out[f"prod__{a}__{b}"] = (df[a] * df[b]).groupby([df[k] for k in cat + list(extra_keys)]).sum().to_numpy()
    return out


def _brute_cr1(X, yv, beta, groups):
    n, p = X.shape
    e = yv - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((p, p))
    gs = np.unique(groups)
    for g_ in gs:
        s = X[groups == g_].T @ e[groups == g_]
        meat += np.outer(s, s)
    G = len(gs)
    V = bread @ meat @ bread * G / (G - 1) * (n - 1) / (n - p)
    return np.sqrt(np.diag(V)), G


def _design(df, cat_levels, cont):
    cols = [np.ones(len(df))] + [(df[c].to_numpy() == l).astype(float) for c, l in cat_levels] + [df[c].to_numpy() for c in cont]
    return np.column_stack(cols)


def test_cluster_se_matches_bruteforce_multi_obs_per_lender():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"g": rng.choice(["a", "b", "c"], n, p=[.6, .3, .1]), "x": rng.normal(size=n),
                       "lei": rng.choice([f"L{i}" for i in range(40)], n)})
    lender_eff = pd.Series(rng.normal(size=40), index=[f"L{i}" for i in range(40)])
    df["y"] = 1 + 2 * df["x"] + (df["g"] == "b") * 0.5 + df["lei"].map(lender_eff) + rng.normal(size=n)
    cells = _cells_from_rows(df, ["g"], ["x"], "y")
    lcells = _cells_from_rows(df, ["g"], ["x"], "y", ["lei"])
    res = stats.ols_from_cells(cells, ["g"], ["x"], "y", lender_cells=lcells)
    base = stats.ols_from_cells(cells, ["g"], ["x"], "y")
    assert np.allclose(res["coef"]["coef"], base["coef"]["coef"], rtol=0, atol=0)       # coefficients unchanged
    X = _design(df, [("g", "b"), ("g", "c")], ["x"])
    se, G = _brute_cr1(X, df["y"].to_numpy(), res["coef"]["coef"].to_numpy(), df["lei"].to_numpy())
    assert np.allclose(res["coef"]["se_cluster"], se, rtol=1e-9)
    assert res["summary"]["G"] == G == 40
    assert np.allclose(res["coef"]["t_cluster"], res["coef"]["coef"] / se)
    assert np.allclose(res["coef"]["ci_hi_cluster"] - res["coef"]["ci_lo_cluster"], 2 * stats.Z95 * se)


def test_cluster_se_one_obs_per_lender_is_classical_times_correction():
    # one observation per lender, |residual| constant => clustered SE = classical SE x sqrt(G(N-1)/((G-1)N)); with G=N the factor = 1
    m = 400
    # (g,x) repeated twice with noise +1 / -1 => noise orthogonal to every function of (g,x)
    df = pd.DataFrame({"g": np.tile(["a", "b", "a", "b"], m // 4), "x": np.tile([1.0, 2.0, 3.0, 4.0], m // 4)})
    half = m // 2
    df = pd.concat([df.iloc[:half], df.iloc[:half]], ignore_index=True)
    e = np.r_[np.ones(half), -np.ones(half)]
    df["y"] = 1 + 2 * df["x"] + (df["g"] == "b") * 0.5 + e
    df["lei"] = [f"L{i}" for i in range(m)]
    cells = _cells_from_rows(df, ["g"], ["x"], "y")
    lcells = _cells_from_rows(df, ["g"], ["x"], "y", ["lei"])
    assert (lcells["n"] == 1).all()
    res = stats.ols_from_cells(cells, ["g"], ["x"], "y", lender_cells=lcells)
    G = res["summary"]["G"]
    assert G == m
    factor = np.sqrt(G * (m - 1) / ((G - 1) * m))
    assert np.allclose(res["coef"]["se_cluster"], res["coef"]["se"] * factor, rtol=1e-9)
