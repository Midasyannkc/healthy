import pandas as pd

from healthy import underwriting as U

CM = U.CostModel(cost_if_diabetes=6000.0, cost_if_not=1000.0, base_rate=0.2)


def test_higher_risk_never_gets_a_lower_price():
    p = pd.Series([0.02, 0.1, 0.2, 0.4, 0.7, 0.95])
    d = U.decide(p, CM)
    assert d.price_relativity.is_monotonic_increasing


def test_price_load_is_capped():
    assert U.decide(pd.Series([0.99]), CM).price_relativity.iloc[0] <= U.LOAD_CAP


def test_actions_follow_relative_risk_bands():
    d = U.decide(pd.Series([0.05, 0.2, 0.3, 0.5]), CM)
    assert d.action.tolist() == ["PREFERRED", "STANDARD", "RATED", "REFER"]


def test_the_engine_never_issues_an_automatic_denial():
    d = U.decide(pd.Series([0.0, 0.5, 1.0]), CM)
    assert not d.action.str.contains("DECLINE|DENY").any()
    assert d.action.iloc[-1] == "REFER"


def test_preferred_is_never_priced_above_standard():
    d = U.decide(pd.Series([0.01, 0.1]), CM)
    assert (d.price_relativity <= 1.0).all()
