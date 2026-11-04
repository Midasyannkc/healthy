import pandas as pd

from healthy import features as F

X, Y, COST = F.build(include_dx250=True)
FORBIDDEN = ("race", "sex", "state", "county", "birth", "zip")


def test_every_applicant_had_no_diabetes_flag_in_2008():
    bene = pd.read_parquet(f"{F.DATA}/cms_bene.parquet")
    flags = bene[bene.BENE_YEAR == 2008].set_index("DESYNPUF_ID").loc[X.index, "SP_DIABETES"]
    assert (flags == 2).all()


def test_label_comes_from_2009_only():
    bene = pd.read_parquet(f"{F.DATA}/cms_bene.parquet")
    f09 = bene[bene.BENE_YEAR == 2009].set_index("DESYNPUF_ID").loc[X.index, "SP_DIABETES"]
    assert ((f09 == 1).astype(int) == Y).all()


def test_history_cutoff_drops_claims_on_or_after_application_date():
    df = pd.DataFrame({"d": pd.to_datetime(["2008-12-31", "2009-01-01", "2009-06-01"])})
    assert len(F.history_cutoff(df, "d")) == 1


def test_a_later_cutoff_can_only_add_history_so_features_use_dates():
    early, _, _ = F.build(cutoff=pd.Timestamp("2008-07-01"))
    assert (early.carrier_claims <= X.loc[early.index, "carrier_claims"]).all()
    assert early.carrier_claims.sum() < X.carrier_claims.sum()


def test_no_protected_or_geographic_feature():
    assert not [c for c in X.columns if any(w in c.lower() for w in FORBIDDEN)]


def test_cost_is_2009_and_not_a_feature():
    assert not [c for c in X.columns if "cost09" in c or "medreimb" in c.lower()]
    assert len(COST) == len(X)
