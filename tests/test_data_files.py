import pandas as pd
import pytest

from healthy import features as F


@pytest.mark.parametrize("name", ["bene", "carrier", "inpatient", "outpatient", "pde"])
def test_parquet_and_csv_hold_the_same_rows(name):
    a = pd.read_parquet(f"{F.DATA}/cms_{name}.parquet")
    b = pd.read_csv(f"{F.DATA}/cms_{name}.csv", low_memory=False)
    assert a.shape == b.shape


def test_beneficiary_sample_is_500_people_over_two_years():
    bene = pd.read_parquet(f"{F.DATA}/cms_bene.parquet")
    assert bene.DESYNPUF_ID.nunique() == 500 and set(bene.BENE_YEAR) == {2008, 2009}
