"""Point-in-time applicant features built from the CMS sample.

An applicant is a beneficiary with no diabetes flag in 2008. Their application date is 2009-01-01. Only claims,
drug fills and enrollment facts dated before that day may be used. The label is the diabetes flag for 2009.

No race, sex, state or county feature is used. Weight is not in claims, so obesity and BMI diagnosis codes
(ICD-9 278 and V85) stand in for it. Drug classes need an NDC lookup that is not available here, so medication
features are counts and costs, not classes."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "cms")
APPLICATION_DATE = pd.Timestamp("2009-01-01")
CHRONIC = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", "SP_DEPRESSN", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"]
# ICD-9 prefixes grouped as a medical history an underwriter would read. 250 (diabetes) is handled separately.
DX_GROUPS = {
    "dx_hypertension": ("401", "402", "403", "404", "405"),
    "dx_lipid": ("272",),
    "dx_obesity_bmi": ("278", "V85"),
    "dx_heart": ("410", "411", "412", "413", "414", "427", "428"),
    "dx_kidney": ("585", "586", "584"),
    "dx_neuropathy_eye": ("357", "362", "366"),
    "dx_circulatory": ("440", "443", "444"),
}
DX_COLS = [f"ICD9_DGNS_CD_{i}" for i in range(1, 6)]


def _read(name):
    return pd.read_parquet(os.path.join(DATA, f"cms_{name}.parquet"))


def _dated(df, col):
    out = df.copy()
    out[col] = pd.to_datetime(out[col], format="%Y-%m-%d", errors="coerce")
    return out


def _dx_long(df, date_col):
    d = df[["DESYNPUF_ID", date_col] + DX_COLS].melt(id_vars=["DESYNPUF_ID", date_col], value_name="dx").dropna(subset=["dx"])
    d["dx"] = d["dx"].astype(str)
    return d


def history_cutoff(df, col, cutoff=APPLICATION_DATE):
    """Keep only rows dated strictly before the application date."""
    return df[df[col] < cutoff]


def build(cutoff=APPLICATION_DATE, include_dx250=False):
    bene = _read("bene")
    b08 = bene[bene.BENE_YEAR == 2008].set_index("DESYNPUF_ID")
    b09 = bene[bene.BENE_YEAR == 2009].set_index("DESYNPUF_ID")
    ids = b08.index.intersection(b09.index)
    applicants = [i for i in ids if b08.loc[i, "SP_DIABETES"] == 2]
    X = pd.DataFrame(index=pd.Index(applicants, name="DESYNPUF_ID"))
    b = b08.loc[applicants]
    birth = pd.to_datetime(b["BENE_BIRTH_DT"], errors="coerce")
    X["age"] = ((cutoff - birth).dt.days / 365.25).values
    X["esrd"] = (b["BENE_ESRD_IND"].astype(str).isin(["Y", "1"])).astype(int).values
    for c in CHRONIC:
        X[c.lower()] = (b[c] == 1).astype(int).values
    X["n_chronic"] = X[[c.lower() for c in CHRONIC]].sum(axis=1)
    X["months_part_b"] = b["BENE_SMI_CVRAGE_TOT_MONS"].astype(float).values

    car = history_cutoff(_dated(_read("carrier"), "CLM_FROM_DT"), "CLM_FROM_DT", cutoff)
    car = car[car.DESYNPUF_ID.isin(applicants)]
    op = history_cutoff(_dated(_read("outpatient"), "CLM_FROM_DT"), "CLM_FROM_DT", cutoff)
    op = op[op.DESYNPUF_ID.isin(applicants)]
    ip = history_cutoff(_dated(_read("inpatient"), "CLM_FROM_DT"), "CLM_FROM_DT", cutoff)
    ip = ip[ip.DESYNPUF_ID.isin(applicants)]
    X["carrier_claims"] = car.groupby("DESYNPUF_ID").size().reindex(applicants).fillna(0).values
    X["visit_days"] = pd.concat([car[["DESYNPUF_ID", "CLM_FROM_DT"]], op[["DESYNPUF_ID", "CLM_FROM_DT"]]]) \
        .drop_duplicates().groupby("DESYNPUF_ID").size().reindex(applicants).fillna(0).values
    X["outpatient_claims"] = op.groupby("DESYNPUF_ID").size().reindex(applicants).fillna(0).values
    X["inpatient_stays"] = ip.groupby("DESYNPUF_ID").size().reindex(applicants).fillna(0).values

    dx = pd.concat([_dx_long(car, "CLM_FROM_DT"), _dx_long(op, "CLM_FROM_DT"), _dx_long(ip, "CLM_FROM_DT")])
    for name, prefixes in DX_GROUPS.items():
        hit = dx[dx.dx.str.startswith(prefixes)].groupby("DESYNPUF_ID").size()
        X[name] = (hit.reindex(applicants).fillna(0) > 0).astype(int).values
    if include_dx250:
        d250 = dx[dx.dx.str.startswith("250")].groupby("DESYNPUF_ID").size()
        X["dx_diabetes_code_seen"] = (d250.reindex(applicants).fillna(0) > 0).astype(int).values

    pde = history_cutoff(_dated(_read("pde"), "SRVC_DT"), "SRVC_DT", cutoff)
    pde = pde[pde.DESYNPUF_ID.isin(applicants)]
    g = pde.groupby("DESYNPUF_ID")
    X["rx_fills"] = g.size().reindex(applicants).fillna(0).values
    X["rx_days_supply"] = g["DAYS_SUPLY_NUM"].sum().reindex(applicants).fillna(0).values
    X["rx_cost"] = g["TOT_RX_CST_AMT"].sum().reindex(applicants).fillna(0).values

    y = (b09.loc[applicants, "SP_DIABETES"] == 1).astype(int)
    cost09 = (b09.loc[applicants, ["MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR"]].sum(axis=1)).astype(float)
    return X, y, cost09
