"""Train, evaluate with repeated cross validation, decide, and write out/report.json and out/decisions.csv."""
import json
import os

import numpy as np
import pandas as pd

from . import features, model as M, underwriting as U

OUT = os.path.join(features.ROOT, "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, cost = features.build(include_dx250=True)
    X = X.drop(columns=[c for c in X if X[c].nunique() == 1])
    rep = {"applicants": int(len(y)), "positives": int(y.sum()), "incidence": round(float(y.mean()), 4)}
    runs = {
        "baseline_age_and_chronic_count": (M.logistic, X[M.SIMPLE]),
        "logistic_no_diabetes_code": (M.logistic, X.drop(columns=["dx_diabetes_code_seen"])),
        "logistic_full": (M.logistic, X),
        "boosted_full": (M.boosted, X),
    }
    oof = {}
    rep["models"] = {}
    for name, (mk, Xs) in runs.items():
        p, aucs = M.oof_scores(mk, Xs, y)
        oof[name] = p
        rep["models"][name] = {**{k: round(v, 4) for k, v in M.summarize(p, y).items()},
                               "auc_repeat_min": round(float(aucs.min()), 4), "auc_repeat_max": round(float(aucs.max()), 4)}
    best = "logistic_full"
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(M.SEED)
    idx = np.arange(len(y))
    boot = []
    for _ in range(2000):
        s_ = rng.choice(idx, len(idx))
        if y.iloc[s_].nunique() == 2:
            boot.append(roc_auc_score(y.iloc[s_], oof[best].iloc[s_]))
    rep["models"][best]["auc_bootstrap_95ci"] = [round(float(np.quantile(boot, .025)), 3), round(float(np.quantile(boot, .975)), 3)]
    boot = []
    for _ in range(2000):
        s_ = rng.choice(idx, len(idx))
        if y.iloc[s_].nunique() == 2:
            boot.append(roc_auc_score(y.iloc[s_], oof["baseline_age_and_chronic_count"].iloc[s_]))
    rep["models"]["baseline_age_and_chronic_count"]["auc_bootstrap_95ci"] = [round(float(np.quantile(boot, .025)), 3), round(float(np.quantile(boot, .975)), 3)]
    cm = U.fit_cost_model(y, cost)
    d = U.decide(oof[best], cm)
    d["actual_diabetes_2009"] = y
    d["cost_2009"] = cost
    tiers = d.groupby("action").agg(applicants=("risk", "size"), actual_rate=("actual_diabetes_2009", "mean"),
                                    avg_price_relativity=("price_relativity", "mean"), avg_cost=("cost_2009", "mean")).round(4)
    rep["tiers"] = tiers.reset_index().to_dict(orient="records")
    rep["cost_model"] = {"avg_cost_if_diabetes": round(cm.cost_if_diabetes, 2), "avg_cost_if_not": round(cm.cost_if_not, 2)}
    # largest coefficients of the final model, for transparency
    final = M.logistic().fit(X, y)
    coef = pd.Series(final[-1].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    rep["top_coefficients_standardized"] = {k: round(float(v), 3) for k, v in coef.head(10).items()}
    d.round(4).to_csv(os.path.join(OUT, "decisions.csv"))
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
