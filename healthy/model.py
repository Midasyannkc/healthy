"""Risk models. A regularized logistic regression is the main model: with a few hundred applicants it is
steadier than trees and its coefficients can be shown to a regulator. A boosted tree is compared, not assumed better."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7
SIMPLE = ["age", "n_chronic"]   # baseline an underwriter could apply by hand


def logistic(C=0.3):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def boosted():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=120, l2_regularization=1.0, random_state=SEED)


def oof_scores(make, X, y, repeats=5, folds=5):
    """Out-of-fold probabilities averaged over repeats, plus per-repeat AUC for an interval."""
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    P = np.zeros((repeats, len(y)))
    aucs = []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        P[k // folds, te] = m.predict_proba(X.iloc[te])[:, 1]
    for r in range(repeats):
        aucs.append(roc_auc_score(y, P[r]))
    return pd.Series(P.mean(axis=0), index=y.index), np.array(aucs)


def summarize(p, y):
    order = p.sort_values(ascending=False).index
    top = order[: max(1, len(order) // 10)]
    base = y.mean()
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "top_decile_rate": float(y.loc[top].mean()), "top_decile_lift": float(y.loc[top].mean() / base)}
