"""Turn a risk probability into an underwriting action and a price relativity.

Actions: PREFERRED, STANDARD, RATED (accept with an extra premium), REFER (hold for medical evidence such as a
nurse exam or an attending physician statement). Healthy never issues a final denial on its own: the top tier goes
to a human underwriter, who can decline. Price relativity is expected 2009 Medicare cost divided by the average,
from the same sample."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# relative risk bands: probability divided by the base rate
BANDS = [(0.0, 0.7, "PREFERRED"), (0.7, 1.3, "STANDARD"), (1.3, 2.0, "RATED"), (2.0, np.inf, "REFER")]
LOAD_CAP = 2.5


@dataclass
class CostModel:
    cost_if_diabetes: float
    cost_if_not: float
    base_rate: float

    @property
    def average_cost(self):
        return self.base_rate * self.cost_if_diabetes + (1 - self.base_rate) * self.cost_if_not

    def relativity(self, p):
        expected = p * self.cost_if_diabetes + (1 - p) * self.cost_if_not
        return np.minimum(expected / self.average_cost, LOAD_CAP)


def fit_cost_model(y, cost):
    return CostModel(float(cost[y == 1].mean()), float(cost[y == 0].mean()), float(y.mean()))


def decide(p, cm: CostModel):
    rr = p / cm.base_rate
    action = pd.Series("STANDARD", index=p.index)
    for lo, hi, name in BANDS:
        action[(rr >= lo) & (rr < hi)] = name
    rel = pd.Series(cm.relativity(p.values), index=p.index)
    rel[action == "PREFERRED"] = np.minimum(rel[action == "PREFERRED"], 1.0)
    return pd.DataFrame({"risk": p, "relative_risk": rr, "action": action, "price_relativity": rel})
