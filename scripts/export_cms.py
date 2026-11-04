"""Convert the CMS DE-SynPUF sample tables (R .rda files from the claimsdb package) to parquet and csv."""
import os
import sys

import pandas as pd
import pyreadr

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for name in ("bene", "carrier", "inpatient", "outpatient", "pde"):
    df = pyreadr.read_r(os.path.join(src, f"{name}.rda"))[name]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(os.path.join(out, f"cms_{name}.parquet"), index=False)
    df.to_csv(os.path.join(out, f"cms_{name}.csv"), index=False)
    print(name, df.shape)
