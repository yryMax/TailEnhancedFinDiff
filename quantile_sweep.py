"""
Trade-off scan: how does relaxing the "unseen" threshold from
train-extremum to train-q-quantile change test-hit count and
the (now non-zero) train-hit count?

For each factor f and each quantile q in [0.99 .. 1.0]:
    For the < direction:  threshold = quantile(train[f], 1 - q)
        test_hits  = #{ test days with f < threshold }
        train_hits = #{ train days with f < threshold }
    For the > direction:  threshold = quantile(train[f], q)
        symmetric

q = 1.0 reproduces the strict extremum (train_hits = 0 by construction
modulo ties). Smaller q => more test hits but train no longer empty.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", default="model/full/factors.csv")
    p.add_argument("--test_start", default="2020-02-01")
    p.add_argument("--test_end",   default="2020-05-31")
    p.add_argument("--quantiles", nargs="+", type=float,
                   default=[1.0, 0.999, 0.995, 0.99, 0.98, 0.95])
    p.add_argument("--factor", default=None,
                   help="Restrict scan to a single factor (e.g. 'market').")
    args = p.parse_args()

    F = pd.read_csv(args.factors, index_col=0, parse_dates=True)
    if "alpha" in F.columns:
        F = F.drop(columns=["alpha"])

    test_mask = (F.index >= pd.Timestamp(args.test_start)) & (F.index <= pd.Timestamp(args.test_end))
    F_train, F_test = F[~test_mask], F[test_mask]
    print(f"train: {len(F_train)} days   test: {len(F_test)} days")
    print()

    factors = [args.factor] if args.factor else list(F.columns)

    rows = []
    for f in factors:
        for q in args.quantiles:
            thr_lo = float(F_train[f].quantile(1 - q))
            thr_hi = float(F_train[f].quantile(q))
            rows.append({
                "factor": f, "op": "<", "q": q, "threshold": thr_lo,
                "test_hits":  int((F_test[f]  < thr_lo).sum()),
                "train_hits": int((F_train[f] < thr_lo).sum()),
            })
            rows.append({
                "factor": f, "op": ">", "q": q, "threshold": thr_hi,
                "test_hits":  int((F_test[f]  > thr_hi).sum()),
                "train_hits": int((F_train[f] > thr_hi).sum()),
            })

    df = pd.DataFrame(rows)
    # Keep rows where there is some test signal at the strictest setting,
    # so the table stays readable.
    keep_factors = (
        df[df["test_hits"] > 0]
        .groupby("factor")["test_hits"].max()
        .sort_values(ascending=False).index.tolist()
    )
    df = df[df["factor"].isin(keep_factors)]
    df["factor"] = pd.Categorical(df["factor"], categories=keep_factors, ordered=True)
    df = df.sort_values(["factor", "op", "q"], ascending=[True, True, False])

    print(df.to_string(index=False,
                       formatters={"threshold": "{:+.5f}".format,
                                   "q": "{:.3f}".format}))

    # OR over all (factor, op) at a chosen quantile.
    print()
    print("=== OR over all per-factor predicates at each q "
          "(unique test days, total train days) ===")
    for q in args.quantiles:
        or_test = pd.Series(False, index=F_test.index)
        or_train = pd.Series(False, index=F_train.index)
        for f in F.columns:
            thr_lo = float(F_train[f].quantile(1 - q))
            thr_hi = float(F_train[f].quantile(q))
            or_test  |= (F_test[f]  < thr_lo) | (F_test[f]  > thr_hi)
            or_train |= (F_train[f] < thr_lo) | (F_train[f] > thr_hi)
        print(f"  q={q:.3f}   test_hits={int(or_test.sum()):4d}    "
              f"train_hits={int(or_train.sum()):5d}")


if __name__ == "__main__":
    main()
