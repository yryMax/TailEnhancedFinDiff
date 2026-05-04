"""
3D-AND search for axis-aligned boxes with train_hits=0 and many test_hits.

Why 3D > 2D
-----------
A train-zero 2D rectangle is bounded above by the strict-1D ceiling (e.g.,
"market < train_min(market)" gives 3 hits in the COVID window). 2D AND can
only marginally improve on this (4 hits in our run).

3D AND can exceed both: deliberately relax the first two factor thresholds
so the 2D rectangle captures MANY test days but ALSO some train days, then
add a third predicate whose threshold lies strictly between the captured
test-day distribution and the captured train-day distribution on a third
factor — kicking out all train hits while preserving most test hits.

This works iff the train near-misses and the test extremes have visibly
different signatures on at least one third factor — i.e., the test window's
stress event is a different *kind* of event from the train near-misses.

Algorithm
---------
Anchor the 2D rectangle at a pair of test days (d_i, d_j); tighten the
3rd predicate against any train days that fall in the rectangle.

  For each (f1, op1, f2, op2, f3, op3) triple:
    For each (i, j) pair of test days (i indexes f1, j indexes f2):
      t1 set so d_i satisfies pred1
      t2 set so d_j satisfies pred2
      M_test  = test days in 2D rectangle
      M_train = train days in 2D rectangle
      If |M_train| == 0: 2D already train-zero; 3D doesn't help here.
      Else:
        For op3 = '<': t3 = min over M_train of f3, captured_test = #M_test with f3 < t3
        For op3 = '>': t3 = max over M_train of f3, captured_test = #M_test with f3 > t3
      Track best (captured_test, train_hits=0).

Complexity ~ |test|^2 * #factor_triples — fine for small test windows.
"""

from __future__ import annotations

import argparse
from itertools import permutations

import numpy as np
import pandas as pd


def _mask(values: np.ndarray, op: str, t: float) -> np.ndarray:
    return values < t if op == "<" else values > t


def search_3d(
    F_train: pd.DataFrame,
    F_test: pd.DataFrame,
    eps: float = 1e-9,
    top: int = 15,
    anchor_topk: int = 30,
) -> pd.DataFrame:
    """
    Anchor pruning: for each (f1, op1) pair, only consider t1 anchored at
    the `anchor_topk` MOST EXTREME test days on that factor in that
    direction (the only ones that could plausibly anchor a tail box).
    Same for (f2, op2). This cuts the search by ~(85/topk)^2 with no loss
    of optimality for tail-stress objectives.
    """
    factors = list(F_train.columns)
    n_test = len(F_test)
    test_arr  = {f: F_test[f].values  for f in factors}
    train_arr = {f: F_train[f].values for f in factors}

    # Pre-compute extreme-direction anchor indices per (factor, op).
    def topk_idx(values: np.ndarray, op: str, k: int) -> np.ndarray:
        # For "<", we want the LARGEST f1_test values to anchor (loosest box that still
        # captures the test day). For ">", smallest values.
        # Wait — actually for "<" with t1 = v1_test[i] + eps, larger v1_test[i] gives a
        # LOOSER t1 (more train+test fall under it). So large v1_test[i] = loosest box,
        # small = tightest. Both ends of test distribution are useful anchors. Use both.
        order = np.argsort(values)
        return np.unique(np.concatenate([order[:k], order[-k:]]))

    anchors = {(f, op): topk_idx(test_arr[f], op, anchor_topk)
               for f in factors for op in ("<", ">")}

    results = []
    factor_pairs = list(permutations(factors, 2))   # ordered (f1, f2), f3 separate
    for f1, f2 in factor_pairs:
        v1_test, v2_test = test_arr[f1], test_arr[f2]
        v1_train, v2_train = train_arr[f1], train_arr[f2]

        for op1 in ("<", ">"):
            anchors1 = anchors[(f1, op1)]
            for op2 in ("<", ">"):
                anchors2 = anchors[(f2, op2)]

                for i in anchors1:
                    t1 = v1_test[i] + eps if op1 == "<" else v1_test[i] - eps
                    m1_train = _mask(v1_train, op1, t1)
                    m1_test  = _mask(v1_test,  op1, t1)
                    if not m1_test.any():
                        continue

                    for j in anchors2:
                        t2 = v2_test[j] + eps if op2 == "<" else v2_test[j] - eps
                        m12_test  = m1_test  & _mask(v2_test,  op2, t2)
                        if not m12_test.any():
                            continue
                        m12_train = m1_train & _mask(v2_train, op2, t2)

                        for f3 in factors:
                            if f3 == f1 or f3 == f2:
                                continue
                            v3_test, v3_train = test_arr[f3], train_arr[f3]

                            for op3 in ("<", ">"):
                                if not m12_train.any():
                                    t3 = (v3_test.max() + eps) if op3 == "<" else (v3_test.min() - eps)
                                    hits = int(m12_test.sum())
                                else:
                                    train_in_v3 = v3_train[m12_train]
                                    if op3 == "<":
                                        t3 = float(train_in_v3.min()) - eps
                                    else:
                                        t3 = float(train_in_v3.max()) + eps
                                    m_final = m12_test & _mask(v3_test, op3, t3)
                                    hits = int(m_final.sum())
                                    if hits == 0:
                                        continue

                                results.append({
                                    "f1": f1, "op1": op1, "t1": float(t1),
                                    "f2": f2, "op2": op2, "t2": float(t2),
                                    "f3": f3, "op3": op3, "t3": float(t3),
                                    "test_hits": hits,
                                })

    df = pd.DataFrame(results)
    df = df.sort_values("test_hits", ascending=False)

    # Deduplicate boxes that are identical sets of (factor, op) regardless of order.
    df["_sig"] = df.apply(
        lambda r: tuple(sorted([(r.f1, r.op1, round(r.t1, 8)),
                                (r.f2, r.op2, round(r.t2, 8)),
                                (r.f3, r.op3, round(r.t3, 8))])),
        axis=1,
    )
    df = df.drop_duplicates("_sig").drop(columns="_sig").reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", default="model/full/factors.csv")
    p.add_argument("--test_start", default="2020-02-01")
    p.add_argument("--test_end",   default="2020-05-31")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    F = pd.read_csv(args.factors, index_col=0, parse_dates=True)
    if "alpha" in F.columns:
        F = F.drop(columns=["alpha"])

    test_mask = (F.index >= pd.Timestamp(args.test_start)) & (F.index <= pd.Timestamp(args.test_end))
    F_train, F_test = F[~test_mask], F[test_mask]
    print(f"train: {len(F_train)} days   test: {len(F_test)} days")
    print()

    df = search_3d(F_train, F_test, top=args.top)
    print(f"=== Top-{args.top} 3-factor AND boxes (train_hits=0 always) ===")
    show = df.head(args.top).copy()
    for c in ("t1", "t2", "t3"):
        show[c] = show[c].map("{:+.5f}".format)
    print(show.to_string(index=False))
    print()

    # Show captured test dates for top-1
    if not df.empty:
        top = df.iloc[0]
        v1_test = F_test[top.f1].values
        v2_test = F_test[top.f2].values
        v3_test = F_test[top.f3].values
        v1_train = F_train[top.f1].values
        v2_train = F_train[top.f2].values
        v3_train = F_train[top.f3].values
        m_test = (_mask(v1_test, top.op1, float(top.t1))
                  & _mask(v2_test, top.op2, float(top.t2))
                  & _mask(v3_test, top.op3, float(top.t3)))
        m_train = (_mask(v1_train, top.op1, float(top.t1))
                   & _mask(v2_train, top.op2, float(top.t2))
                   & _mask(v3_train, top.op3, float(top.t3)))
        print(f"=== Best box ===")
        print(f"  ({top.f1} {top.op1} {float(top.t1):+.5f})  AND  "
              f"({top.f2} {top.op2} {float(top.t2):+.5f})  AND  "
              f"({top.f3} {top.op3} {float(top.t3):+.5f})")
        print(f"  test_hits = {int(m_test.sum())}   train_hits = {int(m_train.sum())}")
        sel = F_test.index[m_test]
        print(f"  dates: {sel.strftime('%Y-%m-%d').tolist()}")


if __name__ == "__main__":
    main()
