"""
Search for AND-of-predicates ("axis-aligned boxes") in factor space that
contain MANY test days but ZERO train days.

Key idea
--------
A single-factor strict threshold (= train extremum) gives a hard ceiling
on test-hit count. Multi-factor AND can BEAT that ceiling: relax one
factor's threshold (some train days now sneak in), then add another factor
predicate that excludes exactly those near-miss train days while keeping
several test days. The AND region is what matters, not each axis alone.

Algorithm
---------
2D search (factor pair, direction pair):
  For every test day d as an "anchor", set t1 such that d satisfies the
  first predicate. For every other test day d' as a "corner", set t2 such
  that d' satisfies the second predicate. Check whether the resulting
  rectangle excludes all train. Count captured test days. Take the
  rectangle with the largest test count.

3D extension: greedy. Take the best 2D box, then add a third predicate
that further excludes any train hits ... but our 2D search already
guarantees train_hits=0, so adding a third predicate can only shrink
test coverage. So 3D is reported only as a tightening for interpretability,
not for capturing more.

The number of candidate rectangles per factor pair is at most |test|^2,
which is tiny (85^2 = 7225 for the COVID window).
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd


def _pred_mask(values: np.ndarray, op: str, t: float) -> np.ndarray:
    return values < t if op == "<" else values > t


def _count_test_train(
    f1_test, f2_test, f1_train, f2_train,
    op1, op2, t1, t2,
) -> tuple[int, int]:
    test_mask  = _pred_mask(f1_test,  op1, t1) & _pred_mask(f2_test,  op2, t2)
    train_mask = _pred_mask(f1_train, op1, t1) & _pred_mask(f2_train, op2, t2)
    return int(test_mask.sum()), int(train_mask.sum())


def search_2d_box(
    F_train: pd.DataFrame, F_test: pd.DataFrame,
    f1: str, op1: str, f2: str, op2: str,
    eps: float = 1e-9,
) -> dict | None:
    """
    Return the best (t1, t2) for the (f1, op1, f2, op2) box such that
    train_hits = 0 and test_hits is maximized. None if impossible.
    """
    f1_test  = F_test[f1].values
    f2_test  = F_test[f2].values
    f1_train = F_train[f1].values
    f2_train = F_train[f2].values

    best = None
    n_test = len(F_test)

    # t1 anchored at each test day's f1 value (slightly looser so that
    # test day satisfies the predicate in the chosen direction).
    for i in range(n_test):
        t1 = f1_test[i] + eps if op1 == "<" else f1_test[i] - eps

        # Train days that would satisfy predicate 1 alone:
        m1_train = _pred_mask(f1_train, op1, t1)
        if not m1_train.any():
            # Train is already excluded by f1 alone; t2 can be anything that
            # admits the most test days. Use the loosest t2.
            t2 = (f2_test.max() + eps) if op2 == "<" else (f2_test.min() - eps)
            test_hits, train_hits = _count_test_train(
                f1_test, f2_test, f1_train, f2_train, op1, op2, t1, t2)
            if train_hits == 0 and (best is None or test_hits > best["test_hits"]):
                best = {"f1": f1, "op1": op1, "t1": float(t1),
                        "f2": f2, "op2": op2, "t2": float(t2),
                        "test_hits": test_hits, "train_hits": 0}
            continue

        # Tightest t2 that excludes all those train days:
        # For op2 == '<': need every such train.f2 >= t2 -> t2 <= min(train.f2 in m1_train)
        # For op2 == '>': need every such train.f2 <= t2 -> t2 >= max(train.f2 in m1_train)
        if op2 == "<":
            t2 = float(f2_train[m1_train].min()) - eps
        else:
            t2 = float(f2_train[m1_train].max()) + eps

        test_hits, train_hits = _count_test_train(
            f1_test, f2_test, f1_train, f2_train, op1, op2, t1, t2)
        if train_hits != 0:
            continue
        if best is None or test_hits > best["test_hits"]:
            best = {"f1": f1, "op1": op1, "t1": float(t1),
                    "f2": f2, "op2": op2, "t2": float(t2),
                    "test_hits": test_hits, "train_hits": 0}

    return best


def search_all_pairs(F_train: pd.DataFrame, F_test: pd.DataFrame) -> pd.DataFrame:
    factors = list(F_train.columns)
    rows = []
    for f1, f2 in product(factors, factors):
        if f1 == f2:
            continue
        for op1, op2 in product(["<", ">"], ["<", ">"]):
            res = search_2d_box(F_train, F_test, f1, op1, f2, op2)
            if res is None:
                continue
            rows.append(res)
    df = pd.DataFrame(rows)
    df = df.sort_values("test_hits", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", default="model/full/factors.csv")
    p.add_argument("--test_start", default="2020-02-01")
    p.add_argument("--test_end",   default="2020-05-31")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    F = pd.read_csv(args.factors, index_col=0, parse_dates=True)
    if "alpha" in F.columns:
        F = F.drop(columns=["alpha"])

    test_mask = (F.index >= pd.Timestamp(args.test_start)) & (F.index <= pd.Timestamp(args.test_end))
    F_train, F_test = F[~test_mask], F[test_mask]
    print(f"train: {len(F_train)} days   test: {len(F_test)} days")
    print()

    df = search_all_pairs(F_train, F_test)

    # Drop rows with t2 = +/- inf (the "f1 alone is enough" trivial case);
    # these duplicate the strict 1D predicate.
    df_nontriv = df.copy()

    print(f"=== Top-{args.top} 2-factor AND boxes (train_hits=0 always) ===")
    show = df_nontriv.head(args.top).copy()
    show["t1"] = show["t1"].map("{:+.5f}".format)
    show["t2"] = show["t2"].map("{:+.5f}".format)
    print(show.to_string(index=False))
    print()

    # Show captured test dates for the top-1 box.
    if not df_nontriv.empty:
        top = df_nontriv.iloc[0]
        f1, op1, t1 = top["f1"], top["op1"], float(top["t1"])
        f2, op2, t2 = top["f2"], top["op2"], float(top["t2"])
        m_test = _pred_mask(F_test[f1].values, op1, t1) & _pred_mask(F_test[f2].values, op2, t2)
        m_train = _pred_mask(F_train[f1].values, op1, t1) & _pred_mask(F_train[f2].values, op2, t2)
        sel = F_test.index[m_test]
        print(f"=== Best box ===")
        print(f"  ({f1} {op1} {t1:+.5f})  AND  ({f2} {op2} {t2:+.5f})")
        print(f"  test_hits = {int(m_test.sum())}   train_hits = {int(m_train.sum())}")
        print(f"  dates: {sel.strftime('%Y-%m-%d').tolist()}")


if __name__ == "__main__":
    main()
