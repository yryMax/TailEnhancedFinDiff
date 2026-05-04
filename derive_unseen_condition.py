"""
Reverse-engineer a stress condition that holds for >0 days in a chosen
test window but for 0 days in the rest of the sample (train).

Allowed predicates: `factor < value` or `factor > value`, optionally combined
by AND (a "box" in factor space outside the train hull).

Computation steps
-----------------
1. Load full25y factor returns and split by `--test_start / --test_end`.
2. For each factor f, compute the strict train extrema:
       lo_f = min(F_train[f]),  hi_f = max(F_train[f])
   The single-factor predicates `f < lo_f` and `f > hi_f` are guaranteed to
   have ZERO train-day hits — anything stricter than the historical extreme.
3. Count test-day hits for each single predicate. This is the per-factor
   "unseen capacity" of the test window.
4. Greedy AND combination: start from the predicate with the largest
   test-hit count and successively add the predicate that retains the most
   test-day coverage while increasing economic specificity. AND of single
   train-zero predicates stays train-zero, so the search only optimises
   interpretability vs. test sample size.
5. Report:
       - per-factor single predicates
       - the largest single-factor predicate
       - greedy AND chains of length 1..K with their test-day counts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def split_train_test(F: pd.DataFrame, test_start: str, test_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = (F.index >= pd.Timestamp(test_start)) & (F.index <= pd.Timestamp(test_end))
    return F[~test_mask].copy(), F[test_mask].copy()


def per_factor_predicates(F_train: pd.DataFrame, F_test: pd.DataFrame, eps: float = 0.0):
    """
    Return a list of dicts, one per (factor, direction), each with:
        factor, op ('<' or '>'), threshold, test_hits (int), train_hits (int, =0 by construction)
    """
    out = []
    for f in F_train.columns:
        lo = float(F_train[f].min()) - eps
        hi = float(F_train[f].max()) + eps

        mask_lo = F_test[f] < lo
        mask_hi = F_test[f] > hi

        out.append({
            "factor": f, "op": "<", "threshold": lo,
            "test_hits": int(mask_lo.sum()),
            "train_hits": int((F_train[f] < lo).sum()),
            "mask": mask_lo,
        })
        out.append({
            "factor": f, "op": ">", "threshold": hi,
            "test_hits": int(mask_hi.sum()),
            "train_hits": int((F_train[f] > hi).sum()),
            "mask": mask_hi,
        })
    return out


def greedy_and_chain(preds: list[dict], max_len: int = 4):
    """
    Greedy: at each step pick the predicate (not already chosen, and not on a
    factor/op already chosen) whose intersection with the current AND keeps
    the most test days.

    Returns list of (chain, joint_mask, test_hits) for chain lengths 1..max_len.
    """
    pool = [p for p in preds if p["test_hits"] > 0]
    if not pool:
        return []

    # Seed with the predicate that has the most test hits.
    pool.sort(key=lambda p: -p["test_hits"])
    chain = [pool[0]]
    joint = pool[0]["mask"].copy()
    history = [(list(chain), joint.copy(), int(joint.sum()))]

    used_factors = {pool[0]["factor"]}
    while len(chain) < max_len:
        best, best_hits = None, -1
        for p in pool:
            if p["factor"] in used_factors:
                continue
            new_mask = joint & p["mask"]
            n = int(new_mask.sum())
            if n > best_hits and n > 0:
                best, best_hits, best_mask = p, n, new_mask
        if best is None:
            break
        chain.append(best)
        used_factors.add(best["factor"])
        joint = best_mask
        history.append((list(chain), joint.copy(), best_hits))

    return history


def fmt_pred(p: dict) -> str:
    return f"{p['factor']:>10s} {p['op']} {p['threshold']:+.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default="model/full/factors.csv",
                        help="Factor returns CSV (output of factor_model.py).")
    parser.add_argument("--test_start", default="2008-08-01")
    parser.add_argument("--test_end",   default="2009-06-30")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="Safety margin added inside the train hull. "
                             "Use eps>0 to require test days to exceed train extremes by a margin.")
    parser.add_argument("--max_len", type=int, default=4,
                        help="Max length of greedy AND chain to report.")
    parser.add_argument("--out", default="model/full/unseen_condition.csv",
                        help="Where to dump the predicates table.")
    args = parser.parse_args()

    F = pd.read_csv(args.factors, index_col=0, parse_dates=True)
    if "alpha" in F.columns:
        F = F.drop(columns=["alpha"])

    F_train, F_test = split_train_test(F, args.test_start, args.test_end)
    print(f"train: {len(F_train)} days  ({F_train.index.min().date()} .. {F_train.index.max().date()})")
    print(f"test : {len(F_test)} days  ({F_test.index.min().date()} .. {F_test.index.max().date()})")
    print()

    # Step 1+2+3: per-factor strict-extreme predicates.
    preds = per_factor_predicates(F_train, F_test, eps=args.eps)
    table = pd.DataFrame([{k: v for k, v in p.items() if k != "mask"} for p in preds])
    table = table.sort_values("test_hits", ascending=False).reset_index(drop=True)

    print("=== Single-factor unseen predicates (train_hits is 0 by construction) ===")
    print(table.to_string(index=False,
                          formatters={"threshold": "{:+.5f}".format}))
    print()

    nonzero = table[table["test_hits"] > 0]
    if nonzero.empty:
        print("No single-factor predicate hits any test day. "
              "Try a different test window or set eps<0.")
        return

    best = preds[
        next(i for i, p in enumerate(preds)
             if p["factor"] == nonzero.iloc[0]["factor"]
             and p["op"] == nonzero.iloc[0]["op"])
    ]
    print(f"=== Best single predicate ===\n  {fmt_pred(best)}   "
          f"=> {best['test_hits']} test days, 0 train days")
    print()

    # Step 4: greedy AND chains.
    history = greedy_and_chain(preds, max_len=args.max_len)
    print(f"=== Greedy AND chains (test_hits, train_hits=0 always) ===")
    for chain, mask, hits in history:
        rule = "  AND  ".join(fmt_pred(p) for p in chain)
        print(f"  len={len(chain)}  hits={hits:3d}  {rule}")
    print()

    # Step 5: OR of all single train-zero predicates (unique test days).
    or_mask = pd.Series(False, index=F_test.index)
    or_terms = []
    for p in preds:
        if p["test_hits"] > 0:
            or_mask = or_mask | p["mask"]
            or_terms.append(fmt_pred(p))
    print(f"=== OR over all single train-zero predicates ===")
    print(f"  unique test days hit: {int(or_mask.sum())}")
    print("  rule = " + "  OR  ".join(or_terms))
    sel = F_test.loc[or_mask]
    print(f"  dates: {sel.index.strftime('%Y-%m-%d').tolist()}")
    print()

    if history:
        chain, mask, hits = history[-1]
        print(f"Test days selected by the length-{len(chain)} AND chain:")
        sel = F_test.loc[mask]
        print(sel.index.strftime("%Y-%m-%d").tolist())

    # Persist the predicates table for downstream use.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nWrote per-factor predicates -> {args.out}")


if __name__ == "__main__":
    main()
