#!/usr/bin/env python3

"""
Score-binned sampling utility for prediction CSV files.

This script creates a representative subset by balancing selections across score
intervals rather than using plain random sampling. It is useful when review
work should cover the full score range (low, medium, and high confidence)
instead of over-representing dense score regions. Negatives are included. 
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_inputs(df: pd.DataFrame, score_column: str, n: int) -> pd.Series:
    if n < 1:
        raise ValueError("n must be >= 1")
    if score_column not in df.columns:
        raise ValueError(f"Input CSV must contain column: {score_column}")

    scores = pd.to_numeric(df[score_column], errors="coerce")
    finite_mask = np.isfinite(scores.to_numpy(dtype=float))
    if not finite_mask.any():
        raise ValueError(f"Column '{score_column}' has no valid numeric values")

    return scores


def _cap_n_to_available(scores: pd.Series, n: int) -> int:
    valid_n = int(np.isfinite(scores.to_numpy(dtype=float)).sum())
    return min(n, valid_n)


def select_uniform_by_bins(
    df: pd.DataFrame,
    n: int,
    score_column: str = "score",
    bins: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select n rows by balancing picks across score bins.

    This is efficient and introduces randomness while keeping score coverage
    more uniform than plain random sampling.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")

    scores = _validate_inputs(df, score_column, n)

    working = df.copy()
    working["_score"] = scores
    working = working[np.isfinite(working["_score"])].copy()
    if working.empty:
        raise ValueError("No valid rows remain after filtering non-numeric scores")

    n = _cap_n_to_available(working["_score"], n)
    if n == len(working):
        return working.drop(columns=["_score"]).reset_index(drop=True)

    lo = float(working["_score"].min())
    hi = float(working["_score"].max())
    if lo == hi:
        return working.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)

    edges = np.linspace(lo, hi, bins + 1)
    working["_bin"] = pd.cut(
        working["_score"],
        bins=edges,
        include_lowest=True,
        labels=False,
        duplicates="drop",
    )

    grouped = {
        int(bin_id): group.index.to_numpy(dtype=int)
        for bin_id, group in working.groupby("_bin", dropna=True, sort=True)
        if len(group) > 0
    }
    if not grouped:
        return working.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    shuffled_by_bin = {}
    for bin_id, indices in grouped.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        shuffled_by_bin[bin_id] = shuffled

    pointers = {bin_id: 0 for bin_id in shuffled_by_bin}
    available_bins = sorted(shuffled_by_bin.keys())
    picked = []

    while len(picked) < n and available_bins:
        next_available_bins = []
        for bin_id in available_bins:
            if len(picked) >= n:
                break

            pos = pointers[bin_id]
            arr = shuffled_by_bin[bin_id]
            if pos < len(arr):
                picked.append(int(arr[pos]))
                pointers[bin_id] = pos + 1
            if pointers[bin_id] < len(arr):
                next_available_bins.append(bin_id)

        available_bins = next_available_bins

    selected = working.loc[picked].drop(columns=["_score", "_bin"], errors="ignore")
    return selected.reset_index(drop=True)


def print_score_summary(df: pd.DataFrame, score_column: str = "score", bins: int = 10) -> None:
    scores = pd.to_numeric(df[score_column], errors="coerce")
    scores = scores[np.isfinite(scores)]
    if scores.empty:
        print("No valid numeric scores in selected sample.")
        return

    print(f"count={len(scores)}, min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")
    counts, edges = np.histogram(scores, bins=max(1, bins))
    print("Histogram:")
    for i, count in enumerate(counts):
        print(f"  [{edges[i]:.4f}, {edges[i + 1]:.4f}]: {int(count)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a score-balanced subset from a CSV using a fast binned sampler."
        )
    )
    parser.add_argument("csv_path", help="Path to input CSV")
    parser.add_argument("n", type=int, help="Desired number of output rows")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <input_stem>_score_uniform_<mode>_n<N>.csv)",
    )
    parser.add_argument(
        "--score-column",
        default="score",
        help="Name of score column (default: score)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of score bins (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--summary-bins",
        type=int,
        default=10,
        help="Histogram bins in terminal summary (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    output_path = (
        Path(args.output)
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_score_uniform_binned_n{args.n}.csv")
    )

    df = pd.read_csv(csv_path)
    sampled = select_uniform_by_bins(
        df=df,
        n=args.n,
        score_column=args.score_column,
        bins=args.bins,
        seed=args.seed,
    )

    sampled.to_csv(output_path, index=False)

    print(f"Wrote {len(sampled)} rows from {len(df)} input rows to: {output_path}")
    print_score_summary(sampled, score_column=args.score_column, bins=args.summary_bins)


if __name__ == "__main__":
    main()
