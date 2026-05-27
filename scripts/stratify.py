#!/usr/bin/env python3

"""
Site/day stratified sampling utility for prediction CSV files.

The script groups detections by site and day, then samples up to N items per group. 
Sampling weights are highest near score 0 and decay with larger positive scores using a 
single Gaussian-width parameter (sigma) estimated from the non-negative score distribution. 
Negative scores are excluded from selection (weight 0), so only scores >= 0 are 
eligible for weighted sampling. 

A second optional uniform random sampling step can be applied to the combined site/day samples to cap the total output size.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _half_gaussian_weights(values: pd.Series, sigma: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    # Negative scores are intentionally excluded from weighted selection.
    eligible = np.isfinite(numeric) & (numeric >= 0)
    safe_values = numeric.where(eligible, other=np.inf)
    weights = np.exp(-(safe_values**2) / (2.0 * sigma**2))
    weights = pd.Series(weights, index=values.index)
    weights = weights.where(eligible, other=0.0)
    return weights


def sample_by_site_and_day(
    df: pd.DataFrame,
    n: int,
    random_seed: int,
    score_column: str = "score",
) -> pd.DataFrame:
    if n < 1:
        raise ValueError("N must be >= 1")

    if "site_id" not in df.columns:
        raise ValueError("Input CSV must contain column: site_id")
    if "start_datetime" not in df.columns:
        raise ValueError("Input CSV must contain column: start_datetime")
    if score_column not in df.columns:
        raise ValueError(f"Input CSV must contain column: {score_column}")

    parsed_datetime = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    if parsed_datetime.isna().all():
        raise ValueError("Could not parse any values in start_datetime")

    working = df.copy()
    working["_date"] = parsed_datetime.dt.date

    valid_scores = pd.to_numeric(working[score_column], errors="coerce")
    valid_scores = valid_scores[np.isfinite(valid_scores) & (valid_scores >= 0)]
    sigma = float(valid_scores.std(ddof=0)) if len(valid_scores) else 0.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0

    sampled_groups = []
    grouped = working.groupby(["site_id", "_date"], dropna=False, sort=False)

    for _, group in grouped:
        weights = _half_gaussian_weights(group[score_column], sigma)
        eligible_group = group[weights > 0].copy()

        if eligible_group.empty:
            continue

        k = min(n, len(eligible_group))
        eligible_weights = weights.loc[eligible_group.index]

        if not np.isfinite(eligible_weights).all() or float(eligible_weights.sum()) <= 0.0:
            sampled = eligible_group.sample(n=k, replace=False, random_state=random_seed)
        else:
            sampled = eligible_group.sample(
                n=k,
                replace=False,
                weights=eligible_weights,
                random_state=random_seed,
            )

        sampled_groups.append(sampled)

    if not sampled_groups:
        return working.drop(columns=["_date"])

    result = pd.concat(sampled_groups, axis=0).sort_values(
        by=["site_id", "_date", "start_datetime"], kind="stable"
    )
    return result.drop(columns=["_date"]).reset_index(drop=True)


def sample_max_items_uniform(df: pd.DataFrame, max_items: int, random_seed: int) -> pd.DataFrame:
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    if len(df) <= max_items:
        return df

    return df.sample(n=max_items, replace=False, random_state=random_seed).reset_index(drop=True)


def print_score_distribution(df: pd.DataFrame, score_column: str = "score", bins: int = 10) -> None:
    if score_column not in df.columns:
        print(f"Score distribution: column '{score_column}' not found.")
        return

    scores = pd.to_numeric(df[score_column], errors="coerce")
    scores = scores[np.isfinite(scores)]

    if scores.empty:
        print("Score distribution: no valid numeric scores in final sample.")
        return

    quantiles = scores.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    print("\nFinal sample score summary")
    print(f"  count: {len(scores)}")
    print(f"  min:   {scores.min():.4f}")
    print(f"  p10:   {quantiles.loc[0.1]:.4f}")
    print(f"  p25:   {quantiles.loc[0.25]:.4f}")
    print(f"  p50:   {quantiles.loc[0.5]:.4f}")
    print(f"  p75:   {quantiles.loc[0.75]:.4f}")
    print(f"  p90:   {quantiles.loc[0.9]:.4f}")
    print(f"  max:   {scores.max():.4f}")

    counts, edges = np.histogram(scores, bins=bins)
    max_count = int(counts.max()) if len(counts) else 0
    bar_width = 30

    print(f"\nScore histogram ({bins} bins)")
    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int(round((count / max_count) * bar_width)) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{left:8.4f}, {right:8.4f}) | {count:4d} | {bar}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample up to N records per (site_id, day), weighted toward score=0 "
            "with a half-Gaussian distribution."
        )
    )
    parser.add_argument("csv_path", help="Path to the input CSV")
    parser.add_argument(
        "-n",
        "--num-per-site-day",
        type=int,
        default=1,
        help="Maximum records to sample per site/day group (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output CSV path (default: <input_stem>_stratified_n<N>.csv in same directory)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=500,
        help="Uniformly sample this many rows after site/day sampling (default: 500)",
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
        else csv_path.with_name(f"{csv_path.stem}_stratified_n{args.num_per_site_day}.csv")
    )

    df = pd.read_csv(csv_path)
    sampled = sample_by_site_and_day(
        df=df,
        n=args.num_per_site_day,
        random_seed=args.seed,
    )
    sampled = sample_max_items_uniform(
        df=sampled,
        max_items=args.max_items,
        random_seed=args.seed,
    )
    sampled.to_csv(output_path, index=False)

    print(
        f"Wrote {len(sampled)} rows sampled from {len(df)} input rows to: {output_path}"
    )
    print_score_distribution(sampled)


if __name__ == "__main__":
    main()
