#!/usr/bin/env python3

"""Sample rows across a score distribution using greedy clustering.

The point of this is to obtain a representative sample of rows that covers the full range of scores, 
for example to search to find a good threshold. 

A simple linear approach (a set of equal-distance target scores) fails if there are outliers leaving most of the target scores in sparse regions, possibly empty.
A simple quantile-based approach (selecting one row from each of n quantiles) will create a distance between scores too great at low density regions. 

Somewhere between is what we are aiming for, which is achieved by mean-based
clustering.

The script repeatedly divides the cluster with the largest priority at that
cluster's mean. Priority is ``variance * item_count ** density_power``.
At a density power of 0, only score variance matters. At 1, the priority is
the within-cluster sum of squared error. Values between them balance score
coverage with observation density without requiring fixed-width bins.

For two or more requested rows, the minimum and maximum are reserved first.
The remaining rows are clustered into ``N - 2`` groups and the median row of
each group is selected. With one requested row, the overall median is used.
When a group column is supplied, this process runs independently for every
group, so the requested count is per group.

Rows with missing, non-numeric, or non-finite scores are excluded. Score
bounds are inclusive, and the original CSV columns are preserved.
"""

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreCluster:
    positions: np.ndarray
    sse: float
    minimum: float
    maximum: float


def _validate_density_power(density_power: float) -> None:
    if not np.isfinite(density_power) or not 0.0 <= density_power <= 1.0:
        raise ValueError("density_power must be finite and between 0 and 1")


def _cluster(positions: np.ndarray, scores: np.ndarray) -> ScoreCluster:
    values = scores[positions]
    mean = float(values.mean())
    sse = float(np.square(values - mean).sum())
    return ScoreCluster(
        positions=positions,
        sse=sse,
        minimum=float(values[0]),
        maximum=float(values[-1]),
    )


def _split_cluster(
    cluster: ScoreCluster, scores: np.ndarray
) -> tuple[ScoreCluster, ScoreCluster] | None:
    if len(cluster.positions) < 2:
        return None

    values = scores[cluster.positions]
    mean = float(values.mean())
    left_mask = values < mean

    if left_mask.any() and (~left_mask).any():
        left_positions = cluster.positions[left_mask]
        right_positions = cluster.positions[~left_mask]
    else:
        # All scores are equal, so split by stable row position to reach the
        # requested sample size without introducing replacement.
        midpoint = len(cluster.positions) // 2
        left_positions = cluster.positions[:midpoint]
        right_positions = cluster.positions[midpoint:]

    return (
        _cluster(left_positions, scores),
        _cluster(right_positions, scores),
    )


def _split_priority(cluster: ScoreCluster, density_power: float) -> float:
    variance = cluster.sse / len(cluster.positions)
    return variance * len(cluster.positions) ** density_power


def _cluster_by_score(
    scores: np.ndarray,
    num_clusters: int,
    density_power: float = 1.0,
) -> list[ScoreCluster]:
    if num_clusters < 1:
        raise ValueError("num_examples must be >= 1")
    if len(scores) == 0:
        raise ValueError("No rows remain after score filtering")
    _validate_density_power(density_power)

    num_clusters = min(num_clusters, len(scores))
    clusters = [_cluster(np.arange(len(scores)), scores)]

    while len(clusters) < num_clusters:
        splittable = [
            index
            for index, cluster in enumerate(clusters)
            if len(cluster.positions) > 1
        ]
        if not splittable:
            break

        # Prefer the cluster with the largest variance-and-density priority.
        # A true score split is preferred over splitting equal-score rows.
        split_index = max(
            splittable,
            key=lambda index: (
                _split_priority(clusters[index], density_power) > 0.0,
                round(_split_priority(clusters[index], density_power), 15),
                len(clusters[index].positions),
                -clusters[index].minimum,
            ),
        )
        split = _split_cluster(clusters[split_index], scores)
        if split is None:
            break

        left, right = split
        clusters[split_index : split_index + 1] = [left, right]

    return sorted(clusters, key=lambda cluster: (cluster.minimum, cluster.maximum))


def _representative_position(cluster: ScoreCluster) -> int:
    return int(cluster.positions[len(cluster.positions) // 2])


def _numeric_scores(df: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column not in df.columns:
        raise ValueError(f"Input CSV must contain score column '{score_column}'")

    return pd.to_numeric(df[score_column], errors="coerce").astype(float)


def sample_accross_scores(
    df: pd.DataFrame,
    num_examples: int,
    score_column: str = "score",
    from_score: float | None = None,
    to_score: float | None = None,
    density_power: float = 1.0,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Return N examples per group using greedy score clustering."""
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    _validate_density_power(density_power)
    if from_score is not None and not np.isfinite(from_score):
        raise ValueError("from_score must be finite")
    if to_score is not None and not np.isfinite(to_score):
        raise ValueError("to_score must be finite")
    if from_score is not None and to_score is not None and from_score > to_score:
        raise ValueError("from_score must be less than or equal to to_score")

    scores = _numeric_scores(df, score_column)
    finite_scores = scores[np.isfinite(scores)]

    lower_bound = finite_scores.min() if from_score is None else from_score
    upper_bound = finite_scores.max() if to_score is None else to_score
    if pd.isna(lower_bound) or pd.isna(upper_bound):
        raise ValueError(f"Column '{score_column}' has no finite numeric scores")
    if group_column is not None and group_column not in df.columns:
        raise ValueError(f"Input CSV must contain group column '{group_column}'")

    eligible_mask = np.isfinite(scores) & scores.ge(lower_bound) & scores.le(upper_bound)
    eligible = df.loc[eligible_mask].copy()
    if eligible.empty:
        raise ValueError(
            f"No rows have '{score_column}' between {lower_bound} and {upper_bound}"
        )

    if group_column is None:
        sampled_groups = [_sample_eligible_rows(
            eligible,
            scores.loc[eligible_mask],
            num_examples,
            density_power,
        )]
    else:
        sampled_groups = []
        for _, group in eligible.groupby(group_column, dropna=False, sort=False):
            sampled_groups.append(
                _sample_eligible_rows(
                    group,
                    scores.loc[group.index],
                    num_examples,
                    density_power,
                )
            )

    return pd.concat(sampled_groups, ignore_index=True)


def _sample_eligible_rows(
    eligible: pd.DataFrame,
    eligible_scores: pd.Series,
    num_examples: int,
    density_power: float,
) -> pd.DataFrame:
    """Sample one score-filtered group."""
    order = np.argsort(eligible_scores.to_numpy(dtype=float), kind="stable")
    eligible = eligible.iloc[order].reset_index(drop=True)
    sorted_scores = eligible_scores.iloc[order].to_numpy(dtype=float)

    requested = min(num_examples, len(eligible))
    if requested == 1:
        selected_positions = [len(eligible) // 2]
    elif requested == 2:
        selected_positions = [0, len(eligible) - 1]
    elif requested == len(eligible):
        selected_positions = list(range(len(eligible)))
    else:
        interior_scores = sorted_scores[1:-1]
        clusters = _cluster_by_score(
            interior_scores,
            requested - 2,
            density_power=density_power,
        )
        interior_positions = [
            _representative_position(cluster) + 1 for cluster in clusters
        ]
        selected_positions = [
            0,
            *interior_positions,
            len(eligible) - 1,
        ]

    return eligible.iloc[selected_positions].reset_index(drop=True)


def write_score_histogram(
    df: pd.DataFrame,
    sampled: pd.DataFrame,
    output_path: Path,
    score_column: str = "score",
    from_score: float | None = None,
    to_score: float | None = None,
    bins: int = 30,
) -> None:
    """Write an SVG histogram with a vertical marker for every sampled row."""
    if bins < 1:
        raise ValueError("histogram bins must be >= 1")

    scores = _numeric_scores(df, score_column)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.empty:
        raise ValueError(f"Column '{score_column}' has no finite numeric scores")

    lower_bound = finite_scores.min() if from_score is None else from_score
    upper_bound = finite_scores.max() if to_score is None else to_score
    eligible_scores = finite_scores[
        finite_scores.ge(lower_bound) & finite_scores.le(upper_bound)
    ].to_numpy(dtype=float)
    if len(eligible_scores) == 0:
        raise ValueError("No scores are available for the histogram")

    sampled_scores = pd.to_numeric(
        sampled[score_column], errors="coerce"
    ).to_numpy(dtype=float)
    sampled_scores = sampled_scores[np.isfinite(sampled_scores)]

    histogram_min = float(lower_bound)
    histogram_max = float(upper_bound)
    if histogram_min == histogram_max:
        padding = max(abs(histogram_min) * 0.05, 0.5)
        histogram_min -= padding
        histogram_max += padding

    counts, edges = np.histogram(
        eligible_scores,
        bins=bins,
        range=(histogram_min, histogram_max),
    )

    width = 1000
    height = 500
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 65
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_bottom = margin_top + plot_height
    max_count = max(int(counts.max()), 1)

    def x_position(score: float) -> float:
        proportion = (score - histogram_min) / (histogram_max - histogram_min)
        return margin_left + proportion * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>"
        "text { font-family: sans-serif; fill: #263238; }"
        ".axis-label { font-size: 13px; }"
        ".tick-label { font-size: 11px; fill: #546e7a; }"
        ".sample-label { font-size: 11px; fill: #b71c1c; }"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
        (
            f'<text x="{width / 2}" y="28" text-anchor="middle" '
            f'font-size="18">Score distribution and sampled rows</text>'
        ),
    ]

    for index, count in enumerate(counts):
        bar_left = x_position(float(edges[index]))
        bar_right = x_position(float(edges[index + 1]))
        bar_height = (int(count) / max_count) * plot_height
        svg.append(
            f'<rect x="{bar_left:.2f}" y="{plot_bottom - bar_height:.2f}" '
            f'width="{max(bar_right - bar_left - 1, 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#4c78a8" opacity="0.8"/>'
        )

    svg.extend(
        [
            f'<line x1="{margin_left}" y1="{plot_bottom}" '
            f'x2="{width - margin_right}" y2="{plot_bottom}" stroke="#263238"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{plot_bottom}" stroke="#263238"/>',
        ]
    )

    for tick in np.linspace(histogram_min, histogram_max, 6):
        x = x_position(float(tick))
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{plot_bottom}" '
                f'x2="{x:.2f}" y2="{plot_bottom + 5}" stroke="#263238"/>',
                f'<text class="tick-label" x="{x:.2f}" y="{plot_bottom + 20}" '
                f'text-anchor="middle">{tick:.3g}</text>',
            ]
        )

    for count_tick in np.linspace(0, max_count, 5, dtype=int):
        y = plot_bottom - (count_tick / max_count) * plot_height
        svg.extend(
            [
                f'<line x1="{margin_left - 5}" y1="{y:.2f}" '
                f'x2="{margin_left}" y2="{y:.2f}" stroke="#263238"/>',
                f'<text class="tick-label" x="{margin_left - 10}" y="{y + 4:.2f}" '
                f'text-anchor="end">{count_tick}</text>',
            ]
        )

    for score in sampled_scores:
        x = x_position(float(score))
        svg.append(
            f'<line class="sample-marker" x1="{x:.2f}" y1="{margin_top}" '
            f'x2="{x:.2f}" y2="{plot_bottom}" stroke="#c62828" '
            'stroke-width="2" opacity="0.8"/>'
        )

    svg.extend(
        [
            (
                f'<text class="axis-label" x="{margin_left + plot_width / 2}" '
                f'y="{height - 12}" text-anchor="middle">'
                f'{escape(score_column)}</text>'
            ),
            (
                f'<text class="sample-label" x="{width - margin_right}" y="47" '
                f'text-anchor="end">Red lines: {len(sampled_scores)} sampled rows</text>'
            ),
            "</svg>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample rows across a score distribution using greedy clustering. "
            "Clusters are selected by variance times cluster size raised to "
            "the configurable density power."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV")
    parser.add_argument(
        "num_examples",
        type=int,
        help="Number of rows to sample",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output CSV path (default: "
            "<input_stem>_sampled_accross_scores_n<N>.csv)"
        ),
    )
    parser.add_argument(
        "--score-column",
        default="score",
        help="Name of the numeric score column (default: score)",
    )
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument(
        "--group-by",
        default="__auto__",
        metavar="COLUMN",
        help=(
            "Sample independently for each value in this column "
            "(default: label when present)"
        ),
    )
    grouping.add_argument(
        "--no-grouping",
        action="store_const",
        const=None,
        dest="group_by",
        help="Sample once across the entire CSV",
    )
    parser.add_argument(
        "--from-score",
        type=float,
        help="Inclusive lower score bound (default: minimum valid score)",
    )
    parser.add_argument(
        "--to-score",
        type=float,
        help="Inclusive upper score bound (default: maximum valid score)",
    )
    parser.add_argument(
        "--density-power",
        "--alpha",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to cluster size (0-1): 0 favors score variance, "
            "1 favors within-cluster sum of squared error (default: 1)"
        ),
    )
    histogram_group = parser.add_mutually_exclusive_group()
    histogram_group.add_argument(
        "--histogram",
        type=Path,
        help="Histogram SVG path (default: beside the sampled CSV)",
    )
    histogram_group.add_argument(
        "--no-histogram",
        action="store_true",
        help="Do not create the score histogram",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    output_path = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_sampled_accross_scores_n{args.num_examples}.csv"
    )

    df = pd.read_csv(args.input_csv)
    if args.group_by == "__auto__":
        group_column = "label" if "label" in df.columns else None
    else:
        group_column = args.group_by
        if group_column is not None and group_column not in df.columns:
            raise ValueError(
                f"Input CSV must contain group column '{group_column}'"
            )
    sampled = sample_accross_scores(
        df,
        num_examples=args.num_examples,
        score_column=args.score_column,
        from_score=args.from_score,
        to_score=args.to_score,
        density_power=args.density_power,
        group_column=group_column,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)

    if not args.no_histogram:
        histogram_path = args.histogram or output_path.with_name(
            f"{output_path.stem}_histogram.svg"
        )
        write_score_histogram(
            df,
            sampled,
            histogram_path,
            score_column=args.score_column,
            from_score=args.from_score,
            to_score=args.to_score,
            bins=args.histogram_bins,
        )
        print(f"Wrote score histogram to: {histogram_path}")

    scores = _numeric_scores(df, args.score_column)
    invalid_count = int((~np.isfinite(scores)).sum())
    valid_scores = scores[np.isfinite(scores)]
    lower_bound = valid_scores.min() if args.from_score is None else args.from_score
    upper_bound = valid_scores.max() if args.to_score is None else args.to_score
    eligible_count = int(
        (np.isfinite(scores) & scores.ge(lower_bound) & scores.le(upper_bound)).sum()
    )
    excluded_count = len(df) - invalid_count - eligible_count

    print(f"Wrote {len(sampled)} rows from {len(df)} input rows to: {output_path}")
    if invalid_count:
        print(
            f"Skipped {invalid_count} rows with non-numeric or non-finite scores"
        )
    if excluded_count:
        print(f"Excluded {excluded_count} rows outside the requested score range")
    expected_count = (
        args.num_examples
        * df.loc[
            np.isfinite(scores)
            & scores.ge(lower_bound)
            & scores.le(upper_bound),
            group_column,
        ].nunique(dropna=False)
        if group_column is not None
        else args.num_examples
    )
    if len(sampled) < expected_count:
        print(
            f"Requested {args.num_examples} rows per "
            f"{'group' if group_column else 'sample'}, but only "
            f"{len(sampled)} rows were available after filtering"
        )


if __name__ == "__main__":
    main()
