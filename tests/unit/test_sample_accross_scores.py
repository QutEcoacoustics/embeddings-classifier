from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from sample_accross_scores import sample_accross_scores, write_score_histogram


def test_samples_both_dense_and_sparse_score_regions():
    scores = (
        [0.0] * 20
        + [0.1] * 12
        + [0.2] * 6
        + [0.3] * 3
        + [0.4] * 2
        + [1.0]
    )
    df = pd.DataFrame({"row_id": range(len(scores)), "score": scores})

    sampled = sample_accross_scores(df, num_examples=6)

    assert len(sampled) == 6
    assert sampled["score"].iloc[0] == pytest.approx(0.0)
    assert sampled["score"].iloc[-1] == pytest.approx(1.0)
    assert sampled["score"].nunique() >= 5
    assert set(sampled["score"]) >= {0.0, 0.1, 0.2, 0.3, 1.0}


def test_selects_exact_number_of_rows_when_scores_are_equal():
    df = pd.DataFrame({"row_id": range(10), "score": [0.5] * 10})

    sampled = sample_accross_scores(df, num_examples=6)

    assert len(sampled) == 6
    assert sampled["row_id"].is_unique
    assert sampled["score"].eq(0.5).all()


def test_single_sample_uses_middle_score():
    df = pd.DataFrame({"row_id": range(5), "score": [0.0, 0.1, 0.3, 0.6, 1.0]})

    sampled = sample_accross_scores(df, num_examples=1)

    assert sampled["score"].tolist() == [0.3]


def test_two_samples_use_only_endpoints():
    df = pd.DataFrame({"score": [0.0, 0.1, 0.3, 0.6, 1.0]})

    sampled = sample_accross_scores(df, num_examples=2)

    assert sampled["score"].tolist() == [0.0, 1.0]


def test_samples_independently_per_group():
    df = pd.DataFrame(
        {
            "label": ["frog-a"] * 5 + ["frog-b"] * 5,
            "score": [0.0, 0.1, 0.2, 0.8, 1.0] * 2,
        }
    )

    sampled = sample_accross_scores(
        df,
        num_examples=3,
        group_column="label",
    )

    assert sampled.groupby("label").size().to_dict() == {
        "frog-a": 3,
        "frog-b": 3,
    }
    assert sampled.groupby("label")["score"].min().to_dict() == {
        "frog-a": 0.0,
        "frog-b": 0.0,
    }
    assert sampled.groupby("label")["score"].max().to_dict() == {
        "frog-a": 1.0,
        "frog-b": 1.0,
    }


def test_grouping_returns_available_rows_for_small_group():
    df = pd.DataFrame(
        {
            "label": ["frog-a"] * 5 + ["frog-b"],
            "score": [0.0, 0.1, 0.2, 0.8, 1.0, 0.4],
        }
    )

    sampled = sample_accross_scores(
        df,
        num_examples=3,
        group_column="label",
    )

    assert sampled.groupby("label").size().to_dict() == {
        "frog-a": 3,
        "frog-b": 1,
    }


def test_rejects_missing_group_column():
    with pytest.raises(ValueError, match="group column 'class_name'"):
        sample_accross_scores(
            pd.DataFrame({"score": [0.1, 0.2]}),
            1,
            group_column="class_name",
        )


def test_reserves_endpoints_and_uses_cluster_medians_in_between():
    df = pd.DataFrame(
        {
            "row_id": range(11),
            "score": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )

    sampled = sample_accross_scores(df, num_examples=4)

    assert sampled["score"].iloc[[0, -1]].tolist() == [0.0, 1.0]
    assert sampled["row_id"].is_unique
    assert len(sampled) == 4


def test_density_power_changes_cluster_priority():
    df = pd.DataFrame(
        {
            "score": [
                0,
                0.05,
                0.05,
                0.1,
                0.2,
                0.2,
                0.2,
                0.3,
                0.3,
                0.4,
                0.4,
                0.6,
                0.8,
                0.8,
                0.8,
                1,
                1,
                2,
                2,
                2,
                2,
                5,
                10,
                10,
            ]
        }
    )

    variance_focused = sample_accross_scores(
        df, num_examples=8, density_power=0.0
    )
    sse_focused = sample_accross_scores(
        df, num_examples=8, density_power=1.0
    )

    assert variance_focused["score"].to_list() != sse_focused["score"].to_list()


def test_filters_with_inclusive_score_bounds_and_custom_column():
    df = pd.DataFrame(
        {
            "row_id": list("abcdef"),
            "confidence": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    sampled = sample_accross_scores(
        df,
        num_examples=3,
        score_column="confidence",
        from_score=0.1,
        to_score=0.4,
    )

    assert sampled["confidence"].iloc[[0, -1]].tolist() == [0.1, 0.4]
    assert sampled["confidence"].ge(0.1).all()
    assert sampled["confidence"].le(0.4).all()


def test_preserves_columns_used_internally():
    df = pd.DataFrame(
        {
            "_sample_score": ["keep-a", "keep-b", "keep-c"],
            "score": [0.0, 0.5, 1.0],
        }
    )

    sampled = sample_accross_scores(df, num_examples=2)

    assert sampled["_sample_score"].tolist() == ["keep-a", "keep-c"]


def test_ignores_invalid_scores():
    df = pd.DataFrame(
        {
            "row_id": list("abcdef"),
            "score": [0.0, np.nan, np.inf, "not-a-score", 0.5, 1.0],
        }
    )

    sampled = sample_accross_scores(df, num_examples=10)

    assert sampled["row_id"].tolist() == ["a", "e", "f"]
    assert sampled["score"].astype(float).tolist() == [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    ("num_examples", "from_score", "to_score", "message"),
    [
        (0, None, None, "num_examples"),
        (1, 2.0, 1.0, "from_score"),
        (1, np.nan, None, "from_score"),
        (1, 3.0, 4.0, "No rows"),
    ],
)
def test_rejects_invalid_sampling_requests(
    num_examples, from_score, to_score, message
):
    df = pd.DataFrame({"score": [0.0, 1.0, 2.0]})

    with pytest.raises(ValueError, match=message):
        sample_accross_scores(
            df,
            num_examples=num_examples,
            from_score=from_score,
            to_score=to_score,
        )


def test_requires_score_column():
    with pytest.raises(ValueError, match="score column 'score'"):
        sample_accross_scores(pd.DataFrame({"confidence": [0.1, 0.2]}), 1)


@pytest.mark.parametrize("density_power", [-0.1, 1.1, np.nan, np.inf])
def test_rejects_invalid_density_power(density_power):
    with pytest.raises(ValueError, match="density_power"):
        sample_accross_scores(
            pd.DataFrame({"score": [0.1, 0.2]}),
            1,
            density_power=density_power,
        )


def test_writes_histogram_with_one_marker_per_sample(tmp_path):
    df = pd.DataFrame({"score": [0.0, 0.1, 0.2, 0.5, 1.0, np.nan]})
    sampled = sample_accross_scores(df, 4)
    histogram_path = tmp_path / "scores.svg"

    write_score_histogram(
        df,
        sampled,
        histogram_path,
        bins=5,
    )

    histogram = histogram_path.read_text(encoding="utf-8")
    assert histogram_path.exists()
    assert histogram.count('class="sample-marker"') == len(sampled)
    assert "<svg" in histogram
