from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


MASTERED_SCALE_DEFAULT = 5.0
ABILITY_SCALE_DEFAULT = 100.0
ABILITY_BUCKETS = (
    (40.0, "low_ability", 0),
    (80.0, "medium_ability", 1),
    (float("inf"), "high_ability", 2),
)

MASTERY_BUCKETS = (
    (0, "not_started_concepts", 0),
    (1, "just_started_concepts", 1),
    (2, "struggle_concepts", 2),
    (3, "good_concepts", 3),
    (4, "strong_concepts", 4),
    (5, "strong_concepts", 4),
)
CONCEPT_BUCKET_RANKS = {
    bucket_key: rank
    for _, bucket_key, rank in MASTERY_BUCKETS
}
ABILITY_BUCKET_RANKS = {
    bucket_key: rank
    for _, bucket_key, rank in ABILITY_BUCKETS
}
UNIQUE_MASTERY_BUCKET_KEYS = tuple(dict.fromkeys(bucket_key for _, bucket_key, _ in MASTERY_BUCKETS))
UNIQUE_ABILITY_BUCKET_KEYS = tuple(dict.fromkeys(bucket_key for _, bucket_key, _ in ABILITY_BUCKETS))


def bucket_items(bucket: Mapping[str, Any]) -> list[str]:
    return list(bucket.get("items", []))


def bucket_value(bucket: Mapping[str, Any]) -> Optional[float]:
    value = bucket.get("value")
    if value is None:
        return None
    return float(value)


def get_mastery_bucket_rank(mastery_status: float) -> int:
    clipped_value = int(np.clip(mastery_status, 0, 5))
    for threshold, bucket_key, rank in MASTERY_BUCKETS:
        if clipped_value <= threshold:
            return rank
    return MASTERY_BUCKETS[-1][2]


def get_mastery_bucket_key(median_mastery_rank: float) -> str:
    clipped_rank = float(np.clip(median_mastery_rank, 0, max(CONCEPT_BUCKET_RANKS.values())))
    ranked_buckets = sorted(
        ((rank, bucket_key) for bucket_key, rank in CONCEPT_BUCKET_RANKS.items()),
        key=lambda item: item[0],
    )
    return min(
        ranked_buckets,
        key=lambda item: (abs(item[0] - clipped_rank), item[0]),
    )[1]


def get_student_ability_bucket_key(student_ability: Optional[float]) -> str:
    if student_ability is None:
        return ABILITY_BUCKETS[1][1]
    for threshold, bucket_key, _ in ABILITY_BUCKETS:
        if student_ability <= threshold:
            return bucket_key
    return ABILITY_BUCKETS[-1][1]
