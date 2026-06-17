from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from numpy import ndarray
from sklearn.base import ClusterMixin

from hecof_llm_cluster import (
    HecofLLMClusterOptimizer,
    SummarizedProfileCluster,
)
from bucket_utils import (
    ABILITY_BUCKETS,
    ABILITY_BUCKET_RANKS,
    CONCEPT_BUCKET_RANKS,
    MASTERY_BUCKETS,
    UNIQUE_ABILITY_BUCKET_KEYS,
    UNIQUE_MASTERY_BUCKET_KEYS,
)

try:
    from util import run_parallel
except ImportError:
    from reference.util import run_parallel


CONCEPT_BUCKET_NAMES = UNIQUE_MASTERY_BUCKET_KEYS
ABILITY_BUCKET_NAMES = UNIQUE_ABILITY_BUCKET_KEYS


@dataclass
class ConceptBucketEmbeddings:
    not_started_concepts: Optional[ndarray]
    just_started_concepts: Optional[ndarray]
    struggle_concepts: Optional[ndarray]
    good_concepts: Optional[ndarray]
    strong_concepts: Optional[ndarray]

    def get(self, bucket_name: str) -> Optional[ndarray]:
        return getattr(self, bucket_name)


@dataclass
class AbilityBucketEmbeddings:
    low_ability: Optional[ndarray]
    medium_ability: Optional[ndarray]
    high_ability: Optional[ndarray]

    def get(self, bucket_name: str) -> Optional[ndarray]:
        return getattr(self, bucket_name)


@dataclass
class ClusterBucketRepresentation:
    concept: ConceptBucketEmbeddings
    ability: AbilityBucketEmbeddings

    def get(self, bucket_name: str) -> Optional[ndarray]:
        if hasattr(self.concept, bucket_name):
            return self.concept.get(bucket_name)
        return self.ability.get(bucket_name)


class HecofBucketEmbeddingClusterOptimizer(HecofLLMClusterOptimizer):
    """
    Variant of `HecofLLMClusterOptimizer` that compares clusters using concept bags
    instead of embedding a cluster-level summary string.

    Key differences from `HecofLLMClusterOptimizer`:
    - each concept name is embedded once and cached
    - each cluster is represented by five averaged embeddings, one for each
      mastery bucket from `Not started` to `Strong`
    - merge/scoring distance combines:
      - the current embedding-based bucket distance
      - a cross-bucket item distance based on ordered bucket ranks
    """

    def __init__(
        self,
        cluster_models: Sequence[ClusterMixin | Callable[[ndarray], ClusterMixin]],
        recluster_model: Optional[ClusterMixin] = None,
        concept_embedding_fn: Optional[Callable[[str], ndarray]] = None,
        summary_embedding_fn: Optional[Callable[[str], ndarray]] = None,
        cross_bucket_distance_weight: float = 0.5,
        **kwargs,
    ):
        selected_embedding_fn = concept_embedding_fn or summary_embedding_fn
        super().__init__(
            cluster_models=cluster_models,
            recluster_model=recluster_model,
            summary_embedding_fn=selected_embedding_fn,
            **kwargs,
        )
        self._cross_bucket_distance_weight = float(np.clip(cross_bucket_distance_weight, 0.0, 1.0))
        self._bucket_embedding_fn = self._summary_embedding_fn
        self._bucket_item_embedding_cache: dict[str, ndarray] = {}
        self._active_feature_columns: list[str] = []

    def _summarize_clusters(
        self,
        clusters: Sequence[SummarizedProfileCluster],
        **kwargs,
    ) -> list[SummarizedProfileCluster]:
        feature_columns = kwargs.get("feature_columns")
        if feature_columns:
            self._active_feature_columns = list(feature_columns)
            self._ensure_bucket_item_embeddings(self._active_feature_columns)
        return super()._summarize_clusters(clusters, **kwargs)

    def _ensure_bucket_item_embeddings(self, feature_columns: Sequence[str]) -> None:
        bucket_items = sorted({
            feature_name.rsplit("_", 1)[0]
            for feature_name in feature_columns
            if feature_name.endswith("_masteryStatus")
        })
        if "cross_concept_avg_ability" in feature_columns:
            bucket_items.append("cross_concept_avg_ability")
        missing_items = [
            item for item in bucket_items
            if item not in self._bucket_item_embedding_cache
        ]
        if not missing_items:
            return

        embeddings = run_parallel(
            missing_items,
            self._bucket_embedding_fn,
            max_workers=self._num_summary_embed_workers,
            desc="embed bucket items",
            disable=not self._verbose,
        )
        for item, embedding in zip(missing_items, embeddings):
            self._bucket_item_embedding_cache[item] = np.array(embedding, dtype=float)

    def _get_cluster_bucket_stats(
        self,
        cluster: SummarizedProfileCluster,
        feature_columns: Sequence[str],
    ) -> dict[str, dict[str, object]]:
        feature_stats = self._get_cluster_feature_stats(cluster, feature_columns)
        bucket_stats = {
            bucket_key: dict(feature_stats.get("concept", {}).get(bucket_key, {}))
            for bucket_key in UNIQUE_MASTERY_BUCKET_KEYS
        }
        for ability_bucket_key in UNIQUE_ABILITY_BUCKET_KEYS:
            bucket_stats[ability_bucket_key] = dict(feature_stats.get("ability", {}).get(ability_bucket_key, {}))
        return bucket_stats

    def _aggregate_bucket_item_embeddings(
        self,
        bucket_payload: dict[str, object],
    ) -> Optional[ndarray]:
        item_names = list(bucket_payload.get("items", []))
        if not item_names:
            return None

        embeddings: list[ndarray] = []
        for item_name in item_names:
            item_name = str(item_name)
            if item_name not in self._bucket_item_embedding_cache:
                continue
            embeddings.append(self._bucket_item_embedding_cache[item_name])

        if not embeddings:
            return None
        return np.mean(np.array(embeddings, dtype=float), axis=0)

    def _build_cluster_representation(
        self,
        bucket_stats: dict[str, dict[str, object]],
    ) -> ClusterBucketRepresentation:
        concept_embeddings = ConceptBucketEmbeddings(
            not_started_concepts=self._aggregate_bucket_item_embeddings(bucket_stats["not_started_concepts"]),
            just_started_concepts=self._aggregate_bucket_item_embeddings(bucket_stats["just_started_concepts"]),
            struggle_concepts=self._aggregate_bucket_item_embeddings(bucket_stats["struggle_concepts"]),
            good_concepts=self._aggregate_bucket_item_embeddings(bucket_stats["good_concepts"]),
            strong_concepts=self._aggregate_bucket_item_embeddings(bucket_stats["strong_concepts"]),
        )
        ability_embeddings = AbilityBucketEmbeddings(
            low_ability=self._aggregate_bucket_item_embeddings(bucket_stats["low_ability"]),
            medium_ability=self._aggregate_bucket_item_embeddings(bucket_stats["medium_ability"]),
            high_ability=self._aggregate_bucket_item_embeddings(bucket_stats["high_ability"]),
        )
        return ClusterBucketRepresentation(
            concept=concept_embeddings,
            ability=ability_embeddings,
        )

    @staticmethod
    def _cosine_distance(left: ndarray, right: ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 1.0
        cosine_similarity = float(np.dot(left, right) / (left_norm * right_norm))
        cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))
        return 1.0 - cosine_similarity

    def _calculate_type_distance(
        self,
        left: ClusterBucketRepresentation,
        right: ClusterBucketRepresentation,
        bucket_names: Sequence[str],
    ) -> Optional[float]:
        bucket_distances: list[float] = []
        for bucket_name in bucket_names:
            left_embedding = left.get(bucket_name)
            right_embedding = right.get(bucket_name)

            if left_embedding is None and right_embedding is None:
                continue
            if left_embedding is None or right_embedding is None:
                bucket_distances.append(1.0)
                continue
            bucket_distances.append(self._cosine_distance(left_embedding, right_embedding))

        if not bucket_distances:
            return None
        return float(np.mean(bucket_distances))

    def _calculate_representation_distance(
        self,
        left: ClusterBucketRepresentation,
        right: ClusterBucketRepresentation,
    ) -> float:
        type_distances = [
            self._calculate_type_distance(left, right, CONCEPT_BUCKET_NAMES),
            self._calculate_type_distance(left, right, ABILITY_BUCKET_NAMES),
        ]
        type_distances = [distance for distance in type_distances if distance is not None]
        if not type_distances:
            return 0.0
        return float(np.mean(type_distances))

    @staticmethod
    def _get_item_bucket_map(
        bucket_stats: dict[str, dict[str, object]],
        bucket_names: Sequence[str],
    ) -> dict[str, str]:
        item_to_bucket: dict[str, str] = {}
        for bucket_name in bucket_names:
            for item_name in bucket_stats.get(bucket_name, {}).get("items", []):
                item_to_bucket[str(item_name)] = bucket_name
        return item_to_bucket

    @staticmethod
    def _bucket_pair_distance(
        left_bucket: str,
        right_bucket: str,
        bucket_ranks: dict[str, int],
    ) -> float:
        max_rank_gap = max(bucket_ranks.values()) - min(bucket_ranks.values())
        if max_rank_gap <= 0:
            return 0.0
        left_rank = bucket_ranks[left_bucket]
        right_rank = bucket_ranks[right_bucket]
        return abs(left_rank - right_rank) / max_rank_gap

    def _calculate_cross_bucket_type_distance(
        self,
        left_bucket_stats: dict[str, dict[str, object]],
        right_bucket_stats: dict[str, dict[str, object]],
        bucket_names: Sequence[str],
        bucket_ranks: dict[str, int],
    ) -> Optional[float]:
        left_map = self._get_item_bucket_map(left_bucket_stats, bucket_names)
        right_map = self._get_item_bucket_map(right_bucket_stats, bucket_names)
        all_items = sorted(set(left_map) | set(right_map))
        if not all_items:
            return None

        item_distances: list[float] = []
        for item_name in all_items:
            left_bucket = left_map.get(item_name)
            right_bucket = right_map.get(item_name)
            if left_bucket is None or right_bucket is None:
                item_distances.append(1.0)
                continue
            item_distances.append(
                self._bucket_pair_distance(left_bucket, right_bucket, bucket_ranks)
            )
        return float(np.mean(item_distances))

    def _calculate_cross_bucket_distance(
        self,
        left_bucket_stats: dict[str, dict[str, object]],
        right_bucket_stats: dict[str, dict[str, object]],
    ) -> float:
        type_distances = [
            self._calculate_cross_bucket_type_distance(
                left_bucket_stats,
                right_bucket_stats,
                CONCEPT_BUCKET_NAMES,
                CONCEPT_BUCKET_RANKS,
            ),
            self._calculate_cross_bucket_type_distance(
                left_bucket_stats,
                right_bucket_stats,
                ABILITY_BUCKET_NAMES,
                ABILITY_BUCKET_RANKS,
            ),
        ]
        type_distances = [distance for distance in type_distances if distance is not None]
        if not type_distances:
            return 0.0
        return float(np.mean(type_distances))

    def _build_cluster_distance_matrix(
        self,
        clusters: Sequence[SummarizedProfileCluster],
        feature_columns: Sequence[str],
    ) -> ndarray:
        self._ensure_bucket_item_embeddings(feature_columns)
        bucket_stats_list = [
            self._get_cluster_bucket_stats(cluster, feature_columns)
            for cluster in clusters
        ]
        representations = [
            self._build_cluster_representation(bucket_stats)
            for bucket_stats in bucket_stats_list
        ]

        distances = np.zeros((len(clusters), len(clusters)), dtype=float)
        for left_index in range(len(representations)):
            for right_index in range(left_index + 1, len(representations)):
                semantic_distance = self._calculate_representation_distance(
                    representations[left_index],
                    representations[right_index],
                )
                cross_bucket_distance = self._calculate_cross_bucket_distance(
                    bucket_stats_list[left_index],
                    bucket_stats_list[right_index],
                )
                distance = (
                    (1.0 - self._cross_bucket_distance_weight) * semantic_distance
                    + self._cross_bucket_distance_weight * cross_bucket_distance
                )
                distances[left_index, right_index] = distance
                distances[right_index, left_index] = distance
        return distances

    def _optimize_merge_small_clusters(
        self,
        clusters: Sequence[SummarizedProfileCluster],
        feature_columns: Sequence[str],
    ) -> list[SummarizedProfileCluster]:
        if len(clusters) <= 1:
            return list(clusters)

        distances = self._build_cluster_distance_matrix(clusters, feature_columns)
        best_clusters = list(clusters)
        highest_clustering_score = self._calculate_clustering_score(best_clusters)

        for threshold in self._cosine_distance_thresholds_to_combine:
            temp_clusters = self._merge_small_clusters(clusters, distances, threshold)
            temp_clusters = self._summarize_clusters(temp_clusters, feature_columns=feature_columns)
            temp_score = self._calculate_clustering_score(temp_clusters)
            if temp_score > highest_clustering_score:
                highest_clustering_score = temp_score
                best_clusters = temp_clusters
        return best_clusters

    def _default_calculate_clustering_score(
        self,
        clusters: Sequence[SummarizedProfileCluster],
    ) -> float:
        if len(clusters) <= 1:
            return 0.0
        if not self._active_feature_columns:
            raise ValueError("Feature columns are required before concept-bag scoring can run.")

        distances = self._build_cluster_distance_matrix(clusters, self._active_feature_columns)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        return float(np.mean(min_distances))
