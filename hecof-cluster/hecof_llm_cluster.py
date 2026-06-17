from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from statistics import mean, median, stdev
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from numpy import ndarray
from sklearn.base import ClusterMixin
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

from bucket_utils import (
    ABILITY_SCALE_DEFAULT,
    ABILITY_BUCKETS,
    MASTERED_SCALE_DEFAULT,
    MASTERY_BUCKETS,
    UNIQUE_ABILITY_BUCKET_KEYS,
    UNIQUE_MASTERY_BUCKET_KEYS,
    bucket_items,
    get_mastery_bucket_key,
    get_mastery_bucket_rank,
    get_student_ability_bucket_key,
)
from util import run_parallel


ProfileInput = Mapping[str, Any]


@dataclass
class SummarizedProfileCluster:
    """
    Container for one cluster of learner profiles plus the LLM-generated summary.

    `member_indices` preserve the original input order so we can reconstruct a
    sklearn-style label vector after clusters are merged or reclustered.
    """

    label: int
    member_indices: list[int]
    vectors: list[ndarray]
    profiles: list[dict[str, Any]]
    summary: Optional[str] = None
    recommended_action: Optional[str] = None

    def merge(self, other: "SummarizedProfileCluster") -> "SummarizedProfileCluster":
        return SummarizedProfileCluster(
            label=min(self.label, other.label),
            member_indices=self.member_indices + other.member_indices,
            vectors=self.vectors + other.vectors,
            profiles=self.profiles + other.profiles,
            summary=None,
            recommended_action=None,
        )

    def set_summary(self, summary: Optional[str], recommended_action: Optional[str] = None) -> None:
        self.summary = (summary or "").strip() or None
        self.recommended_action = (recommended_action or "").strip() or None

    def __len__(self) -> int:
        return len(self.profiles)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SummarizedProfileCluster):
            return False
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash((self.label, tuple(sorted(self.member_indices))))


def _fit_predict_cluster_model(model: ClusterMixin, vectors: ndarray) -> ndarray:
    if hasattr(model, "fit_predict"):
        return model.fit_predict(vectors)
    fitted = model.fit(vectors)
    if hasattr(fitted, "predict"):
        return fitted.predict(vectors)
    raise AttributeError(f"Cluster model {model} does not support fit_predict() or predict().")


def _default_summary_embedding_fn(text: str) -> ndarray:
    from langchain_openai import OpenAIEmbeddings

    return np.array(OpenAIEmbeddings(model="text-embedding-3-small").embed_query(text))


class HecofLLMClusterOptimizer:
    """
    Clusters structured learner profiles using concept-level mastery features
    plus an optional cross-concept average ability feature,
    then uses summary embeddings to merge clusters that are semantically similar.

    Workflow:
    1. Convert profiles into a numeric feature matrix.
    2. Try each candidate clustering model and keep the best-scoring result.
    3. Break up oversized clusters via `recluster_model`.
        4. Build a deterministic cluster summary string from aggregated stats.
        5. Merge undersized clusters when their summaries are semantically close.
    """

    def __init__(
        self,
        cluster_models: Sequence[ClusterMixin | Callable[[ndarray], ClusterMixin]],
        recluster_model: Optional[ClusterMixin] = None,
        summary_embedding_fn: Optional[Callable[[str], ndarray]] = None,
        calculate_clustering_score: Optional[Callable[[list[SummarizedProfileCluster]], float]] = None,
        cosine_distance_thresholds_to_combine: Optional[Sequence[float]] = None,
        small_cluster_size: int = 3,
        mastery_scale: float = MASTERED_SCALE_DEFAULT,
        ability_scale: float = ABILITY_SCALE_DEFAULT,
        top_concepts_per_summary: int = 4,
        num_summarization_workers: int = 25,
        num_summary_embed_workers: int = 50,
        verbose: bool = True,
    ):
        self._cluster_models = cluster_models
        self._recluster_model = recluster_model or AffinityPropagation(damping=0.7)
        self._summary_embedding_fn = summary_embedding_fn or _default_summary_embedding_fn
        self._calculate_clustering_score = calculate_clustering_score or self._default_calculate_clustering_score
        self._cosine_distance_thresholds_to_combine = cosine_distance_thresholds_to_combine or [0.2, 0.25, 0.3]
        self._small_cluster_size = small_cluster_size
        self._mastery_scale = mastery_scale
        self._ability_scale = ability_scale
        self._top_concepts_per_summary = top_concepts_per_summary
        self._num_summarization_workers = num_summarization_workers
        self._num_summary_embed_workers = num_summary_embed_workers
        self._verbose = verbose

    def fit_predict_profiles(
        self,
        profiles: Sequence[ProfileInput],
        feature_columns: Optional[Sequence[str]] = None,
    ) -> tuple[ndarray, list[SummarizedProfileCluster]]:
        """
        Cluster learner profiles and return:
        - `labels`: one label per input profile in original order
        - `clusters`: summarized clusters after split/merge post-processing

        Profiles may be either:
        - flattened dicts with columns like `<concept>_masteryStatus` plus `cross_concept_avg_ability`
        - raw API-style profiles with a `concepts` array/dict
        """

        if not profiles:
            raise ValueError("`profiles` must contain at least one learner profile.")

        flattened_profiles = [self._flatten_profile(profile) for profile in profiles]
        feature_columns = list(feature_columns or self._infer_feature_columns(flattened_profiles))
        if not feature_columns:
            raise ValueError("Could not infer numeric feature columns from learner profiles.")
        vectors = self._build_feature_matrix(flattened_profiles, feature_columns)

        best_clusters: Optional[list[SummarizedProfileCluster]] = None
        highest_clustering_score = float("-inf")
        chosen_model: Optional[ClusterMixin | Callable[[ndarray], ClusterMixin]] = None

        for cluster_model in self._cluster_models:
            if self._verbose:
                print(f"Clustering model: {cluster_model}")

            labels = self._get_labels(vectors, cluster_model)
            clusters = self._build_clusters(labels, vectors, flattened_profiles)
            summarized_clusters = self._summarize_clusters(clusters, feature_columns=feature_columns)
            summarized_clusters = self._optimize_merge_small_clusters(summarized_clusters, feature_columns=feature_columns)
            clustering_score = self._calculate_clustering_score(summarized_clusters)

            if clustering_score > highest_clustering_score:
                highest_clustering_score = clustering_score
                best_clusters = summarized_clusters
                chosen_model = cluster_model

        if best_clusters is None:
            raise ValueError("No clustering result was produced. Check the input profiles and cluster models.")

        if self._verbose:
            print("Highest clustering score", highest_clustering_score)
            print("Best model", chosen_model)

        labels = np.full(len(flattened_profiles), -1, dtype=int)
        for new_label, cluster in enumerate(best_clusters):
            cluster.label = new_label
            for index in cluster.member_indices:
                labels[index] = new_label
        return labels, best_clusters

    @staticmethod
    def _get_labels(
        vectors: ndarray,
        cluster_model: ClusterMixin | Callable[[ndarray], ClusterMixin],
    ) -> ndarray:
        if callable(cluster_model):
            cluster_model = cluster_model(vectors)
        return _fit_predict_cluster_model(cluster_model, vectors)

    @staticmethod
    def _coerce_numeric(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isnan(value):
                return None
            return float(value)
        return None

    def _flatten_profile(self, profile: ProfileInput) -> dict[str, Any]:
        """
        Accept either a flat feature row or the learner-profile JSON shape from the notebook.
        """

        if "concepts" not in profile:
            return dict(profile)

        flattened: dict[str, Any] = {}
        student_id = (
            profile.get("student_id")
            or profile.get("displayName")
            or profile.get("account", {}).get("name")
        )
        if student_id:
            flattened["student_id"] = student_id

        concepts = profile.get("concepts") or []
        if isinstance(concepts, Mapping):
            iterable = []
            for guid, concept in concepts.items():
                concept_data = dict(concept)
                concept_data.setdefault("guid", guid)
                iterable.append(concept_data)
        else:
            iterable = list(concepts)

        for concept in iterable:
            concept_name = concept.get("conceptName") or concept.get("guid")
            if not concept_name:
                continue
            flattened[f"{concept_name}_masteryStatus"] = concept.get("masteryStatus")
        ability_values = [
            self._coerce_numeric(concept.get("ability"))
            for concept in iterable
        ]
        ability_values = [value for value in ability_values if value is not None]
        if ability_values:
            flattened["cross_concept_avg_ability"] = round(mean(ability_values), 2)
        return flattened

    def _infer_feature_columns(self, profiles: Sequence[dict[str, Any]]) -> list[str]:
        """
        Prefer concept mastery fields plus cross-concept ability. Fall back to any numeric columns.
        """

        ordered_keys: list[str] = []
        seen = set()
        for profile in profiles:
            for key in profile.keys():
                if key not in seen:
                    seen.add(key)
                    ordered_keys.append(key)

        preferred = [
            key for key in ordered_keys
            if key.endswith("_masteryStatus") or key == "cross_concept_avg_ability"
        ]
        if preferred:
            return preferred

        return [
            key for key in ordered_keys
            if self._coerce_numeric(profiles[0].get(key)) is not None
        ]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        if feature_name.endswith("_masteryStatus") and self._mastery_scale:
            return value / self._mastery_scale
        if feature_name == "cross_concept_avg_ability" and self._ability_scale:
            return value / self._ability_scale
        return value

    def _build_feature_matrix(
        self,
        profiles: Sequence[dict[str, Any]],
        feature_columns: Sequence[str],
    ) -> ndarray:
        rows = []
        for profile in profiles:
            row = []
            for feature_name in feature_columns:
                value = self._coerce_numeric(profile.get(feature_name))
                row.append(self._scale_feature_value(feature_name, value or 0.0))
            rows.append(row)
        return np.array(rows, dtype=float)

    @staticmethod
    def _build_clusters_from_cluster_results(
        labels: ndarray,
        vectors: ndarray,
        profiles: Sequence[dict[str, Any]],
        member_indices: Sequence[int],
    ) -> tuple[list[SummarizedProfileCluster], list[int]]:
        clusters: list[SummarizedProfileCluster] = []
        noise_indices: list[int] = []

        for label in np.unique(labels):
            group_positions = np.where(labels == label)[0]
            group_member_indices = [member_indices[position] for position in group_positions]
            group_vectors = [vectors[position] for position in group_positions]
            group_profiles = [profiles[position] for position in group_positions]
            if label == -1:
                noise_indices.extend(group_member_indices)
                continue
            clusters.append(
                SummarizedProfileCluster(
                    label=int(label),
                    member_indices=group_member_indices,
                    vectors=group_vectors,
                    profiles=group_profiles,
                )
            )
        return clusters, noise_indices

    def _build_clusters(
        self,
        labels: ndarray,
        vectors: ndarray,
        profiles: Sequence[dict[str, Any]],
        member_indices: Optional[Sequence[int]] = None,
    ) -> list[SummarizedProfileCluster]:
        member_indices = list(member_indices or range(len(profiles)))
        clusters, noise_indices = self._build_clusters_from_cluster_results(
            labels,
            vectors,
            profiles,
            member_indices,
        )
        clusters = self._recluster_large_clusters(clusters)

        if noise_indices:
            noise_positions = [member_indices.index(index) for index in noise_indices]
            noise_vectors = np.array([vectors[position] for position in noise_positions], dtype=float)
            noise_profiles = [profiles[position] for position in noise_positions]
            noise_labels = _fit_predict_cluster_model(self._recluster_model, noise_vectors)
            noise_clusters, _ = self._build_clusters_from_cluster_results(
                noise_labels,
                noise_vectors,
                noise_profiles,
                noise_indices,
            )
            clusters.extend(noise_clusters)
        return clusters

    def _get_cluster_feature_stats(
        self,
        cluster: SummarizedProfileCluster,
        feature_columns: Sequence[str],
    ) -> dict[str, Any]:
        concept_stats: dict[str, dict[str, float]] = {}
        cluster_ability_values: list[float] = []

        for feature_name in feature_columns:
            if feature_name == "cross_concept_avg_ability":
                values = [
                    self._coerce_numeric(profile.get(feature_name))
                    for profile in cluster.profiles
                ]
                values = [value for value in values if value is not None]
                cluster_ability_values.extend(values)
                continue

            values = [
                self._coerce_numeric(profile.get(feature_name))
                for profile in cluster.profiles
            ]
            values = [value for value in values if value is not None]
            if not values:
                continue

            concept_name, metric_name = feature_name.rsplit("_", 1)
            concept_stats.setdefault(concept_name, {})
            if metric_name == "masteryStatus":
                mastery_ranks = [get_mastery_bucket_rank(value) for value in values]
                concept_stats[concept_name]["median_mastery_rank"] = round(median(mastery_ranks), 2)

        concepts_by_bucket: dict[str, list[dict[str, Any]]] = {
            bucket_key: [] for bucket_key in UNIQUE_MASTERY_BUCKET_KEYS
        }
        for concept_name, stats in concept_stats.items():
            if "median_mastery_rank" not in stats:
                continue
            mastery_bucket = get_mastery_bucket_key(stats["median_mastery_rank"])
            stat = {
                "concept_name": concept_name,
                "median_mastery_rank": stats["median_mastery_rank"],
                "mastery_bucket": mastery_bucket,
            }
            concepts_by_bucket[mastery_bucket].append(stat)

        for bucket_threshold, bucket_key, _ in MASTERY_BUCKETS:
            reverse = bucket_threshold >= 3
            concepts_by_bucket[bucket_key].sort(
                key=lambda item: (
                    -item["median_mastery_rank"] if reverse else item["median_mastery_rank"],
                    item["concept_name"],
                )
            )

        student_ability_median = (
            round(median(cluster_ability_values), 2)
            if cluster_ability_values else None
        )
        active_ability_bucket_key = get_student_ability_bucket_key(student_ability_median)
        feature_stats: dict[str, Any] = {
            "concept": {},
            "ability": {},
        }
        for bucket_key in UNIQUE_MASTERY_BUCKET_KEYS:
            bucket_concepts = concepts_by_bucket[bucket_key]
            bucket_payload = {
                "items": [concept["concept_name"] for concept in bucket_concepts],
                "value": (
                    round(
                        median([concept["median_mastery_rank"] for concept in bucket_concepts]),
                        2,
                    )
                    if bucket_concepts else None
                ),
            }
            feature_stats["concept"][bucket_key] = bucket_payload
        for ability_bucket_key in UNIQUE_ABILITY_BUCKET_KEYS:
            bucket_payload = {
                "items": (
                    ["cross_concept_avg_ability"]
                    if ability_bucket_key == active_ability_bucket_key
                    else []
                ),
                "value": (
                    student_ability_median
                    if ability_bucket_key == active_ability_bucket_key
                    else None
                ),
            }
            feature_stats["ability"][ability_bucket_key] = bucket_payload
        return feature_stats

    @staticmethod
    def _get_active_ability_bucket_key(ability_buckets: Mapping[str, Mapping[str, Any]]) -> str:
        for ability_bucket_key in UNIQUE_ABILITY_BUCKET_KEYS:
            if bucket_items(ability_buckets.get(ability_bucket_key, {})):
                return ability_bucket_key
        return ABILITY_BUCKETS[1][1]

    def _render_cluster_for_embedding(
        self,
        cluster: SummarizedProfileCluster,
        feature_columns: Sequence[str],
    ) -> str:
        """
        Convert a numeric cluster into a canonical text form for embeddings.
        """

        concept_stats = self._get_cluster_feature_stats(cluster, feature_columns)
        active_ability_bucket_key = self._get_active_ability_bucket_key(concept_stats["ability"])
        lines = [
            f"cluster_size={len(cluster)}",
            f"median_cross_concept_avg_ability_bucket={active_ability_bucket_key}",
        ]
        for bucket_key in UNIQUE_MASTERY_BUCKET_KEYS:
            bucket_names = (
                "|".join(
                    bucket_items(concept_stats["concept"][bucket_key])[: self._top_concepts_per_summary]
                )
                or "none"
            )
            lines.append(f"{bucket_key}={bucket_names}")
        return "\n".join(lines)

    def _render_cluster_for_prompt(
        self,
        cluster: SummarizedProfileCluster,
        feature_columns: Sequence[str],
    ) -> str:
        """
        Convert a numeric cluster into a human-readable text block.
        """

        concept_stats = self._get_cluster_feature_stats(cluster, feature_columns)
        active_ability_bucket_key = self._get_active_ability_bucket_key(concept_stats["ability"])
        lines = [
            f"cluster_size: {len(cluster)}",
            f"median_cross_concept_avg_ability_bucket: {active_ability_bucket_key}",
        ]
        for bucket_key in UNIQUE_MASTERY_BUCKET_KEYS:
            lines.append(f"{bucket_key}:")
            for item in bucket_items(
                concept_stats["concept"][bucket_key]
            )[: self._top_concepts_per_summary]:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def _summarize_cluster(
        self,
        cluster: SummarizedProfileCluster,
        feature_columns: Sequence[str],
    ) -> SummarizedProfileCluster:
        if len(cluster) == 1:
            cluster.set_summary(self._render_cluster_for_embedding(cluster, feature_columns))
            if self._verbose:
                print(f"Cluster summary | size={len(cluster)} | summary={cluster.summary}")
            return cluster

        cluster.set_summary(self._render_cluster_for_embedding(cluster, feature_columns))
        if self._verbose:
            print(f"Cluster summary | size={len(cluster)} | summary={cluster.summary}")
        return cluster

    def _summarize_clusters(
        self,
        clusters: Sequence[SummarizedProfileCluster],
        **kwargs: Any,
    ) -> list[SummarizedProfileCluster]:
        return run_parallel(
            clusters,
            partial(self._summarize_cluster, **kwargs),
            max_workers=self._num_summarization_workers,
            desc="summarize profile clusters",
            disable=not self._verbose,
        )

    def _embed_cluster_summaries(self, clusters: Sequence[SummarizedProfileCluster]) -> ndarray:
        texts = []
        for cluster in clusters:
            texts.append(cluster.summary or "")
        embeddings = run_parallel(
            texts,
            self._summary_embedding_fn,
            max_workers=self._num_summary_embed_workers,
            desc="embed cluster summaries",
            disable=not self._verbose,
        )
        return np.array(embeddings, dtype=float)

    def _optimize_merge_small_clusters(
        self,
        clusters: Sequence[SummarizedProfileCluster],
        feature_columns: Sequence[str],
    ) -> list[SummarizedProfileCluster]:
        if len(clusters) <= 1:
            return list(clusters)

        embeddings = self._embed_cluster_summaries(clusters)
        distances = pairwise_distances(embeddings, metric="cosine")

        best_clusters = list(clusters)
        highest_clustering_score = self._calculate_clustering_score(best_clusters)

        for threshold in self._cosine_distance_thresholds_to_combine:
            temp_clusters = self._merge_small_clusters(clusters, distances, threshold)
            # Merged clusters need fresh summaries before they are scored again.
            temp_clusters = self._summarize_clusters(temp_clusters, feature_columns=feature_columns)
            temp_score = self._calculate_clustering_score(temp_clusters)
            if temp_score > highest_clustering_score:
                highest_clustering_score = temp_score
                best_clusters = temp_clusters
        return best_clusters

    def _merge_small_clusters(
        self,
        clusters: Sequence[SummarizedProfileCluster],
        distances: ndarray,
        threshold: float,
    ) -> list[SummarizedProfileCluster]:
        working_clusters = list(clusters)
        consumed = set()

        # Start with the smallest clusters so the merge stage preferentially
        # absorbs them into the closest compatible cluster.
        indices_by_size = sorted(range(len(working_clusters)), key=lambda index: len(working_clusters[index]))

        for index in indices_by_size:
            if index in consumed:
                continue
            cluster = working_clusters[index]
            if len(cluster) > self._small_cluster_size:
                continue

            candidate_indices = [
                other_index
                for other_index in range(len(working_clusters))
                if other_index != index
                and other_index not in consumed
                and distances[index, other_index] < threshold
            ]
            if not candidate_indices:
                continue

            best_target = min(
                candidate_indices,
                key=lambda other_index: (distances[index, other_index], -len(working_clusters[other_index])),
            )
            working_clusters[best_target] = working_clusters[best_target].merge(cluster)
            consumed.add(index)

        return [
            cluster for cluster_index, cluster in enumerate(working_clusters)
            if cluster_index not in consumed
        ]

    def _default_calculate_clustering_score(
        self,
        clusters: Sequence[SummarizedProfileCluster],
    ) -> float:
        """
        Higher is better.

        We score clusters by how far apart their semantic summaries are. This is
        the same idea as `llm_clusterer.py`, but applied to learner-pattern
        descriptions instead of source-text topics.
        """

        if len(clusters) <= 1:
            return 0.0

        embeddings = self._embed_cluster_summaries(clusters)
        distances = pairwise_distances(embeddings, metric="cosine")
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        return float(np.mean(min_distances))

    @staticmethod
    def _get_large_clusters(clusters: Sequence[SummarizedProfileCluster]) -> list[SummarizedProfileCluster]:
        if len(clusters) <= 2:
            return []

        large_clusters = []
        for cluster in clusters:
            other_cluster_lengths = [len(other) for other in clusters if other != cluster]
            if len(other_cluster_lengths) < 2:
                continue
            threshold = mean(other_cluster_lengths) + 3 * stdev(other_cluster_lengths)
            if len(cluster) > threshold:
                large_clusters.append(cluster)
        return large_clusters

    def _recluster_large_clusters(
        self,
        clusters: Sequence[SummarizedProfileCluster],
    ) -> list[SummarizedProfileCluster]:
        large_clusters = set(self._get_large_clusters(clusters))
        if not large_clusters:
            return list(clusters)

        remaining_clusters = [cluster for cluster in clusters if cluster not in large_clusters]
        for large_cluster in large_clusters:
            vectors = np.array(large_cluster.vectors, dtype=float)
            labels = _fit_predict_cluster_model(self._recluster_model, vectors)
            reclustered = self._build_clusters(
                labels,
                vectors,
                large_cluster.profiles,
                member_indices=large_cluster.member_indices,
            )
            if len(reclustered) <= 1 and sum(len(cluster) for cluster in reclustered) == len(large_cluster):
                remaining_clusters.append(large_cluster)
                continue
            remaining_clusters.extend(reclustered)
        return remaining_clusters
