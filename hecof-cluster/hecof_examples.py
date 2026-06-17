from __future__ import annotations

from statistics import mean, median

import pandas as pd
from sklearn.cluster import AffinityPropagation, KMeans

from hecof_llm_cluster import HecofLLMClusterOptimizer
from util import guess_optimal_n_clusters


DATA_PATH = "observable/hecof/synthetic_student_data.csv"


def load_flat_profiles(csv_path: str = DATA_PATH) -> list[dict]:
    """
    Load learner profiles from the wide CSV used in the notebook and convert them
    into the flattened dict format expected by `fit_predict_profiles()`.
    """

    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def get_kmeans(vectors):
    """
    Example of passing a callable model factory, matching the pattern used in
    `llm_clusterer.py`.
    """

    n_clusters = guess_optimal_n_clusters(
        vectors,
        lambda n: KMeans(n_clusters=n, random_state=0, n_init="auto"),
    )
    return KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")


def local_cluster_summary(rendered_cluster: str):
    """
    Lightweight summary function that avoids any OpenAI dependency.

    The optimizer renders each numeric cluster into a compact text block. This
    example inspects that text and returns a summary/action pair in the same
    shape as the LLM-backed implementation.
    """

    lowered = rendered_cluster.lower()
    if "median mastery 0" in lowered or "median mastery 1" in lowered:
        return "Learners are struggling across multiple concepts", "guided mastery"
    if "median mastery 4" in lowered or "median mastery 5" in lowered:
        return "Learners are broadly confident and ready to discuss", "think-pair-share"
    return "Learners show partial understanding and need reinforcement", "revision"


def print_clusters(clusters):
    for cluster in clusters:
        print(
            f"label={cluster.label} size={len(cluster)} "
            f"action={cluster.recommended_action} summary={cluster.summary}"
        )


def example_with_flat_profiles():
    """
    Minimal example using the flattened CSV rows directly.

    This path is useful during local development because it does not require
    OpenAI credentials.
    """

    profiles = load_flat_profiles()
    optimizer = HecofLLMClusterOptimizer(
        cluster_models=[
            get_kmeans,
            AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100),
        ],
        recluster_model=AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100),
        get_cluster_summary=local_cluster_summary,
        summary_embedding_fn=lambda text: [len(text), text.count("guided"), text.count("revision"), text.count("think")],
        verbose=True,
    )
    labels, clusters = optimizer.fit_predict_profiles(profiles)
    print("labels shape:", labels.shape)
    print_clusters(clusters)


def example_with_selected_features():
    """
    Example that clusters using only a subset of concept columns.

    This is useful when you want topic-level grouping rather than clustering on
    every concept available in the profile.
    """

    profiles = load_flat_profiles()
    feature_columns = [
        column
        for column in profiles[0].keys()
        if "Experiment Extraction of Olive-based compounds" in column
        or "Factors influencing extraction" in column
    ]
    optimizer = HecofLLMClusterOptimizer(
        cluster_models=[get_kmeans],
        recluster_model=AffinityPropagation(damping=0.75),
        get_cluster_summary=local_cluster_summary,
        summary_embedding_fn=lambda text: [len(text), text.count("struggling"), text.count("confident")],
        verbose=True,
    )
    labels, clusters = optimizer.fit_predict_profiles(profiles, feature_columns=feature_columns)
    print("selected-feature labels:", labels.tolist())
    print_clusters(clusters)


def example_with_api_style_profiles():
    """
    Example using the raw learner-profile shape from the notebook/API.

    `HecofLLMClusterOptimizer` flattens the `concepts` payload internally, so
    you can pass the original structure directly.
    """

    api_profiles = [
        {
            "student_id": "student_a",
            "concepts": [
                {"conceptName": "Factors influencing extraction", "masteryStatus": 1, "ability": 42},
                {"conceptName": "Innovative extraction techniques", "masteryStatus": 2, "ability": 51},
            ],
        },
        {
            "student_id": "student_b",
            "concepts": [
                {"conceptName": "Factors influencing extraction", "masteryStatus": 4, "ability": 83},
                {"conceptName": "Innovative extraction techniques", "masteryStatus": 5, "ability": 88},
            ],
        },
        {
            "student_id": "student_c",
            "concepts": [
                {"conceptName": "Factors influencing extraction", "masteryStatus": 4, "ability": 79},
                {"conceptName": "Innovative extraction techniques", "masteryStatus": 4, "ability": 80},
            ],
        },
    ]

    optimizer = HecofLLMClusterOptimizer(
        cluster_models=[lambda vectors: KMeans(n_clusters=2, random_state=0, n_init="auto")],
        recluster_model=AffinityPropagation(damping=0.75),
        get_cluster_summary=local_cluster_summary,
        summary_embedding_fn=lambda text: [len(text), text.count("struggling"), text.count("confident")],
        verbose=False,
    )
    labels, clusters = optimizer.fit_predict_profiles(api_profiles)
    print("api-style labels:", labels.tolist())
    print_clusters(clusters)


def example_with_default_llm_summary():
    """
    Example using the default OpenAI-backed summary and summary embeddings.

    Requirements:
    - `OPENAI_API_KEY` must be set
    - network access must be available
    """

    profiles = load_flat_profiles()
    optimizer = HecofLLMClusterOptimizer(
        cluster_models=[get_kmeans],
        recluster_model=AffinityPropagation(damping=0.75),
        verbose=True,
    )
    labels, clusters = optimizer.fit_predict_profiles(profiles)
    print("labels shape:", labels.shape)
    print_clusters(clusters)


def summarize_cluster_statistics(clusters, profiles):
    """
    Optional helper that shows how to inspect the returned clusters.
    """

    for cluster in clusters:
        mastery_columns = [column for column in profiles[0].keys() if column.endswith("_masteryStatus")]
        ability_columns = [column for column in profiles[0].keys() if column.endswith("_ability")]
        mastery_values = []
        ability_values = []
        for profile in cluster.profiles:
            mastery_values.extend([profile[column] for column in mastery_columns if pd.notna(profile.get(column))])
            ability_values.extend([profile[column] for column in ability_columns if pd.notna(profile.get(column))])
        print(
            f"cluster={cluster.label} size={len(cluster)} "
            f"median_mastery={median(mastery_values):.2f} avg_ability={mean(ability_values):.2f}"
        )


if __name__ == "__main__":
    example_with_flat_profiles()
