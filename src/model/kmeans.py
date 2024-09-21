"""
KMeans clustering implementation module.
- This module provides a KMeansClusterer class for performing KMeans clustering
  on given data.

Imports:
    - typing: For type hinting
    - NumPy, KMeans, silhouette_score, MAX_CLUSTERS and matplotlib plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
import os
from typing import Dict, Any, List
from clearml import Task
from src.config import (
    np, KMeans, silhouette_score, MAX_CLUSTERS, plt
)
from src.utils.scores import calculate_clustering_scores
from src.utils.visualization import plot_kmeans

MAX_CLUSTERS = 10  # Define this constant if not imported from config

class KMeansClusterer:
    """KMeans clustering implementation."""

    def __init__(self, task=None):
        self.max_clusters = MAX_CLUSTERS
        if task is None:
            self.task = Task.init(
                project_name='CAESAR',
                task_name='kmeans',
                task_type=Task.TaskTypes.training
            )
        else:
            self.task = task

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        inertia = []
        silhouette_scores = []
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2

        optimal_k = optimal_k_silhouette

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, clusters)

        # Log each score separately
        for metric, score in scores.items():
            self.task.logger.report_scalar(title="Clustering Score", series=metric, value=score, iteration=0)

        # Plot and log the clustering results
        plot_kmeans(features_scaled, clusters, self.task)

        self.task.connect({"n_clusters": optimal_k})
        return {'scores': scores, 'optimal_k': optimal_k}

    @staticmethod
    def find_elbow(k_values: List[int], inertias: List[float]) -> int:
        diffs = np.diff(inertias)
        elbow_index = np.argmax(diffs) + 1
        return k_values[elbow_index]

    def close_task(self):
        if hasattr(self, 'task'):
            self.task.close()