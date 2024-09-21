"""
DBSCAN Clustering implementation module.

This module provides a DBSCANClusterer class for performing 
DBSCAN Clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, NearestNeighbors, DBSCAN, K_NEIGHBORS, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any
from clearml import Task
from src.config import (
    np, NearestNeighbors, DBSCAN, K_NEIGHBORS, plt
)
from src.utils.scores import calculate_clustering_scores
from src.utils.visualization import plot_dbscan

class DBSCANClusterer:
    """DBSCAN clustering implementation."""

    def __init__(self, task=None):
        self.k = K_NEIGHBORS
        if task is None:
            self.task = Task.init(
                project_name='CAESAR',
                task_name='dbscan',
                task_type=Task.TaskTypes.training
            )
        else:
            self.task = task

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(features_scaled)
        distances, _ = neigh.kneighbors(features_scaled)
        sorted_distances = np.sort(distances[:, self.k-1], axis=0)

        knee_point = self.find_knee_point(sorted_distances)
        eps = sorted_distances[knee_point]
        min_samples = self.k

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, clusters)
        for metric, score in scores.items():
            self.task.logger.report_scalar(title="Clustering Score", series=metric, value=score, iteration=0)

        # Plot and log the clustering results
        plot_dbscan(features_scaled, clusters, self.task)

        self.task.connect({"eps": eps, "min_samples": min_samples})

        return {
            'scores': scores,
            'eps': eps,
            'min_samples': min_samples
        }

    @staticmethod
    def find_knee_point(distances: np.ndarray) -> int:
        diffs = np.diff(distances)
        knee_point = np.argmax(diffs) + 1
        return knee_point

    def close_task(self):
        if hasattr(self, 'task'):
            self.task.close()