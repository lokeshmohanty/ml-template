"""
HDBSCAN Clustering implementation module.

This module provides an HDBSCANClusterer class for performing 
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) 
clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, hdbscan, MIN_CLUSTER_SIZE_FACTOR, MIN_SAMPLES_FACTOR, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any
from clearml import Task
import numpy as np
import hdbscan

from src.utils.scores import calculate_clustering_scores
from src.utils.visualization import plot_hdbscan

MIN_CLUSTER_SIZE_FACTOR = 0.02
MIN_SAMPLES_FACTOR = 0.01

class HDBSCANClusterer:
    """HDBSCAN clustering implementation."""

    def __init__(self, task=None):
        if task is None:
            self.task = Task.init(
                project_name='CAESAR',
                task_name='hdbscan_clusterer',
                task_type=Task.TaskTypes.training
            )
        else:
            self.task = task

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        min_cluster_size = max(5, int(MIN_CLUSTER_SIZE_FACTOR * len(features_scaled)))
        min_samples = max(5, int(MIN_SAMPLES_FACTOR * len(features_scaled)))

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, labels)
        for metric, score in scores.items():
            self.task.logger.report_scalar(title="Clustering Score", series=metric, value=score, iteration=0)

        # Plot and log the clustering results
        plot_hdbscan(features_scaled, labels, self.task)

        self.task.connect({"min_cluster_size": min_cluster_size, "min_samples": min_samples})

        return {
            'scores': scores,
            'labels': labels,
            'parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples
            }
        }

    def close_task(self):
        if hasattr(self, 'task'):
            self.task.close()