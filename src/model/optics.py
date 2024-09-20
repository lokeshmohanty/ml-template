"""
OPTICS Clustering implementation module.

This module provides an OPTICSClusterer class for performing 
OPTICS (Ordering Points To Identify the Clustering Structure) clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, OPTICS, NearestNeighbors, MIN_SAMPLES_FACTOR, XI, MIN_CLUSTER_SIZE_FACTOR, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any

from config import (
    np, OPTICS, NearestNeighbors,
    MIN_SAMPLES_FACTOR, XI, MIN_CLUSTER_SIZE_FACTOR, plt
)
from utils.scores import calculate_clustering_scores
from utils.visualization import plot_optics
from clearml import Task

class OPTICSClusterer:
    def __init__(self, task=None):
        if task is None:
            self.task = Task.init(
                project_name='CAESAR',
                task_name='optics',
                task_type=Task.TaskTypes.training
                ) 
        else:
            self.task = task

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        min_samples = max(5, int(MIN_SAMPLES_FACTOR * len(features_scaled)))
        min_cluster_size = max(5, int(MIN_CLUSTER_SIZE_FACTOR * len(features_scaled)))

        optics = OPTICS(min_samples=min_samples, xi=XI, min_cluster_size=min_cluster_size)
        labels = optics.fit_predict(features_scaled)

        neighbors = NearestNeighbors(n_neighbors=optics.min_samples).fit(features_scaled)
        core_distances_nn = np.sort(neighbors.kneighbors(features_scaled)[0][:, -1])

        scores = calculate_clustering_scores(features_scaled, labels)

        cluster_labels = np.unique(labels)
        cluster_densities = {}
        for cluster_label in cluster_labels:
            cluster_points = features_scaled[labels == cluster_label]
            if len(cluster_points) > 0:
                density = len(cluster_points) / (np.pi * (np.max(core_distances_nn[optics.ordering_][labels == cluster_label]) ** 2))
                cluster_densities[cluster_label] = density
                
        for metric, score in scores.items():
            self.task.logger.report_scalar(title="Clustering Score", series=metric, value=score, iteration=0)
        
        # Plot and log the clustering results
        plot_optics(features_scaled, labels, self.task)
        
        self.task.connect({"min_samples": min_samples, "xi": XI, "min_cluster_size": min_cluster_size})

        return {
            'scores': scores,
            'labels': labels,
            'cluster_densities': cluster_densities,
            'parameters': {
                'min_samples': min_samples,
                'xi': XI,
                'min_cluster_size': min_cluster_size
            }
        }
        
    def close_task(self):
        if hasattr(self, 'task'):
            self.task.close()