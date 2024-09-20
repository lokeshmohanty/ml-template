"""
Agglomerative Clustering implementation module.

This module provides an AgglomerativeClusterer class for performing 
Agglomerative Clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, AgglomerativeClustering, silhouette_score, MAX_CLUSTERS, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any
from clearml import Task
from config import (
    np, AgglomerativeClustering, silhouette_score, MAX_CLUSTERS, plt
)
from utils.scores import calculate_clustering_scores
from utils.visualization import plot_agglomerative
class AgglomerativeClusterer:
    """
    A class for performing Agglomerative Clustering.

    This class provides a method to run Agglomerative Clustering on given data and find the optimal number of clusters.
    """
    def __init__(self, task=None):
        if task is None:
            self.task = Task.init(
                project_name='CAESAR',
                task_name='agglomerative',
                task_type=Task.TaskTypes.training
            )
        else:
            self.task = task

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Run Agglomerative Clustering on the given data.

        Parameters
        ----------
        _ : Any
            Unused parameter (kept for consistency with other clusterers).
            
        features_scaled : np.ndarray
            The scaled feature array to cluster.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the clustering scores, labels, and the optimal number of clusters.
        """
        
        max_clusters = min(MAX_CLUSTERS, int(np.sqrt(len(features_scaled))))

        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg_clustering.fit_predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        labels = agg_clustering.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, labels)
        for metric, score in scores.items():
            self.task.logger.report_scalar(title="Clustering Score", series=metric, value=score, iteration=0)
        
        # Plot and log the clustering results
        plot_agglomerative(features_scaled, labels, self.task)
        
        self.task.connect({"n_clusters": optimal_k})

        return {
            'scores': scores,
            'labels': labels,
            'optimal_k': optimal_k
        }

    def close_task(self):
        if hasattr(self, 'task'):
            self.task.close()