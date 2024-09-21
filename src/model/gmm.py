"""
Gaussian Mixture Model (GMM) Clustering implementation module.

This module provides a GMMClusterer class for performing 
Gaussian Mixture Model clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, GaussianMixture, silhouette_score, MAX_COMPONENTS, RANDOM_STATE, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any
from clearml import Task
from config import (
    np, GaussianMixture, silhouette_score,
    MAX_COMPONENTS, RANDOM_STATE, plt
)
from utils.scores import calculate_clustering_scores
from utils.visualization import plot_gmm

class GMMClusterer:
    def __init__(self, task=None):
        self.max_components = MAX_COMPONENTS
        if task is None:
            self.task = Task.init(
                project_name='CAESAR',
                task_name='gmm',
                task_type=Task.TaskTypes.training
                ) 
        else:
            self.task = task

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        silhouette_scores = []
        bic_scores = []

        for k in range(2, self.max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
            gmm.fit(features_scaled)
            labels = gmm.predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
            bic_scores.append(gmm.bic(features_scaled))

        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        optimal_k_bic = bic_scores.index(min(bic_scores)) + 2

        optimal_k = optimal_k_silhouette  # Use silhouette score's optimal k as default

        gmm = GaussianMixture(n_components=optimal_k, random_state=RANDOM_STATE)
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)
        scores = calculate_clustering_scores(features_scaled, labels)
        
        for metric, score in scores.items():
            self.task.logger.report_scalar(title="Clustering Score", series=metric, value=score, iteration=0)
        
        # Plot and log the clustering results
        plot_gmm(features_scaled, labels, self.task)
        
        self.task.connect({"n_components": optimal_k})

        return {
            'scores': scores,
            'optimal_k': optimal_k
        }

    def close_task(self):
        if hasattr(self, 'task'):
            self.task.close()


