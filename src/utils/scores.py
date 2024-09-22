"""
Clustering Score Calculation Module.

This module provides functionality to calculate various clustering performance scores.

Functions:
    calculate_clustering_scores: Calculates multiple clustering performance scores.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def calculate_clustering_scores(features_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
    else:
        silhouette_avg = calinski_harabasz = davies_bouldin = -1
    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }