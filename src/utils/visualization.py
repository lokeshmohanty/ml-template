"""
Clustering Visualization Module.

This module provides functions to visualize clustering results using 3D plots.

Functions:
    plot_clusters_3d: Creates a 3D scatter plot of clustered data.
    plot_kmeans: Plots KMeans clustering results.
    plot_gmm: Plots Gaussian Mixture Model clustering results.
    plot_dbscan: Plots DBSCAN clustering results.
    plot_ensemble: Plots Ensemble clustering results.
    plot_agglomerative: Plots Agglomerative clustering results.
    plot_optics: Plots OPTICS clustering results.
    plot_hdbscan: Plots HDBSCAN clustering results.
"""
from typing import List,Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from clearml import Task, Logger

def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str, task: Optional[Task] = None) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    
    unique_labels = np.unique(labels)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    color_map = plt.colormaps['rainbow']
    norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
    
    for label in unique_labels:
        if label == -1:
            color = 'k'  # Black color for noise
        else:
            color = color_map(norm(label))
        
        mask = (labels == label)
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2],
                   c=[color], label=f'Cluster {label}')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()
    ax.set_title(title)
    
    if task:
        try:
            logger = task.get_logger()
            if logger:
                logger.report_matplotlib_figure(title=title, series="Clusters", iteration=0, figure=fig)
                
                # Report cluster sizes
                for label in unique_labels:
                    count = np.sum(labels == label)
                    logger.report_scalar(title="Cluster Sizes", series=f"Cluster {label}", value=count, iteration=0)
            else:
                print("Warning: Logger not available. Skipping ClearML logging.")
        except Exception as e:
            print(f"Warning: Error in ClearML logging - {str(e)}. Skipping ClearML logging.")
    
    plt.close(fig)
    
    return pca_result, unique_labels

def plot_kmeans(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot KMeans clustering results."""
    plot_clusters_3d(features_scaled, labels, 'KMeans Clustering with 3D PCA', task)

def plot_gmm(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot Gaussian Mixture Model clustering results."""
    plot_clusters_3d(features_scaled, labels, 'Gaussian Mixture Model Clustering with 3D PCA', task)

def plot_dbscan(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot DBSCAN clustering results."""
    plot_clusters_3d(features_scaled, labels, 'DBSCAN Clustering with 3D PCA', task)

def plot_ensemble(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot Ensemble clustering results."""
    plot_clusters_3d(features_scaled, labels, 'Ensemble Clustering with 3D PCA', task)

def plot_agglomerative(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot Agglomerative clustering results."""
    plot_clusters_3d(features_scaled, labels, 'Agglomerative Clustering with 3D PCA', task)

def plot_optics(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot OPTICS clustering results."""
    plot_clusters_3d(features_scaled, labels, 'OPTICS Clustering with 3D PCA', task)

def plot_hdbscan(features_scaled: np.ndarray, labels: np.ndarray, task: Task):
    """Plot HDBSCAN clustering results."""
    plot_clusters_3d(features_scaled, labels, 'HDBSCAN Clustering with 3D PCA', task)
    
    




