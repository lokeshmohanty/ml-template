from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import hdbscan
from clearml import Task
from clearml.automation import PipelineDecorator

from utils.scores import calculate_clustering_scores
from utils.visualization import plot_clusters_3d

@PipelineDecorator.component(return_values=["initial_dataframe"], cache=True, task_type=Task.TaskTypes.data_processing)
def synthetic_data(n_samples=1000):
    print("Generating synthetic radar data")
    np.random.seed(42)  # For reproducibility of the results

    signal_duration = np.random.uniform(1e-6, 1e-3, n_samples) * 1e6
    azimuthal_angle = np.random.uniform(0, 360, n_samples)
    elevation_angle = np.random.uniform(-90, 90, n_samples)
    pri = np.random.uniform(1e-3, 1, n_samples) * 1e6
    start_time = datetime.now()
    timestamps: List[float] = [
        start_time + timedelta(microseconds=int(x))
        for x in np.cumsum(np.random.uniform(0, 1000, n_samples))
    ]
    timestamps = [(t - start_time).total_seconds() * 1e6 for t in timestamps]
    signal_strength = np.random.uniform(-100, 0, n_samples)
    signal_frequency = np.random.uniform(30, 30000, n_samples)
    amplitude = np.random.uniform(0, 10, n_samples)

    df = pd.DataFrame({
        'Signal Duration (microsec)': signal_duration,
        'Azimuthal Angle (degrees)': azimuthal_angle,
        'Elevation Angle (degrees)': elevation_angle,
        'PRI (microsec)': pri,
        'Timestamp (microsec)': timestamps,
        'Signal Strength (dBm)': signal_strength,
        'Signal Frequency (MHz)': signal_frequency,
        'Amplitude': amplitude
    })

    print("Synthetic data generation completed")
    return df

@PipelineDecorator.component(return_values=["preprocessed_data"], cache=True, task_type=Task.TaskTypes.data_processing)
def preprocess_data(initial_dataframe, batch_size=32, shuffle=True):
    print("Starting comprehensive data preprocessing")

    # Initial preprocessing
    dataset = torch.utils.data.TensorDataset(torch.tensor(initial_dataframe.values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    all_data = []
    for batch in dataloader:
        all_data.append(batch[0].numpy())
    features = np.vstack(all_data)
    features_scaled = StandardScaler().fit_transform(features)

    # Convert to PyTorch tensor for advanced preprocessing
    data_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    def add_noise(tensor, noise_level=0.01):
        return tensor + noise_level * torch.randn_like(tensor)

    def apply_constraints(tensor):
        tensor[:, 0] *= 1 + torch.rand_like(tensor[:, 0]) * 0.1 - 0.05  # Signal duration: uniform noise, 4-5%
        tensor[:, 1] += torch.normal(0, 10, tensor[:, 1].shape)  # Azimuthal angle: Gaussian with std. dev = 10 degrees
        tensor[:, 1] = tensor[:, 1] % 360
        tensor[:, 2] += torch.normal(0, 10, tensor[:, 2].shape)  # Elevation angle: Gaussian with std. dev = 10 degrees
        tensor[:, 2] = torch.clamp(tensor[:, 2], -90, 90)
        tensor[:, 6] = 15000 + torch.normal(0, 10, tensor[:, 6].shape)  # Frequency: Gaussian with mean = 15 GHz, std dev = 10 MHz
        tensor[:, 6] = torch.clamp(tensor[:, 6], 6000, 16000)
        tensor[:, 5] += torch.rand_like(tensor[:, 5]) * 3 - 1.5  # Signal strength: ±1.5dB uniform noise
        return tensor

    # Apply advanced preprocessing
    preprocessed_data = add_noise(data_tensor)
    preprocessed_data = apply_constraints(preprocessed_data)

    print("Comprehensive data preprocessing completed successfully")
    return preprocessed_data.numpy()

@PipelineDecorator.component(return_values=["model_name", "scores", "optimal_k"], cache=True, task_type=Task.TaskTypes.training)
def kmeans(preprocessed_data):
    print("Training and evaluating KMeans model")

    task = Task.current_task()
    max_clusters = 10
    inertia = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_model.fit(preprocessed_data)
        inertia.append(kmeans_model.inertia_)
        silhouette_scores.append(silhouette_score(preprocessed_data, kmeans_model.labels_))
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans_model.fit_predict(preprocessed_data)
    scores = calculate_clustering_scores(preprocessed_data, clusters)
    plot_clusters_3d(preprocessed_data, clusters, 'KMeans Clustering with 3D PCA', task)

    for metric, score in scores.items():
        task.get_logger().report_scalar(title="KMeans Clustering Score", series=metric, value=score, iteration=0)

    task.connect({"kmeans_optimal_k": optimal_k})

    return "run_kmeans", scores, optimal_k

@PipelineDecorator.component(return_values=["model_name", "scores", "optimal_k"], cache=True, task_type=Task.TaskTypes.training)
def agglomerative(preprocessed_data):
    print("Training and evaluating Agglomerative Clustering model")
    task = Task.current_task()
    max_clusters = min(10, int(np.sqrt(len(preprocessed_data))))
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg_clustering.fit_predict(preprocessed_data)
        silhouette_scores.append(silhouette_score(preprocessed_data, labels))

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
    labels = agg_clustering.fit_predict(preprocessed_data)

    scores = calculate_clustering_scores(preprocessed_data, labels)
    plot_clusters_3d(preprocessed_data, labels, 'Agglomerative Clustering with 3D PCA', task)

    for metric, score in scores.items():
        task.get_logger().report_scalar(title="Agglomerative Clustering Score", series=metric, value=score, iteration=0)

    task.connect({"agglomerative_optimal_k": optimal_k})

    return "run_agglomerative", scores, optimal_k

@PipelineDecorator.component(return_values=["model_name", "scores", "eps", "min_samples"], cache=True, task_type=Task.TaskTypes.training)
def dbscan(preprocessed_data):
    print("Training and evaluating DBSCAN model")

    task = Task.current_task()
    k = 4
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(preprocessed_data)
    distances, _ = neigh.kneighbors(preprocessed_data)
    sorted_distances = np.sort(distances[:, k-1], axis=0)

    knee_point = np.argmax(np.diff(sorted_distances)) + 1
    eps = sorted_distances[knee_point]
    min_samples = k

    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan_model.fit_predict(preprocessed_data)

    scores = calculate_clustering_scores(preprocessed_data, labels)
    plot_clusters_3d(preprocessed_data, labels, 'DBSCAN Clustering with 3D PCA', task)

    for metric, score in scores.items():
        task.get_logger().report_scalar(title="Dbscan Clustering Score", series=metric, value=score, iteration=0)

    task.connect({"eps": eps, "min_samples": min_samples})

    return "run_dbscan", scores, eps, min_samples

@PipelineDecorator.component(return_values=["model_name", "scores", "optimal_k", "ensemble_type"], cache=True, task_type=Task.TaskTypes.training)
def ensemble(preprocessed_data):
    print("Training and evaluating Ensemble Clustering model")
    def evaluate_clustering_models(features_scaled):
        results = []
        for n_clusters in range(2, 4):
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans_model.fit_predict(features_scaled)
            kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)

            gmm_model = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_model.fit(features_scaled)
            gmm_labels = gmm_model.predict(features_scaled)
            gmm_silhouette = silhouette_score(features_scaled, gmm_labels)

            results.append((n_clusters, kmeans_silhouette, gmm_silhouette))

        return pd.DataFrame(results, columns=['n_clusters', 'kmeans_silhouette', 'gmm_silhouette'])

    def soft_voting_ensemble(features_scaled, n_clusters):
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(features_scaled)

        gmm_model = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_model.fit(features_scaled)
        gmm_labels = gmm_model.predict(features_scaled)

        ensemble_labels = np.round((kmeans_labels + gmm_labels) / 2).astype(int)
        return ensemble_labels

    def majority_voting_ensemble(features_scaled, n_clusters):
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(features_scaled)

        gmm_model = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_model.fit(features_scaled)
        gmm_labels = gmm_model.predict(features_scaled)

        gmm_labels_aligned = align_clusters(kmeans_labels, gmm_labels)
        ensemble_labels = np.where(kmeans_labels == gmm_labels_aligned, kmeans_labels, -1)

        return ensemble_labels

    def stacking_ensemble(features_scaled, n_clusters, n_init=10):
        kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(features_scaled)
        kmeans_distances = kmeans_model.transform(features_scaled)

        gmm_model = GaussianMixture(n_components=n_clusters, n_init=n_init, random_state=42)
        gmm_model.fit(features_scaled)
        gmm_proba = gmm_model.predict_proba(features_scaled)

        meta_features = np.hstack([kmeans_distances, gmm_proba])
        meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_clf.fit(meta_features, kmeans_labels)
        ensemble_labels = meta_clf.predict(meta_features)

        return ensemble_labels

    def align_clusters(kmeans_labels, gmm_labels):
        size = max(kmeans_labels.max(), gmm_labels.max()) + 1
        matrix = np.zeros((size, size), dtype=np.int64)
        for k, g in zip(kmeans_labels, gmm_labels):
            matrix[k, g] += 1
        row_ind, col_ind = linear_sum_assignment(-matrix)
        aligned_labels = np.zeros_like(gmm_labels)
        for i, j in zip(row_ind, col_ind):
            aligned_labels[gmm_labels == j] = i
        return aligned_labels

    task = Task.current_task()
    results = evaluate_clustering_models(preprocessed_data)
    optimal_n = int(results.loc[results[['kmeans_silhouette', 'gmm_silhouette']].mean(axis=1).idxmax(), 'n_clusters'])

    ensemble_methods = [soft_voting_ensemble, majority_voting_ensemble, stacking_ensemble]
    ensemble_scores = []
    ensemble_labels = []

    for method in ensemble_methods:
        labels = method(preprocessed_data, optimal_n)
        score = silhouette_score(preprocessed_data, labels)
        ensemble_scores.append(score)
        ensemble_labels.append(labels)

    best_method_index = int(np.argmax(ensemble_scores))
    best_labels = ensemble_labels[best_method_index]

    scores = calculate_clustering_scores(preprocessed_data, best_labels)
    plot_clusters_3d(preprocessed_data, best_labels, 'Ensemble Clustering with 3D PCA', task)

    for metric, score in scores.items():
        task.get_logger().report_scalar(title="Ensemble Clustering Score", series=metric, value=score, iteration=0)

    task.connect({"ensemble_optimal_k": optimal_n, "ensemble_type": best_method_index + 1})

    return "run_ensemble", scores, optimal_n, best_method_index + 1

@PipelineDecorator.component(return_values=["model_name", "scores", "parameters"], cache=True, task_type=Task.TaskTypes.training)
def optics(preprocessed_data):
    print("Training and evaluating OPTICS model")
    task = Task.current_task()
    min_samples = max(5, int(0.1 * len(preprocessed_data)))
    min_cluster_size = max(5, int(0.05 * len(preprocessed_data)))

    optics_model = OPTICS(min_samples=min_samples, xi=0.05, min_cluster_size=min_cluster_size)
    labels = optics_model.fit_predict(preprocessed_data)

    scores = calculate_clustering_scores(preprocessed_data, labels)
    plot_clusters_3d(preprocessed_data, labels, 'OPTICS Clustering with 3D PCA', task)

    for metric, score in scores.items():
        task.get_logger().report_scalar(title="OPTICS Clustering Score", series=metric, value=score, iteration=0)

    parameters = {
        'min_samples': min_samples,
        'xi': 0.05,
        'min_cluster_size': min_cluster_size
    }
    task.connect(parameters)

    return "run_optics", scores, parameters

@PipelineDecorator.component(return_values=["model_name", "scores", "parameters"], cache=True, task_type=Task.TaskTypes.training)
def hdbscan_clustering(preprocessed_data):
    print("Training and evaluating HDBSCAN model")

    task = Task.current_task()
    min_cluster_size = max(5, int(0.05 * len(preprocessed_data)))
    min_samples = max(5, int(0.05 * len(preprocessed_data)))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(preprocessed_data)

    scores = calculate_clustering_scores(preprocessed_data, labels)
    plot_clusters_3d(preprocessed_data, labels, 'HDBSCAN Clustering with 3D PCA', task)

    for metric, score in scores.items():
        task.get_logger().report_scalar(title="HDBSCAN Clustering Score", series=metric, value=score, iteration=0)

    parameters = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples
    }
    task.connect(parameters)

    return "run_hdbscan", scores, parameters

@PipelineDecorator.component(return_values=["model_name", "scores", "parameters"], cache=True, task_type=Task.TaskTypes.training)
def gmm(preprocessed_data):
    print('Training and evaluating GMM model')
    task = Task.current_task()
    max_components = 10
    silhouette_scores = []
    bic_scores = []
    for k in range(2, max_components + 1):
        gmm_model = GaussianMixture(n_components=k, random_state=42)
        gmm_model.fit(preprocessed_data)
        labels = gmm_model.predict(preprocessed_data)
        silhouette_scores.append(silhouette_score(preprocessed_data, labels))
        bic_scores.append(gmm_model.bic(preprocessed_data))
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    gmm_model = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_model.fit(preprocessed_data)
    labels = gmm_model.predict(preprocessed_data)
    scores = calculate_clustering_scores(preprocessed_data, labels)
    plot_clusters_3d(preprocessed_data, labels, 'Gaussian Mixture Model Clustering with 3D PCA', task)
    for metric, score in scores.items():
        task.get_logger().report_scalar(title='GMM Clustering Score', series=metric, value=score, iteration=0)
    task.connect({'gmm_optimal_k': optimal_k})
    return 'run_gmm', scores, optimal_k

@PipelineDecorator.pipeline(name="Radar Analysis Pipeline", project="CAESAR", version="1.0.0")
def execute_pipeline():
    print("Starting Radar Analysis Pipeline")
    initial_dataframe = synthetic_data()
    preprocessed_data = preprocess_data(initial_dataframe)

    # Execute all clustering models
    kmeans_results = kmeans(preprocessed_data)
    gmm_results = gmm(preprocessed_data)
    agglomerative_results = agglomerative(preprocessed_data)
    dbscan_results = dbscan(preprocessed_data)
    ensemble_results = ensemble(preprocessed_data)
    optics_results = optics(preprocessed_data)
    hdbscan_results = hdbscan_clustering(preprocessed_data)

    results = {
        kmeans_results[0]: {"scores": kmeans_results[1], "optimal_k": kmeans_results[2]},
        gmm_results[0]: {"scores": gmm_results[1], "optimal_k": gmm_results[2]},
        agglomerative_results[0]: {"scores": agglomerative_results[1], "optimal_k": agglomerative_results[2]},
        dbscan_results[0]: {"scores": dbscan_results[1], "eps": dbscan_results[2], "min_samples": dbscan_results[3]},
        ensemble_results[0]: {"scores": ensemble_results[1], "optimal_k": ensemble_results[2], "ensemble_type": ensemble_results[3]},
        optics_results[0]: {"scores": optics_results[1], "parameters": optics_results[2]},
        hdbscan_results[0]: {"scores": hdbscan_results[1], "parameters": hdbscan_results[2]}
    }

    print("Model evaluation scores:")
    for model_name, result in results.items():
        print(f"{model_name}:")
        if "optimal_k" in result:
            print(f"  Optimal k/components: {result['optimal_k']}")
        if "eps" in result:
            print(f"  Eps: {result['eps']}")
            print(f"  Min samples: {result['min_samples']}")
        if "ensemble_type" in result:
            print(f"  Ensemble type: {result['ensemble_type']}")
        if "parameters" in result:
            print(f"  Parameters: {result['parameters']}")
        print(f"  Scores: {result['scores']}")
        print()

    return results

if __name__ == "__main__":
    try:
        PipelineDecorator.run_locally()
        results = execute_pipeline()
        print("Process completed")
        print("Final results:", results)
    except Exception as e:
        print(f"An error occurred: {str(e)}")