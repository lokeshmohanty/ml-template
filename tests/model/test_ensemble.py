"""
Test module for Ensemble Clustering.

This module contains unit tests for the EnsembleClusterer class
from the src.model.ensemble module.

Imports:
    - sys, os: For modifying Python path
    - numpy (as np): For numerical operations
    - pandas (as pd): For DataFrame operations
    - pytest: For test fixtures and assertions
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - EnsembleClusterer from src.model.ensemble: The class being tested
"""
import sys
import os

from src.config import (
    np, pd, pytest, BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.ensemble import EnsembleClusterer
from tests.conftest import mock_task
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

@pytest.fixture
def dataloader():
    """Create a DataLoader for test data."""
    return get_dataloader(batch_size=BATCH_SIZE, shuffle=True)

@pytest.fixture
def features_scaled(dataloader):
    """Prepare scaled feature data for clustering."""
    all_data = []
    for batch in dataloader:
        all_data.append(batch)
    return np.concatenate(all_data, axis=0)

def test_ensemble_init(mock_task):
    """Test the initialization of EnsembleClusterer."""
    ensemble = EnsembleClusterer(task=mock_task)
    assert hasattr(ensemble, 'max_clusters')
    assert ensemble.max_clusters > 0

def test_ensemble_run(features_scaled, mock_task):
    """Test the run method of EnsembleClusterer."""
    ensemble = EnsembleClusterer(task=mock_task)
    results = ensemble.run(None, features_scaled)
    # Mock the get_logger method
    mock_task.get_logger.return_value = None

    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']

    assert 'optimal_k' in results
    assert results['optimal_k'] == int(results['optimal_k']), "optimal_k is not an integer-like value"
    assert isinstance(results['optimal_k'], (int, np.integer)), "optimal_k is not an int or numpy integer"
    assert results['optimal_k'] > 0

    assert 'ensemble_type' in results
    assert isinstance(results['ensemble_type'], int), "ensemble_type is not a Python int"
    assert 1 <= results['ensemble_type'] <= 3

def test_ensemble_evaluate_clustering_models(features_scaled, mock_task):
    """Test the evaluate_clustering_models method of EnsembleClusterer."""
    ensemble = EnsembleClusterer(task=mock_task)
    results = ensemble.evaluate_clustering_models(features_scaled)
    assert isinstance(results, pd.DataFrame)
    assert 'n_clusters' in results.columns
    assert 'kmeans_silhouette' in results.columns
    assert 'gmm_silhouette' in results.columns

def test_ensemble_soft_voting(features_scaled, mock_task):
    """Test the soft_voting_ensemble method of EnsembleClusterer."""
    ensemble = EnsembleClusterer(task=mock_task)
    labels = ensemble.soft_voting_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)

def test_ensemble_majority_voting(features_scaled, mock_task):
    """Test the majority_voting_ensemble method of EnsembleClusterer."""
    ensemble = EnsembleClusterer(task=mock_task)
    labels = ensemble.majority_voting_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)

def test_ensemble_stacking(features_scaled, mock_task):
    """Test the stacking_ensemble method of EnsembleClusterer."""
    ensemble = EnsembleClusterer(task=mock_task)
    labels = ensemble.stacking_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)