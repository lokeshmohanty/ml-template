"""
Test module for Agglomerative Clustering.

This module contains unit tests for the AgglomerativeClusterer class
from the src.model.agglomerative module.

Imports:
    - sys, os: For modifying Python path
    - numpy (as np): For numerical operations
    - pytest: For test fixtures and assertions
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - AgglomerativeClusterer from src.model.agglomerative: The class being tested
"""
import sys
import os
from src.config import ( 
    np,pytest,BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.agglomerative import AgglomerativeClusterer
from tests.conftest import mock_task

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

@pytest.fixture(scope='session')
def dataloader():
    """Create a DataLoader for test data."""
    return get_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

@pytest.fixture
def features_scaled(dataloader):
    """Prepare scaled feature data for clustering."""
    all_data = []
    for batch in dataloader:
        all_data.append(batch)
    return np.concatenate(all_data, axis=0)

def test_agglomerative_init(mock_task):
    """Test the initialization of AgglomerativeClusterer."""
    agglomerative = AgglomerativeClusterer(task=mock_task)
    assert hasattr(agglomerative, 'run')

def test_agglomerative_run(features_scaled, mock_task):
    """Test the run method of AgglomerativeClusterer."""
    agglomerative = AgglomerativeClusterer(task=mock_task)
    
    # Mock the get_logger method
    mock_task.get_logger.return_value = None
    
    results = agglomerative.run(None, features_scaled)

    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']

    assert 'labels' in results
    assert isinstance(results['labels'], np.ndarray)
    assert len(results['labels']) == len(features_scaled)

    assert 'optimal_k' in results
    assert isinstance(results['optimal_k'], int)
    assert results['optimal_k'] > 0