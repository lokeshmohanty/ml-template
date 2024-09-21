import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import torch, StandardScaler, BATCH_SIZE
from model.kmeans import KMeansClusterer
from model.dbscan import DBSCANClusterer
from model.gmm import GMMClusterer
from model.ensemble import EnsembleClusterer
from model.agglomerative import AgglomerativeClusterer
from model.optics import OPTICSClusterer
from model.hdbscan_clusterer import HDBSCANClusterer
from data.radar_synthetic import get_dataloader
from model.siamese import SiameseNetwork

def run_selected_model(args):
    model_classes = {
        'kmeans': KMeansClusterer,
        'dbscan': DBSCANClusterer,
        'gmm': GMMClusterer,
        'ensemble': EnsembleClusterer,
        'agglomerative': AgglomerativeClusterer,
        'optics': OPTICSClusterer,
        'hdbscan': HDBSCANClusterer,
        'siamese': SiameseNetwork
    }

    model_class = model_classes.get(args.model)
    if not model_class:
        raise ValueError(f"Unsupported model: {args.model}")

    model = model_class()
    dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=True)

    all_data = []
    for batch in dataloader:
        if isinstance(batch, dict) and 'data' in batch:
            all_data.append(batch['data'])
        else:
            all_data.append(batch)

    features = torch.cat(all_data, dim=0).numpy()
    features_scaled = StandardScaler().fit_transform(features)

    results = model.run(None, features_scaled)
    print(results)

    if hasattr(model, 'close_task'):
        model.close_task()

def main():
    parser = argparse.ArgumentParser(description='Clustering and Siamese Network for Radar Signal Classification')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', type=str,
                        choices=['kmeans', 'dbscan', 'gmm', 'ensemble', 'agglomerative', 'optics', 'hdbscan', 'siamese'],
                        required=True,
                        help='Choose the model to run')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                        help='number of classes for classification (default: 2)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    args = parser.parse_args()
    run_selected_model(args)

if __name__ == "__main__":
    main()