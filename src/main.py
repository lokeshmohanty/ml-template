import sys
import os
import argparse
from typing import Dict, Any, Tuple

# Add the parent directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    torch, pd, DataLoader, StandardScaler, BATCH_SIZE, 
    StepLR, optim, load_and_split_data, print_class_distribution
)
from src.model.kmeans import KMeansClusterer
from src.model.dbscan import DBSCANClusterer
from src.model.gmm import GMMClusterer
from src.model.ensemble import EnsembleClusterer
from src.model.agglomerative import AgglomerativeClusterer
from src.model.optics import OPTICSClusterer
from src.model.hdbscan_clusterer import HDBSCANClusterer
from src.data.radar_synthetic import get_dataloader
from src.model.siamese import SiameseNetwork

class ClusteringModelSelector:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.models: Dict[str, Any] = {
            'kmeans': KMeansClusterer(),
            'dbscan': DBSCANClusterer(),
            'gmm': GMMClusterer(),
            'ensemble': EnsembleClusterer(),
            'agglomerative': AgglomerativeClusterer(),
            'optics': OPTICSClusterer(),
            'hdbscan': HDBSCANClusterer(),
        }
        self.scaler = StandardScaler()

    def prepare_data(self) -> Tuple[pd.DataFrame, Any]:
        all_data = []
        for batch in self.dataloader:
            all_data.append(batch)
        all_data = torch.cat(all_data, dim=0)
        
        data_np = all_data.numpy()
        
        columns = [
            'Signal Duration (microsec)', 'Azimuthal Angle (degrees)',
            'Elevation Angle (degrees)', 'PRI (microsec)', 'Timestamp (microsec)',
            'Signal Strength (dBm)', 'Signal Frequency (MHz)', 'Amplitude'
        ]
        df = pd.DataFrame(data_np, columns=columns)
        
        features_scaled = self.scaler.fit_transform(data_np)
        
        return df, features_scaled

    def select_model(self, model_name: str) -> None:
        df, features_scaled = self.prepare_data()
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        results = self.models[model_name].run(df, features_scaled)

        print(f"Results for {model_name}:")
        if 'scores' in results:
            for metric, score in results['scores'].items():
                print(f"{metric}: {score:.2f}")
        else:
            for metric, score in results.items():
                print(f"{metric}: {score:.2f}")

        if 'optimal_k' in results:
            print(f"Optimal number of clusters: {results['optimal_k']}")

        if 'ensemble_type' in results:
            ensemble_types = {1: "Soft Voting", 2: "Majority Voting", 3: "Stacking"}
            print(f"Ensemble type: {ensemble_types[results['ensemble_type']]}")

def run_siamese_network(args, device, train_loader, test_loader):
    siamesemodel = SiameseNetwork(args.num_classes).to(device)
    optimizer = optim.Adadelta(siamesemodel.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        siamesemodel.train_epoch(args,siamesemodel,device, train_loader, optimizer, epoch)
        siamesemodel.test(device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(siamesemodel.state_dict(), "siamese_radar_signal.pt")

def main() -> None:
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
    parser.add_argument('--model', type=str, choices=['clustering', 'siamese'], required=True,
                        help='Choose between clustering and siamese network')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                        help='number of classes for classification (default: 2)')
    
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == 'clustering':
        dataloader = get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
        model_selector = ClusteringModelSelector(dataloader=dataloader)

        print("Select a clustering model:")
        model_names = list(model_selector.models.keys())
        for i, model_name in enumerate(model_names, start=1):
            print(f"{i}. {model_name}")

        while True:
            choice = input("Enter the number corresponding to your choice: ")
            try:
                choice = int(choice)
                if 1 <= choice <= len(model_names):
                    selected_model = model_names[choice - 1]
                    break
                print(f"Invalid choice. Please enter a number between 1 and {len(model_names)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

        model_selector.select_model(selected_model)

    elif args.model == 'siamese':
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # Get the directory of the main.py file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root
        project_root = os.path.dirname(current_dir)
        # Construct the full path to the data file
        data_file_path = os.path.join(project_root, 'GOLD_XYZ_OSC.0001_1024.hdf5')
        train_dataset, test_dataset = load_and_split_data(data_file_path, num_classes=args.num_classes)
        
        print("Train set distribution:")
        print_class_distribution(train_dataset)
        print("\nTest set distribution:")
        print_class_distribution(test_dataset)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        run_siamese_network(args, device, train_loader, test_loader)

if __name__ == "__main__":
    main()