from typing import List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from clearml import Dataset

class RadarDataset(TorchDataset):
    """A dataset class for synthetic radar data."""

    def __init__(self, n_samples: int = 1000):
        self.data = self.generate_radar_data(n_samples)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)

    @staticmethod
    def generate_radar_data(n_samples: int = 1000) -> pd.DataFrame:
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

        return df

def get_dataloader(batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    dataset = RadarDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def create_and_upload_dataset(dataset_name: str, dataset_project: str, n_samples: int = 1000):
    # Create a ClearML dataset
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=dataset_project)

    # Generate synthetic radar data
    radar_data = RadarDataset.generate_radar_data(n_samples)

    # Save the data to a CSV file
    csv_filename = f"{dataset_name}.csv"
    radar_data.to_csv(csv_filename, index=False)

    # Add the CSV file to the dataset
    dataset.add_files(csv_filename)

    # Upload the dataset
    dataset.upload()

    # Finalize the dataset
    dataset.finalize()

    print(f"Dataset '{dataset_name}' has been created and uploaded to the ClearML server.")

if __name__ == "__main__":
    # Example usage
    create_and_upload_dataset(
        dataset_name="synthetic_radar_data",
        dataset_project="radar_analysis",
        n_samples=10000
    )