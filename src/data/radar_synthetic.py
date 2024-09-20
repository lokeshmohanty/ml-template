"""
Synthetic Radar Data Generator and Loader Module.

This module provides functionality to generate synthetic radar data
and load it using PyTorch's Dataset and DataLoader classes.

Classes:
    RadarDataset: A custom Dataset class for synthetic radar data.

Functions:
    get_dataloader: Creates and returns a DataLoader for the RadarDataset.
"""

# from typing import List
# from src.config import (
#     np, pd, torch, Dataset, DataLoader, datetime, timedelta
# )
# class RadarDataset(Dataset):
#     """
#     A custom Dataset class for synthetic radar data.

#     This class generates synthetic radar data and provides methods
#     to access it as a PyTorch Dataset.

#     Attributes:
#         data (pd.DataFrame): The generated synthetic radar data.
#     """

#     def __init__(self, n_samples: int = 1000):
#         """
#         Initialize the RadarDataset.

#         Args:
#             n_samples (int): Number of samples to generate. Default is 1000.
#         """
#         self.data = self.generate_radar_data(n_samples)

#     def __len__(self) -> int:
#         """
#         Get the number of samples in the dataset.

#         Returns:
#             int: The number of samples in the dataset.
#         """
#         return len(self.data)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         """
#         Get a sample from the dataset.

#         Args:
#             idx (int): The index of the sample to retrieve.

#         Returns:
#             torch.Tensor: The sample at the given index.
#         """
#         return torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)

#     @staticmethod
#     def generate_radar_data(n_samples: int = 1000) -> pd.DataFrame:
#         """
#         Generate synthetic radar data.

#         Args:
#             n_samples (int): Number of samples to generate. Default is 1000.

#         Returns:
#             pd.DataFrame: A DataFrame containing the generated radar data.
#         """
#         np.random.seed(42)  # For reproducibility of the results

#         signal_duration = np.random.uniform(1e-6, 1e-3, n_samples) * 1e6
#         azimuthal_angle = np.random.uniform(0, 360, n_samples)
#         elevation_angle = np.random.uniform(-90, 90, n_samples)
#         pri = np.random.uniform(1e-3, 1, n_samples) * 1e6
#         start_time = datetime.now()
#         timestamps: List[float] = [
#             start_time + timedelta(microseconds=int(x))
#             for x in np.cumsum(np.random.uniform(0, 1000, n_samples))
#         ]
#         timestamps = [(t - start_time).total_seconds() * 1e6 for t in timestamps]
#         signal_strength = np.random.uniform(-100, 0, n_samples)
#         signal_frequency = np.random.uniform(30, 30000, n_samples)
#         amplitude = np.random.uniform(0, 10, n_samples)

#         df = pd.DataFrame({
#             'Signal Duration (microsec)': signal_duration,
#             'Azimuthal Angle (degrees)': azimuthal_angle,
#             'Elevation Angle (degrees)': elevation_angle,
#             'PRI (microsec)': pri,
#             'Timestamp (microsec)': timestamps,
#             'Signal Strength (dBm)': signal_strength,
#             'Signal Frequency (MHz)': signal_frequency,
#             'Amplitude': amplitude
#         })

#         return df

# def get_dataloader(batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
#     """
#     Create a DataLoader for the RadarDataset.

#     Args:
#         batch_size (int): Number of samples per batch. Default is 32.
#         shuffle (bool): Whether to shuffle the data. Default is True.
#         num_workers (int): Number of subprocesses to use for data loading. Default is 4.

#     Returns:
#         DataLoader: A PyTorch DataLoader for the RadarDataset.
#     """
    
#     dataset = RadarDataset()
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# from typing import List
# from src.config import (
#     np, pd, torch, Dataset, DataLoader, datetime, timedelta
# )
# from clearml import Dataset as ClearMLDataset, StorageManager
# import os

# class RadarDataset(torch.utils.data.Dataset):
#     def __init__(self, n_samples: int = 1000):
#         self.data = self.generate_radar_data(n_samples)
#         self.data_path = self.save_data()

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         sample = self.data.iloc[idx]
#         return torch.tensor(sample.values, dtype=torch.float32)

#     @staticmethod
#     def generate_radar_data(n_samples: int = 1000) -> pd.DataFrame:
#         np.random.seed(42)  # For reproducibility of the results

#         signal_duration = np.random.uniform(1e-6, 1e-3, n_samples) * 1e6
#         azimuthal_angle = np.random.uniform(0, 360, n_samples)
#         elevation_angle = np.random.uniform(-90, 90, n_samples)
#         pri = np.random.uniform(1e-3, 1, n_samples) * 1e6
#         start_time = datetime.now()
#         timestamps: List[float] = [
#             start_time + timedelta(microseconds=int(x))
#             for x in np.cumsum(np.random.uniform(0, 1000, n_samples))
#         ]
#         timestamps = [(t - start_time).total_seconds() * 1e6 for t in timestamps]
#         signal_strength = np.random.uniform(-100, 0, n_samples)
#         signal_frequency = np.random.uniform(30, 30000, n_samples)
#         amplitude = np.random.uniform(0, 10, n_samples)

#         df = pd.DataFrame({
#             'Signal Duration (microsec)': signal_duration,
#             'Azimuthal Angle (degrees)': azimuthal_angle,
#             'Elevation Angle (degrees)': elevation_angle,
#             'PRI (microsec)': pri,
#             'Timestamp (microsec)': timestamps,
#             'Signal Strength (dBm)': signal_strength,
#             'Signal Frequency (MHz)': signal_frequency,
#             'Amplitude': amplitude
#         })

#         return df

#     def save_data(self) -> str:
#         file_path = 'radar_data.csv'
#         self.data.to_csv(file_path, index=False)
#         return file_path

# def upload_dataset(data_path: str):
#     # Create a new dataset in ClearML
#     clearml_dataset = ClearMLDataset.create(
#         dataset_name='Radar Dataset',
#         dataset_project='Synthetic Radar Data'
#     )
#     # Add data file
#     clearml_dataset.add_files(data_path)
#     # Upload the dataset
#     clearml_dataset.upload(output_url='s3://your-bucket/path')  # Specify your storage
#     clearml_dataset.finalize()

# def get_dataloader(batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
#     dataset = RadarDataset()
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# if __name__ == "__main__":
#     # Generate and load data
#     radar_data_loader = get_dataloader()

#     # Optional: Run through the data once (for demonstration)
#     for data in radar_data_loader:
#         print(data.shape)

#     # Assuming data is saved in the RadarDataset initialization
#     dataset = RadarDataset()
#     upload_dataset(dataset.data_path)

from typing import List
from clearml import Dataset, StorageManager
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datetime import datetime, timedelta

class RadarDataset(TorchDataset):
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