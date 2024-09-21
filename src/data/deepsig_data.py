# Library Imports
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RadioSignalDataset(Dataset):
    """Defines the DeepSig RadioML Dataset"""
    def __init__(self, file_path, num_classes=24):
        np.random.seed(37)
        with h5py.File(file_path, 'r') as data:
            self.features = data['X'][:]
            self.labels = np.argmax(data['Y'][:], axis=1)
            self.snr = data['Z'][:]

        self.num_classes = min(num_classes, 24)
        self.class_indices = np.random.choice(24, self.num_classes, replace=False)

        mask = np.isin(self.labels, self.class_indices)
        self.features = self.features[mask]
        self.labels = self.labels[mask]
        self.snr = self.snr[mask]

        self.class_map = {old: new for new, old in enumerate(self.class_indices)}
        self.labels = np.array([self.class_map[y] for y in self.labels])

        self.group_examples()

    def group_examples(self):
        self.grouped_examples = {i: (self.labels == i).nonzero()[0] for i in range(self.num_classes)}
        for i, examples in self.grouped_examples.items():
            print(f"Modulation type {i}: {len(examples)} examples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        anchor = self.features[index].reshape(1024, 2)
        anchor_label = self.labels[index]

        positive_index = random.choice(self.grouped_examples[anchor_label])
        while positive_index == index:
            positive_index = random.choice(self.grouped_examples[anchor_label])
        positive = self.features[positive_index].reshape(1024, 2)

        negative_label = random.choice([i for i in range(self.num_classes) if i != anchor_label])
        negative_index = random.choice(self.grouped_examples[negative_label])
        negative = self.features[negative_index].reshape(1024, 2)

        return torch.from_numpy(anchor).float(), torch.from_numpy(positive).float(), torch.from_numpy(negative).float(), torch.tensor(anchor_label).long()

    def get_stratified_split(self, test_size=0.2, random_state=42):
        return train_test_split(
            range(len(self.features)),
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )

def load_and_split_data(file_path, num_classes=5, test_size=0.2, random_state=42):
    dataset = RadioSignalDataset(file_path, num_classes=num_classes)
    train_indices, test_indices = dataset.get_stratified_split(test_size=test_size, random_state=random_state)
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)

def print_class_distribution(dataset):
    labels = [dataset[i][3].item() for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples")