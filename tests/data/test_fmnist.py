import sys
import os
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.fmnist import FMNIST

def test_init():
    dataset = FMNIST()
    assert isinstance(dataset, Dataset)

def test_len():
    assert True

def test_getitem():
    assert True
