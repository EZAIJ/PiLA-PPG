from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _compute_derivatives(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vpg = np.gradient(data, axis=1)
    apg = np.gradient(vpg, axis=1)
    return vpg, apg


def _zscore_per_sample(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def make_feature_columns(start: int, end: int) -> List[str]:
    return [str(i) for i in range(start, end + 1)]


def load_pretrain_data(csv_path: str, label_column: str, feature_start: int, feature_end: int):
    df = pd.read_csv(csv_path)
    feature_columns = make_feature_columns(feature_start, feature_end)
    data = df[feature_columns].values.astype(np.float32)
    labels = df[label_column].values
    return data, labels


def load_finetune_data(csv_path: str, label_as: str, label_ar: str, feature_start: int, feature_end: int):
    df = pd.read_csv(csv_path)
    feature_columns = make_feature_columns(feature_start, feature_end)
    data = df[feature_columns].values.astype(np.float32)
    labels_as = df[label_as].values
    labels_ar = df[label_ar].values
    return data, labels_as, labels_ar


class PPGDatasetSingleLabel(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data_ppg = data
        self.data_vpg, self.data_apg = _compute_derivatives(data)
        self.data_vpg = _zscore_per_sample(self.data_vpg)
        self.data_apg = _zscore_per_sample(self.data_apg)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data_ppg)

    def __getitem__(self, index: int):
        ppg_item = torch.tensor(self.data_ppg[index], dtype=torch.float32).unsqueeze(0)
        vpg_item = torch.tensor(self.data_vpg[index], dtype=torch.float32).unsqueeze(0)
        apg_item = torch.tensor(self.data_apg[index], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return (ppg_item, vpg_item, apg_item), label


class PPGDatasetMultiLabel(Dataset):
    def __init__(self, data: np.ndarray, labels_as: np.ndarray, labels_ar: np.ndarray):
        self.data_ppg = data
        self.data_vpg, self.data_apg = _compute_derivatives(data)
        self.data_vpg = _zscore_per_sample(self.data_vpg)
        self.data_apg = _zscore_per_sample(self.data_apg)
        self.labels_as = labels_as
        self.labels_ar = labels_ar

    def __len__(self) -> int:
        return len(self.data_ppg)

    def __getitem__(self, index: int):
        ppg_item = torch.tensor(self.data_ppg[index], dtype=torch.float32).unsqueeze(0)
        vpg_item = torch.tensor(self.data_vpg[index], dtype=torch.float32).unsqueeze(0)
        apg_item = torch.tensor(self.data_apg[index], dtype=torch.float32).unsqueeze(0)
        label_as = torch.tensor(self.labels_as[index], dtype=torch.float32)
        label_ar = torch.tensor(self.labels_ar[index], dtype=torch.float32)
        return (ppg_item, vpg_item, apg_item), label_as, label_ar
