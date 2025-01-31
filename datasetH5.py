import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, pad_size=None, normalize=False):
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r')
        self.data_q = self.hdf['data_q']
        self.data_y = self.hdf['data_y']
        self.metadata = self.hdf['metadata']
        self.lengths = self.hdf['len']
        # invert  self.csv_index
        self.csv_index = self.hdf['csv_index']
        # self.csv_index = {value: index for index, value in enumerate(self.csv_index)}
        self.pad_size = pad_size
        self.normalize = normalize

    def __len__(self):
        return self.data_q.shape[0]

    def __getitem__(self, idx):

        data_q = torch.tensor(self.data_q[idx], dtype=torch.float32)
        data_y = torch.tensor(self.data_y[idx], dtype=torch.float32)
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)
        csv_file_index = torch.tensor(self.csv_index[idx])

        if self.normalize:
            data_q, data_y = self._normalize_data(data_q, data_y)
        if self.pad_size:
            data_q, data_y = self._pad_data(data_q, data_y)

        return data_q, data_y, metadata, csv_file_index

    def close(self):
        self.hdf.close()

    def _normalize_data(self, data_q, data_y):
        min_q = torch.min(data_q)
        max_q = torch.max(data_q)
        if max_q > min_q:
            data_q = (data_q - min_q) / (max_q - min_q)
        else:
            data_q = torch.zeros_like(data_q)

        return data_q, data_y

    def _pad_data(self, data_q, data_y):
        def pad(data, pad_size):
            num_rows = data.size(0)
            if num_rows < pad_size:
                padding_rows = pad_size - num_rows
                pad_tensor = torch.zeros(padding_rows, dtype=torch.float32)
                return torch.cat([data, pad_tensor], dim=0)
            return data[:pad_size]

        padded_q = pad(data_q, self.pad_size)
        padded_y = pad(data_y, self.pad_size)

        return padded_q, padded_y

def load_data_from_file(file_path):
    data_q, data_y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', 'Q', 'q', 'Q;', 'q;')):
                continue
            tokens = line.replace(';', ' ').replace(',', ' ').split()
            try:
                values = [float(token) if token.lower() != 'nan' else float('nan') for token in tokens]
            except ValueError:
                continue
            if len(values) == 2:
                values = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in values]
                data_q.append(values[0])
                data_y.append(values[1])
    return np.array(data_q), np.array(data_y)


if __name__ == "__main__":
    hdf5_file = 'data.h5'
    csv_file = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'
    data_dir = '../AUTOFILL_data/datav2/Base_de_donnee'
    pad_size = 2000

    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    dataset = HDF5Dataset(hdf5_file=hdf5_file, pad_size=pad_size)
    indices_to_test = [0, 1, 2, 3, 4]

    for idx in indices_to_test:
        data_q_h5, data_y_h5, metadata, csv_file_index = dataset[idx]
        file_path = os.path.join(data_dir, df.iloc[int(csv_file_index.item())]['path'])
        normalized_path = os.path.normpath(file_path.replace('\\', os.sep).replace('/', os.sep))

        if not os.path.exists(normalized_path):
            print(f"Fichier introuvable : {normalized_path}")
            continue

        data_q_txt, data_y_txt = load_data_from_file(normalized_path)

        match_q = np.allclose(data_q_h5.numpy()[:len(data_q_txt)], data_q_txt, atol=1e-6)
        match_y = np.allclose(data_y_h5.numpy()[:len(data_y_txt)], data_y_txt, atol=1e-6)

        print(f"Index {idx}:")
        print(f"Index csv: {int(csv_file_index.item())}")
        print(f"Path: {normalized_path}")
        print(f"data_q_h5: {data_q_h5}")
        print(f"data_q_txt: {data_q_txt}")
        print(f"data_y_h5: {data_y_h5}")
        print(f"data_y_txt: {data_y_txt}")
        print(f"Match Q: {match_q}")
        print(f"Match Y: {match_y}")
        print("=" * 40)

    dataset.close()
