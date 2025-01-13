import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset


class DatasetHDF5(Dataset):
    """Dataset personnalisé pour charger les données depuis un fichier HDF5."""

    def __init__(self, hdf5_file, dataframe, categorical_cols, numerical_cols, pad_size=80):
        """
        Args:
            hdf5_file (str): Chemin du fichier HDF5.
            dataframe (pd.DataFrame): DataFrame contenant les informations des fichiers.
            categorical_cols (list): Liste des colonnes catégorielles.
            numerical_cols (list): Liste des colonnes numériques.
            pad_size (int): Taille maximale à laquelle chaque échantillon sera padé.
        """
        self.hdf5_file = hdf5_file
        self.dataframe = dataframe
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.pad_size = pad_size

        with h5py.File(self.hdf5_file, 'r') as f:
            self.data_q = f['data']
            self.metadata = f['metadata']

        self.cat_vocab = {col: {val: idx for idx, val in enumerate(self.metadata[col][:])} for col in
                          self.categorical_cols}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        dataset_id = f'data_{idx}'

        metadata = self._process_metadata(row)
        data_q = torch.tensor(self.data_q[f'{dataset_id}_q'][:], dtype=torch.float32)
        data_y = torch.tensor(self.data_q[f'{dataset_id}_y'][:], dtype=torch.float32)

        data_q, data_y = self._pad_data(data_q, data_y)

        return data_q, data_y, metadata

    def _process_metadata(self, row):
        """Encode les variables catégorielles et numériques en un tenseur."""
        cat_features = [self.cat_vocab[col].get(str(row[col]), -1) for col in self.categorical_cols]
        num_features = [row[col] if not pd.isnull(row[col]) else 0.0 for col in self.numerical_cols]
        return torch.tensor(cat_features + num_features, dtype=torch.float32)

    def _pad_data(self, data_q, data_y):
        """Applique le padding ou la troncature sur deux listes de données (Q et Y)."""

        def pad(data):
            num_rows = data.size(0)
            if num_rows < self.pad_size:
                padding_rows = self.pad_size - num_rows
                pad_tensor = torch.zeros(padding_rows, dtype=torch.float32)
                return torch.cat([data, pad_tensor], dim=0)
            elif num_rows > self.pad_size:
                return data[:self.pad_size]
            return data

        padded_q = pad(data_q)
        padded_y = pad(data_y)

        return padded_q, padded_y
