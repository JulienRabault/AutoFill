import math
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from collections import defaultdict, Counter
from tqdm import tqdm


class CustomDatasetVAE(Dataset):
    def __init__(self, dataframe, data_dir='/projects/pnria/DATA/AUTOFILL/',
                 pad_size=80, cache_limit=250000):
        self.dataframe = dataframe
        self.categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        self.numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']
        self.data_dir = data_dir
        self.pad_size = pad_size
        self.cache_limit = cache_limit

        self.cat_vocab = self._build_cat_vocab()
        self.metadata_tensor = self._preprocess_metadata()
        self.data_cache = OrderedDict()
        self.timing_stats = {'metadata': 0.0, 'load_data': 0.0, 'processing': 0.0}
        self.sample_count = 0

        self._display_dataframe_info()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        metadata = self.metadata_tensor[idx]
        relative_path = self.dataframe.iloc[idx]['path']
        file_path = os.path.join(self.data_dir, relative_path)

        if file_path in self.data_cache:
            data_q, data_y = self.data_cache.pop(file_path)
            self.data_cache[file_path] = (data_q, data_y)
        else:
            try:
                q, y = self._load_data_from_file(file_path)
            except Exception as e:
                warnings.warn(f"Erreur fichier {file_path}: {e}")
                q = y = torch.zeros(self.pad_size)
            self.data_cache[file_path] = (q, y)
            if len(self.data_cache) > self.cache_limit:
                self.data_cache.popitem(last=False)

        data_q, data_y = self.data_cache[file_path]
        data_q, data_y = self._normalize_data(data_q, data_y)
        data_q, data_y = self._pad_data(data_q, data_y)
        return {"data_q": data_q, "data_y": data_y, "metadata": metadata, "csv_index": idx}

    def _build_cat_vocab(self):
        return {
            col: {val: idx for idx, val in enumerate(self.dataframe[col].astype(str).unique())}
            for col in self.categorical_cols
        }

    def _preprocess_metadata(self):
        cat_data = [
            self.dataframe[col].astype(str).map(self.cat_vocab[col]).fillna(-1).astype(float)
            for col in self.categorical_cols
        ]
        num_data = self.dataframe[self.numerical_cols].fillna(0.0).astype(float)
        combined = pd.concat([pd.DataFrame(cat_data).T, num_data], axis=1)
        return torch.tensor(combined.values, dtype=torch.float32)

    def _load_data_from_file(self, file_path):
        normalized_path = os.path.normpath(file_path.replace('\\', os.sep).replace('/', os.sep))
        try:
            df = pd.read_csv(
                normalized_path,
                comment='#',
                sep='[;,\\s]+',
                engine='python',
                usecols=[0, 1],
                names=['q', 'y'],
                header=None,
                dtype=np.float32
            )
        except Exception as e:
            raise ValueError(f"Read error ({file_path}): {e}")
        df = df.dropna().replace([np.inf, -np.inf], 0.0)
        return (
            torch.tensor(df['q'].values, dtype=torch.float32),
            torch.tensor(df['y'].values, dtype=torch.float32)
        )

    def _normalize_data(self, q, y):
        if len(q) > 0:
            q_min, q_max = q.min(), q.max()
            if q_max > q_min:
                q = (q - q_min) / (q_max - q_min)
        return q, y

    def _pad_data(self, q, y):
        def _pad(tensor):
            return torch.nn.functional.pad(
                tensor[:self.pad_size],
                (0, max(0, self.pad_size - len(tensor))),
                mode='constant',
                value=0
            )
        return _pad(q), _pad(y)

    def _display_dataframe_info(self):
        print("\nInformations sur le Dataset :")
        print("==================================================")
        print(f"- Nombre total d'échantillons : {len(self.dataframe)}")
        print(f"- Taille maximale du cache : {self.cache_limit}")
        print(f"- Colonnes catégoriques : {self.categorical_cols}")
        print(f"- Colonnes numériques : {self.numerical_cols}")
        print("\nStatistiques des colonnes numériques :")
        print(self.dataframe[self.numerical_cols].describe().round(2))
        print("==================================================\n")


    def display_data_y_shapes(self):
        """
        Count occurrences of each data_y length per 'shape' category.
        """
        shape_length_counts = defaultdict(Counter)

        for idx in tqdm(range(len(self.dataframe)), desc="Processing files"):
            row = self.dataframe.iloc[idx]
            relative_path = row['path']
            shape_category = str(row['shape']).lower()
            file_path = os.path.join(self.data_dir, relative_path)

            try:
                _, y = self._load_data_from_file(file_path)
                shape_length_counts[shape_category][len(y)] += 1
            except Exception:
                continue

        print("\nDistribution des longueurs de data_y par 'shape':")
        for shape_cat, length_counter in shape_length_counts.items():
            print(f"\nShape: {shape_cat}")
            for length, count in sorted(length_counter.items()):
                print(f"  Length {length}: {count} sample(s)")


if __name__ == "__main__":
    data_csv_path = "/projects/pnria/DATA/AUTOFILL/all_data_v1.csv"
    data_dir = "/projects/pnria/DATA/AUTOFILL/"
    df = pd.read_csv(data_csv_path)
    df = df[df["technique"].astype(str).str.lower() == "saxs"].reset_index(drop=True)
    dataset = CustomDatasetVAE(dataframe=df, data_dir=data_dir)
    print("####### saxs #######")
    dataset.display_data_y_shapes()