import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataset.transformations import SequentialTransformer


class TXTDataset(Dataset):
    def __init__(self, dataframe, data_dir='/projects/pnria/DATA/AUTOFILL/',
                 transform=None, cache_limit=250000):
        self.dataframe = dataframe
        self.categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        self.numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']
        self.data_dir = data_dir
        self.cache_limit = cache_limit
        self.transformer_q = SequentialTransformer(transform.get("q", None))
        self.transformer_y = SequentialTransformer(transform.get("y", None))

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
        data_y_min = data_y.min()
        data_y_max = data_y.max()
        data_q = self.transformer_q.fit_transform(data_q)
        data_y = self.transformer_y.fit_transform(data_y)
        data_q = torch.as_tensor(data_q, dtype=torch.float32)
        data_y = torch.as_tensor(data_y, dtype=torch.float32)
        return {"data_q": data_q.unsqueeze(0), "data_y": data_y.unsqueeze(0), "data_y_min": data_y_min,
                "data_y_max": data_y_max,
                "metadata": metadata, "csv_index": idx, "path": file_path}

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
