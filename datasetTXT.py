import math
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDatasetVAE(Dataset):
    def __init__(self, dataframe, data_dir='../datav2/Base_de_donnee',
                 pad_size=80, enable_timing=False, cache_limit=250000):
        self.dataframe = dataframe
        self.categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        self.numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']
        self.data_dir = data_dir
        self.pad_size = pad_size
        self.enable_timing = enable_timing
        self.cache_limit = cache_limit

        # Pré-traitement des métadonnées
        self.cat_vocab = self._build_cat_vocab()
        self.metadata_tensor = self._preprocess_metadata()

        # Cache LRU pour les données chargées
        self.data_cache = OrderedDict()

        # Stockage des temps pour calcul des moyennes
        self.timing_stats = {'metadata': 0.0, 'load_data': 0.0, 'processing': 0.0}
        self.sample_count = 0

        # Affichage des infos du dataset
        self._display_dataframe_info()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        timers = {'start': time.perf_counter()} if self.enable_timing else None

        # Récupération des métadonnées pré-calculées
        metadata = self.metadata_tensor[idx]
        if self.enable_timing:
            timers['metadata'] = time.perf_counter()

        # Chargement des données avec cache LRU
        relative_path = self.dataframe.iloc[idx]['path']
        file_path = os.path.join(self.data_dir, relative_path)

        # Gestion du cache
        if file_path in self.data_cache:
            # Mise à jour de la position LRU
            data_q, data_y = self.data_cache.pop(file_path)
            self.data_cache[file_path] = (data_q, data_y)
        else:
            try:
                q, y = self._load_data_from_file(file_path)
            except Exception as e:
                warnings.warn(f"Erreur fichier {file_path}: {e}")
                q = y = torch.zeros(self.pad_size)

            # Ajout au cache avec éviction LRU si nécessaire
            self.data_cache[file_path] = (q, y)
            if len(self.data_cache) > self.cache_limit:
                self.data_cache.popitem(last=False)

        data_q, data_y = self.data_cache[file_path]

        if self.enable_timing:
            timers['load_data'] = time.perf_counter()

        # Post-processing
        data_q, data_y = self._normalize_data(data_q, data_y)
        data_q, data_y = self._pad_data(data_q, data_y)

        if self.enable_timing:
            timers['processing'] = time.perf_counter()
            self._update_timing_stats(timers)

        return {"data_q": data_q, "data_y": data_y, "metadata": metadata, "csv_index": idx}

    def _build_cat_vocab(self):
        """Construction vectorisée du vocabulaire catégoriel"""
        return {
            col: {val: idx for idx, val in enumerate(self.dataframe[col].astype(str).unique())}
            for col in self.categorical_cols
        }

    def _preprocess_metadata(self):
        """Pré-traitement vectorisé des métadonnées"""
        # Traitement des catégoriques
        cat_data = [
            self.dataframe[col].astype(str).map(self.cat_vocab[col]).fillna(-1).astype(float)
            for col in self.categorical_cols
        ]

        # Traitement des numériques
        num_data = self.dataframe[self.numerical_cols].fillna(0.0).astype(float)

        # Combinaison des données
        combined = pd.concat([pd.DataFrame(cat_data).T, num_data], axis=1)
        return torch.tensor(combined.values, dtype=torch.float32)

    def _load_data_from_file(self, file_path):
        """Chargement optimisé avec pandas"""
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
        """Normalisation vectorisée"""
        if len(q) > 0:
            q_min, q_max = q.min(), q.max()
            if q_max > q_min:
                q = (q - q_min) / (q_max - q_min)
        return q, y

    def _pad_data(self, q, y):
        """Padding optimisé"""

        def _pad(tensor):
            return torch.nn.functional.pad(
                tensor[:self.pad_size],
                (0, max(0, self.pad_size - len(tensor))),
                mode='constant',
                value=0
            )

        return _pad(q), _pad(y)

    def _update_timing_stats(self, timers):
        """Mise à jour des statistiques de temps moyen"""
        self.sample_count += 1
        self.timing_stats['metadata'] += timers['metadata'] - timers['start']
        self.timing_stats['load_data'] += timers['load_data'] - timers['metadata']
        self.timing_stats['processing'] += timers['processing'] - timers['load_data']

        if self.sample_count % 100 == 0:
            self._print_timing_summary()

    def _print_timing_summary(self):
        """Affichage des moyennes des temps de traitement"""
        print("\nMoyennes des temps de traitement (par échantillon) :")
        print("==================================================")
        for stage, total_time in self.timing_stats.items():
            print(f"- {stage.ljust(12)} : {total_time / self.sample_count:.6f} s")
        print("==================================================\n")

    def _display_dataframe_info(self):
        """Affichage des informations du dataset"""
        print("\nInformations sur le Dataset :")
        print("==================================================")
        print(f"- Nombre total d'échantillons : {len(self.dataframe)}")
        print(f"- Taille maximale du cache : {self.cache_limit}")
        print(f"- Colonnes catégoriques : {self.categorical_cols}")
        print(f"- Colonnes numériques : {self.numerical_cols}")

        # Statistiques sur les valeurs numériques
        print("\nStatistiques des colonnes numériques :")
        print(self.dataframe[self.numerical_cols].describe().round(2))
        print("==================================================\n")