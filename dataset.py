import os
import warnings

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F

class CustomDatasetVAE(Dataset):
    """Dataset personnalisé qui gère les données de diffusion et les métadonnées
    avec des encodages catégoriels, numériques, et normalisation des données.
    """

    def __init__(self, dataframe, data_dir='../AUTOFILL_data/datav2/Base_de_donnee', pad_size=80):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame contenant les informations des données.
            data_dir (str): Répertoire contenant les fichiers de données.
            pad_size (int): Taille maximale pour les vecteurs de diffusion.
        """
        self.dataframe = dataframe
        self.categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        self.numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']
        self.data_dir = data_dir
        self.pad_size = pad_size
        self.cat_vocab = self._build_cat_vocab(self.dataframe, self.categorical_cols)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        relative_path = row['path']
        # Chargement et normalisation des données principales
        try:
            data = self._load_data_from_file(os.path.join(self.data_dir, relative_path))
            data = self._pad_data(data)
            x_q, y = data[:, 0], data[:, 1]  # Séparer X(q) et Y
        except Exception as e:
            x_q = torch.zeros(self.pad_size, dtype=torch.float32)
            y = torch.zeros(self.pad_size, dtype=torch.float32)
            warnings.warn(f"Erreur lors du chargement du fichier {relative_path}: {e}")
        # Traitement des métadonnées
        cat_features, num_features = self._process_metadata(row)
        x_q_normalized = self._normalize_x_q(x_q)
        return (y, cat_features, num_features, x_q_normalized)

    def _build_cat_vocab(self, df, categorical_cols):
        """Construit le vocabulaire pour chaque variable catégorielle."""
        return {
            col: {val: idx for idx, val in enumerate(df[col].astype(str).unique())}
            for col in categorical_cols
        }

    def _process_metadata(self, row):
        """Encode les variables catégorielles et numériques en tenseurs."""
        cat_features = [self.cat_vocab[col].get(str(row[col]), -1) for col in self.categorical_cols]
        cat_features = torch.tensor(cat_features, dtype=torch.long)
        num_features = [row[col] if not pd.isnull(row[col]) else 0.0 for col in self.numerical_cols]
        num_features = torch.tensor(num_features, dtype=torch.float32)
        return cat_features, num_features

    def _normalize_x_q(self, x_q):
        """Normalise les valeurs de X(q) entre 0 et 1."""
        min_val = torch.min(x_q)
        max_val = torch.max(x_q)
        return (x_q - min_val) / (max_val - min_val + 1e-8)

    def _load_data_from_file(self, file_path):
        """Charge les données depuis un fichier texte avec gestion des valeurs manquantes."""
        normalized_path = file_path.replace('\\', '/' )
        file_path = normalized_path
        if not os.path.exists(file_path):
            warnings.warn(f"Fichier manquant: {file_path}")
            return torch.empty((0, 2), dtype=torch.float32)

        data = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = self._parse_line(line)
                if tokens:
                    data.append(tokens)
        if not data:
            raise ValueError(f"Pas de données valides trouvées dans: {file_path}")

        data = torch.tensor(data, dtype=torch.float32)
        # Normalisation des colonnes pour éviter les ordres de grandeur trop différents
        data[:, 1] = (data[:, 1] - torch.mean(data[:, 1])) / (torch.std(data[:, 1]) + 1e-8)
        return data

    def get_cat_vocab_sizes(self):
        """Retourne la taille du vocabulaire pour chaque variable catégorielle."""
        return [len(self.cat_vocab[col]) for col in self.categorical_cols]

    def _parse_line(self, line):
        """Parse une ligne de données en gestionnant les différents séparateurs."""
        line = line.strip()
        if not line or line.startswith(('#', 'Q', 'q', 'Q;', 'q;')):
            return None
        if ';' in line:
            tokens = line.split(';')
        elif ',' in line:
            tokens = line.split(',')
        else:
            tokens = line.split()
        try:
            return [float(token) if token.lower() != 'nan' else 0.0 for token in tokens]
        except ValueError:
            return None

    def _pad_data(self, data):
        """Uniformise la taille des données avec du padding ou de la troncature."""
        num_rows, num_features = data.shape
        if num_rows < self.pad_size:
            pad_tensor = torch.zeros((self.pad_size - num_rows, num_features), dtype=torch.float32)
            return torch.cat([data, pad_tensor], dim=0)
        return data[:self.pad_size, :]

print(os.path.exists('../AUTOFILL_data/datav2/Base_de_donnee/Au/Cylindre/Simulation/LES'))