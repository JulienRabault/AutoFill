import math
import os
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDatasetVAE(Dataset):
    """Dataset personnalisé qui gère l'encodage de variables catégorielles, numériques
    et charge des données à partir de fichiers texte.
    Il gère également les fichiers contenant des 'nan' en les remplaçant par 0.0.
    Ajout de la fonctionnalité de padding pour uniformiser la taille des échantillons.
    """

    def __init__(self, dataframe, data_dir='../datav2/Base_de_donnee', pad_size=80):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame contenant les données.
            data_dir (str): Répertoire contenant les fichiers de données.
            pad_size (int): Taille maximale à laquelle chaque échantillon sera padé.
        """
        self.dataframe = dataframe
        self.categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        self.numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']
        self.data_dir = data_dir
        self.pad_size = pad_size
        self.cat_vocab = self._build_cat_vocab(self.dataframe, self.categorical_cols)

        # Affichage des informations du DataFrame
        self._display_dataframe_info()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        relative_path = row['path']
        metadata = self._process_metadata(row)
        try:
            data_q, data_y = self._load_data_from_file(os.path.join(self.data_dir, relative_path))
            data_q, data_y = self._normalize_data(data_q, data_y)
            data_q, data_y = self._pad_data(data_q, data_y)
        except Exception as e:
            data_q = torch.zeros((self.pad_size,), dtype=torch.float32)
            data_y = torch.zeros((self.pad_size,), dtype=torch.float32)
            warnings.warn(f"Erreur lors du chargement du fichier {relative_path}: {e}")
        return data_q, data_y, metadata

    def _build_cat_vocab(self, df, categorical_cols):
        """Construit le vocabulaire pour chaque variable catégorielle."""
        return {
            col: {val: idx for idx, val in enumerate(df[col].astype(str).unique())}
            for col in categorical_cols
        }

    def _process_metadata(self, row):
        """Encode les variables catégorielles et numériques en un tenseur."""
        cat_features = [self.cat_vocab[col].get(str(row[col]), -1) for col in self.categorical_cols]
        num_features = [row[col] if not pd.isnull(row[col]) else 0.0 for col in self.numerical_cols]
        return torch.tensor(cat_features + num_features, dtype=torch.float32)

    def _load_data_from_file(self, file_path):
        """Charge les données d'un fichier texte et retourne deux listes séparées."""
        normalized_path = os.path.normpath(file_path.replace('\\', os.sep).replace('/', os.sep))
        full_path = normalized_path

        if not os.path.exists(full_path):
            warnings.warn(f"Fichier manquant: {full_path}")
            return [], []

        data_q = []  # Liste pour la première colonne
        data_y = []  # Liste pour la deuxième colonne
        expected_num_columns = 2  # S'assurer qu'on attend deux colonnes

        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', 'Q', 'q', 'Q;', 'q;')):
                    continue

                if ';' in line:
                    tokens = line.split(';')
                elif ',' in line:
                    tokens = line.split(',')
                else:
                    tokens = line.split()

                try:
                    values = [float(token) if token.lower() != 'nan' else float('nan') for token in tokens if token != '']
                except ValueError:
                    continue

                if len(values) != expected_num_columns:
                    continue

                values = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in values]

                if len(values) == expected_num_columns:
                    data_q.append(values[0])
                    data_y.append(values[1])

        if not data_q or not data_y:
            raise ValueError(f"Pas de données valides trouvées dans: {full_path}")

        return data_q, data_y

    def _normalize_data(self, data_q, data_y):
        """
        Normalise deux listes de données (Q et Y) entre 0 et 1.

        Args:
            data_q (list): Liste des données de la première colonne (Q).
            data_y (list): Liste des données de la deuxième colonne (Y).

        Returns:
            torch.Tensor, torch.Tensor: Tenseurs normalisés pour Q et Y.
        """
        data_q = torch.tensor(data_q, dtype=torch.float32)
        data_y = torch.tensor(data_y, dtype=torch.float32)

        def normalize(data):
            min_val = torch.min(data)
            max_val = torch.max(data)
            range_val = max_val - min_val
            range_val = range_val if range_val != 0 else 1.0
            return (data - min_val) / range_val

        normalized_q = normalize(data_q)
        normalized_y = normalize(data_y)

        return normalized_q, normalized_y

    def _pad_data(self, data_q, data_y):
        """
        Applique le padding ou la troncature sur deux listes de données (Q et Y) pour qu'elles aient une taille uniforme.

        Args:
            data_q (list): Liste des données de la première colonne (Q).
            data_y (list): Liste des données de la deuxième colonne (Y).

        Returns:
            torch.Tensor, torch.Tensor: Tenseurs padés à la taille `pad_size`.
        """

        def pad(data, pad_size):
            num_rows = data.size(0)
            if num_rows < pad_size:
                padding_rows = pad_size - num_rows
                pad_tensor = torch.zeros(padding_rows, dtype=torch.float32)
                return torch.cat([data, pad_tensor], dim=0)
            elif num_rows > pad_size:
                return data[:pad_size]
            return data

        padded_q = pad(data_q, self.pad_size)
        padded_y = pad(data_y, self.pad_size)

        return padded_q, padded_y

    def _display_dataframe_info(self):
        """Affiche les informations du DataFrame pour aider au débogage et à la validation."""
        print("\n=== Informations du DataFrame ===")
        print(f"Nombre de lignes : {len(self.dataframe)}")
        print("Colonnes disponibles :", self.dataframe.columns.tolist())
        print("Aperçu des premières lignes :")
        print(self.dataframe.head())
        print("=================================\n")
