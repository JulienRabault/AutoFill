
import os
import json
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import warnings
import math


class TextToHDF5Converter:
    def __init__(self, dataframe, data_dir, output_dir, pad_size=90, hdf_cache=100000,
                 final_output_file='final_output.h5', exclude=['path', 'researcher', 'date', 'material']):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.pad_size = pad_size
        self.hdf_cache = hdf_cache
        self.metadata_cols = [col for col in dataframe.columns if col not in exclude]
        self.conversion_dict = {col: {} for col in self.metadata_cols if dataframe[col].dtype == object}
        self.hdf_data = self._initialize_hdf_data()
        self.hdf_files = None
        self.final_output_file = final_output_file
        self.exclude = exclude

    def _initialize_hdf_data(self):
        return [
            np.zeros((self.hdf_cache, self.pad_size)),  # data_q
            np.zeros((self.hdf_cache, self.pad_size)),  # data_y
            np.zeros((self.hdf_cache,)),  # original len
            np.zeros((self.hdf_cache,)),  # csv index
            np.zeros((self.hdf_cache,)),  # pair index
            {col: np.zeros((self.hdf_cache,)) for col in self.metadata_cols}  # Métadonnées
        ]

    def _create_hdf(self, output_file):
        hdf = h5py.File(output_file, "w")
        hdf.create_dataset("data_q", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("data_y", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("len", (1,), maxshape=(None,))
        hdf.create_dataset("csv_index", (1,), maxshape=(None,))
        hdf.create_dataset("pair_index", (1,), maxshape=(None,))

        # Création des datasets pour chaque métadonnée
        for col in self.metadata_cols:
            hdf.create_dataset(col, (1,), maxshape=(None,), dtype=np.float64)

        return hdf

    def _flush_into_hdf5(self, current_index, current_size):
        """Sauvegarde les données en batch dans le fichier HDF5"""
        self.hdf_files["data_q"].resize((current_index + current_size, self.pad_size))
        self.hdf_files["data_q"][current_index:current_index + current_size, :] = self.hdf_data[0][:current_size, :]
        self.hdf_files["data_y"].resize((current_index + current_size, self.pad_size))
        self.hdf_files["data_y"][current_index:current_index + current_size, :] = self.hdf_data[1][:current_size, :]
        self.hdf_files["len"].resize((current_index + current_size,))
        self.hdf_files["len"][current_index:current_index + current_size] = self.hdf_data[2][:current_size]
        self.hdf_files["csv_index"].resize((current_index + current_size,))
        self.hdf_files["csv_index"][current_index:current_index + current_size] = self.hdf_data[3][:current_size]
        self.hdf_files["pair_index"].resize((current_index + current_size,))
        self.hdf_files["pair_index"][current_index:current_index + current_size] = self.hdf_data[4][:current_size]

        for col in self.metadata_cols:
            self.hdf_files[col].resize((current_index + current_size,))
            self.hdf_files[col][current_index:current_index + current_size] = self.hdf_data[5][col][:current_size]

        self.hdf_files.flush()

    def _load_data_from_file(self, file_path):
        normalized_path = os.path.normpath(file_path.replace('\\', os.sep).replace('/', os.sep))
        full_path = normalized_path

        if not os.path.exists(full_path):
            warnings.warn(f"Fichier manquant: {full_path}")
            return [], []

        data_q = []
        data_y = []
        expected_num_columns = 2

        with open(full_path, 'r', encoding="utf-8" ) as f:
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
        return np.array(data_q, dtype=np.float64), np.array(data_y, dtype=np.float64)

    def _pad_data(self, data_q, data_y):
        """Applique du padding ou tronque les séries temporelles"""
        if len(data_q) < self.pad_size:
            data_q = np.pad(data_q, (0, self.pad_size - len(data_q)), 'constant')
            data_y = np.pad(data_y, (0, self.pad_size - len(data_y)), 'constant')
        elif len(data_q) > self.pad_size:
            data_q = data_q[:self.pad_size]
            data_y = data_y[:self.pad_size]
        return data_q, data_y

    def _convert_metadata(self, row):
        """Convertit les métadonnées en valeurs numériques"""
        converted = {}
        for col in self.metadata_cols:
            value = row[col]

            if pd.isna(value) or value is None:  # Vérifie si la valeur est NaN ou None
                converted[col] = -1
                continue  # Passe à la colonne suivante

            if isinstance(value, str):  # Si c'est une chaîne de caractères (catégorique)
                if value not in self.conversion_dict[col]:
                    self.conversion_dict[col][value] = len(self.conversion_dict[col])
                converted[col] = self.conversion_dict[col][value]
            else:  # Si c'est un nombre (int ou float)
                converted[col] = float(value)

        return converted

    def convert(self):
        output_file = os.path.join(self.output_dir, self.final_output_file)
        self.hdf_files = self._create_hdf(output_file)
        current_size = 0
        current_index = 0

        with tqdm(total=len(self.dataframe), desc="Processing files") as pbar:
            for idx, row in self.dataframe.iterrows():
                pbar.update(1)
                file_path = os.path.join(self.data_dir, row['path'])
                try:
                    data_q, data_y = self._load_data_from_file(file_path)
                    original_len = len(data_q)
                    data_q, data_y = self._pad_data(data_q, data_y)
                except Exception as e:
                    data_q, data_y = np.zeros((self.pad_size,)), np.zeros((self.pad_size,))
                    original_len = 0
                    warnings.warn(f"Erreur fichier {row['path']}: {e}")

                metadata = self._convert_metadata(row)
                self.hdf_data[0][current_size] = data_q
                self.hdf_data[1][current_size] = data_y
                self.hdf_data[2][current_size] = original_len
                self.hdf_data[3][current_size] = idx
                self.hdf_data[4][current_size] = (row['pair_index'] if 'pair_index' in row else -1)
                for col in self.metadata_cols:
                    self.hdf_data[5][col][current_size] = metadata[col]
                current_size += 1

                if current_size == self.hdf_cache:
                    self._flush_into_hdf5(current_index, current_size)
                    current_index += self.hdf_cache
                    current_size = 0

        if current_size > 0:
            self._flush_into_hdf5(current_index, current_size)
        self.hdf_files.close()

        with open("conversion_dict.json", "w") as f:
            json.dump(self.conversion_dict, f)

        print("Conversion terminée, dictionnaire sauvegardé.")


if __name__ == "__main__":
    data_csv_path = '/mnt/data/WORK/AUTOFILL_data/datav2bis/metadatabis.csv'
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"CSV file not found: {data_csv_path}")

    dataframe = pd.read_csv(data_csv_path)
    # dataframe_c = (dataframe[(dataframe['technique'] == 'les')])
    # dataframe_c = dataframe_c[(dataframe['material'] == 'ag')].sample(frac=0.1)
    data_dir = '/mnt/data/WORK/AUTOFILL_data/datav2bis/SAXS_100k_Curve_Au_Ag/SAXS_100k_Curve_Au_Ag/'
    final_output_file = 'databis.h5'
    print(dataframe.columns)
    print(dataframe.head())
    converter = TextToHDF5Converter(dataframe=dataframe, data_dir=data_dir, output_dir='./', final_output_file=final_output_file)
    converter.convert()
    print(f"Data successfully converted to {final_output_file}")
