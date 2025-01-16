import h5py
import pandas as pd
import os
import warnings
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_txt_to_hdf5(dataframe, data_dir='../datav2/Base_de_donnee', output_file='data.h5', flush_interval=10000):
    """
    Convertit les fichiers texte listés dans le dataframe en un fichier HDF5.

    :param dataframe: DataFrame contenant les chemins et métadonnées
    :param data_dir: Chemin du répertoire contenant les fichiers texte
    :param output_file: Nom du fichier de sortie HDF5
    :param flush_interval: Nombre de fichiers avant chaque flush dans le fichier HDF5
    """
    # Création du fichier HDF5
    f = h5py.File(output_file, 'w')
    metadata_group = f.create_group('metadata')
    data_group = f.create_group('data')

    categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
    numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']

    # Stockage des métadonnées
    cat_vocab = {}
    for col in categorical_cols:
        cat_vocab[col] = dataframe[col].astype(str).unique().tolist()
        metadata_group.create_dataset(col, data=[s.encode('utf-8') for s in cat_vocab[col]])

    for col in numerical_cols:
        metadata_group.create_dataset(col, data=dataframe[col].values)

    # Traitement des fichiers texte
    buffer = []
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing files"):
        relative_path = row['path']
        try:
            data_q, data_y = _load_data_from_file(os.path.join(data_dir, relative_path))

            dataset_id = f'data_{idx}'
            buffer.append((f'{dataset_id}_q', data_q))
            buffer.append((f'{dataset_id}_y', data_y))

            # Flush du buffer
            if len(buffer) >= flush_interval:
                _flush_to_hdf5(data_group, buffer)
                buffer = []

        except Exception as e:
            warnings.warn(f"Error loading file {relative_path}: {e}")
            continue

    # Flush final
    if buffer:
        _flush_to_hdf5(data_group, buffer)

    f.close()


def _flush_to_hdf5(data_group, buffer):
    """
    Enregistre les données du buffer dans le fichier HDF5.

    :param data_group: Groupe de données dans le fichier HDF5
    :param buffer: Liste de tuples (nom du dataset, données)
    """
    for dataset_name, data in buffer:
        data_group.create_dataset(dataset_name, data=data)


def _load_data_from_file(file_path):
    """
    Charge les données depuis un fichier texte.

    :param file_path: Chemin complet du fichier texte
    :return: Deux listes : data_q et data_y
    """
    normalized_path = os.path.normpath(file_path.replace('\\', os.sep).replace('/', os.sep))
    full_path = normalized_path

    if not os.path.exists(full_path):
        warnings.warn(f"File missing: {full_path}")
        return [], []

    data_q = []
    data_y = []
    with open(full_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(('#', 'Q', 'q', 'Q;', 'q;')):
                continue
            tokens = line.split()
            if len(tokens) == 2:
                try:
                    q_value = float(tokens[0])
                    y_value = float(tokens[1])

                    data_q.append(q_value)
                    data_y.append(y_value)
                except ValueError:
                    warnings.warn(f"Error converting line: {line}")
                    continue
    return data_q, data_y


def main():
    """
    Point d'entrée principal du programme.
    """
    data_csv_path = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"CSV file not found: {data_csv_path}")
    dataframe = pd.read_csv(data_csv_path)

    required_columns = ['path', 'material', 'type', 'method', 'shape', 'researcher', 'technique', 'concentration',
                        'opticalPathLength', 'd', 'h']
    if not all(col in dataframe.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    logger.info("Starting conversion of .txt files to HDF5...")
    convert_txt_to_hdf5(dataframe, data_dir='../AUTOFILL_data/datav2/Base_de_donnee', output_file='data.h5')
    logger.info("Conversion complete.")


if __name__ == '__main__':
    main()
