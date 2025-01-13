import h5py
import pandas as pd
import os
import warnings
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_txt_to_hdf5(dataframe, data_dir='../datav2/Base de donnée', output_file='data.h5'):
    with h5py.File(output_file, 'w') as f:
        metadata_group = f.create_group('metadata')
        data_group = f.create_group('data')

        categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']

        cat_vocab = {}
        for col in categorical_cols:
            cat_vocab[col] = dataframe[col].astype(str).unique().tolist()
            metadata_group.create_dataset(col, data=cat_vocab[col])

        for col in numerical_cols:
            metadata_group.create_dataset(col, data=dataframe[col].values)

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing files"):
            relative_path = row['path']
            try:
                data_q, data_y = _load_data_from_file(os.path.join(data_dir, relative_path))

                dataset_id = f'data_{idx}'

                data_group.create_dataset(f'{dataset_id}_q', data=data_q)
                data_group.create_dataset(f'{dataset_id}_y', data=data_y)

            except Exception as e:
                warnings.warn(f"Error loading file {relative_path}: {e}")
                continue


def _load_data_from_file(file_path):
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