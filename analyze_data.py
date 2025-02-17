import os

import numpy as np
import pandas as pd
import torch

from datasetH5 import HDF5Dataset
from datasetTXT import CustomDatasetVAE

data_csv_path = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'
if not os.path.exists(data_csv_path):
    raise FileNotFoundError(f"Le fichier CSV spécifié est introuvable: {data_csv_path}")
dataframe = pd.read_csv(data_csv_path)
print(f"Nombre de données totales: {len(dataframe)}")
dataframe_LES = (dataframe[(dataframe['technique'] == 'les')])
dataframe_LES = dataframe_LES[(dataframe_LES['material'] == 'ag')].sample(frac=0.1)

print(f"Nombre de données LES: {len(dataframe_LES)}")
pad_size = 80

custom_dataset = CustomDatasetVAE(dataframe=dataframe_LES, data_dir='../AUTOFILL_data/datav2/Base_de_donnee', pad_size=pad_size)
custom_data = [custom_dataset[i] for i in range(10)]

hdf5_file = 'data_pad_90.h5'
pad_size = 80
conversion_dict_path = 'conversion_dict.json'
metadata_filters={"technique": ["les"], "material": ["ag"]}

dataset = HDF5Dataset(hdf5_file=hdf5_file, pad_size=pad_size, metadata_filters=metadata_filters,
                      conversion_dict_path=conversion_dict_path, frac=0.1)
hdf5_dataset = HDF5Dataset(hdf5_file=hdf5_file, pad_size=pad_size, metadata_filters=metadata_filters, conversion_dict_path=conversion_dict_path, frac=0.1)
hdf5_data = [hdf5_dataset[i] for i in range(10)]

# Fonction pour comparer les éléments
def compare_elements(custom_element, hdf5_element):
    data_q_txt, data_y_txt = custom_element[0], custom_element[1]
    data_q_h5, data_y_h5 = hdf5_element[0], hdf5_element[1]

    match_q = np.allclose(data_q_h5.numpy()[:len(data_q_txt)], data_q_txt, atol=1e-6)
    match_y = np.allclose(data_y_h5.numpy()[:len(data_y_txt)], data_y_txt, atol=1e-6)

    print(f"data_q_h5: {data_q_h5}")
    print(f"data_q_txt: {data_q_txt}")
    print(f"data_y_h5: {data_y_h5}")
    print(f"data_y_txt: {data_y_txt}")
    print(f"Match Q: {match_q}")
    print(f"Match Y: {match_y}")
    print("=" * 40)

# Comparer les 10 premiers éléments
for i in range(10):
    if not compare_elements(custom_data[i], hdf5_data[i]):
        print(f"Les éléments à l'index {i} sont différents.")
    else:
        print(f"Les éléments à l'index {i} sont identiques.")