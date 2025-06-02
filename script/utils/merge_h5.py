import h5py
import numpy as np


def merge_hdf5(file1, file2, output_file):
    """Fusionne deux fichiers HDF5 en un seul."""
    with h5py.File(file1, "r") as hdf1, h5py.File(file2, "r") as hdf2, h5py.File(output_file, "w") as hdf_out:
        for key in hdf1.keys():
            # Lire les données des deux fichiers
            data1 = hdf1[key][:]
            data2 = hdf2[key][:]

            # Fusionner les données
            merged_data = np.concatenate([data1, data2], axis=0)

            # Créer le dataset avec la bonne shape
            hdf_out.create_dataset(key, data=merged_data, maxshape=(None,) + merged_data.shape[1:], dtype=data1.dtype)


merge_hdf5("data.h5", "data_saxs.h5", "all_data.hdf5")
