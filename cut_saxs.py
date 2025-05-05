import numpy as np
from matplotlib import pyplot as plt
from dataset.datasetH5 import HDF5Dataset
from dataset.transformations import SequentialTransformer

hdf5_file = "/mnt/data/WORK/AUTOFILL_data/AUTOFILL_data/all_data_saxs_v2.h5"
conversion_dict_path = "/mnt/data/WORK/AUTOFILL_data/AUTOFILL_data/all_data_saxs_v2.json"

transform = {
    'y': {
        'MinMaxNormalizer': {},
        'PaddingTransformer': {
            'pad_size': 50,
            'value': 0}
    },
    'q': {
        'PaddingTransformer': {
            'pad_size': 50,
            'value': 0}}}

dataset = HDF5Dataset(
    hdf5_file=hdf5_file,
    metadata_filters={"technique": ["saxs"]},
    conversion_dict_path=conversion_dict_path,
    sample_frac=1,
    requested_metadata=["material", "shape"],
    transform=transform
)

sample = dataset[10]
q_values = sample["data_q"].numpy()
y_values = sample["data_y"].numpy()
print(f"q_values: {len([y for y in y_values[0] if y>0])}")
plt.figure(figsize=(10, 6))
plt.loglog(q_values, y_values, linestyle='-', marker='o', markersize=5, color='blue', alpha=0.7)
plt.title('Log-Log Plot of y vs q')
plt.xlabel('q')
plt.ylabel('y')
plt.grid(True)
plt.show()

dataset.close()
