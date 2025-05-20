import json
import os
import warnings

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset.transformations import *


class PairHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, metadata_filters=None, conversion_dict_path=None,
                 sample_frac=1, requested_metadata=[],
                 transformer_q_saxs=Pipeline(),
                transformer_y_saxs=Pipeline(),
                transformer_q_les=Pipeline(),
                transformer_y_les=Pipeline(),
                 **kwargs):
        """
        Optimized PyTorch Dataset for HDF5 files with selective metadata preprocessing
        """
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r', swmr=True)

        required_keys = ['data_q_saxs', 'data_y_saxs', 'data_q_les', 'data_y_les', 'csv_index']
        missing = [k for k in required_keys if k not in self.hdf]
        if missing:
            raise RuntimeError(
                f"Missing required datasets in HDF5 file: {missing}\n"
                "Your HDF5 file is not compatible with PairVAE. "
                "Refer to the README (section 5) and generate it using scripts/05_pair_txtTOhdf5.py."
            )

        self.data_q_saxs = self.hdf['data_q_saxs']
        self.data_y_saxs = self.hdf['data_y_saxs']
        self.data_q_les = self.hdf['data_q_les']
        self.data_y_les = self.hdf['data_y_les']
        self.csv_index = self.hdf['csv_index']

        self.transformer_q_saxs = transformer_q_saxs
        self.transformer_y_saxs = transformer_y_saxs
        self.transformer_q_les = transformer_q_les
        self.transformer_y_les = transformer_y_les

        all_metadata_cols = [col for col in self.hdf.keys() if col not in
                             ['data_q_saxs', 'data_y_saxs', 'data_q_les', 'data_y_les', 'len', 'csv_index']]
        self.metadata_datasets = {col: self.hdf[col] for col in all_metadata_cols}

        self.requested_metadata = self._validate_requested_metadata(requested_metadata, all_metadata_cols)
        self.conversion_dict = self._load_conversion_dict(conversion_dict_path)

        # Filters
        self.metadata_filters = metadata_filters or {}
        self.filtered_indices = self._apply_metadata_filters()

        # Validate and apply data fraction
        self._validate_frac(sample_frac)
        self.sample_frac = sample_frac
        if 0 < sample_frac < 1:
            self._apply_data_fraction(sample_frac)

        # Print initialization info
        self._print_init_info()

        self.transformer_y_saxs.fit(self.data_y_saxs[self.filtered_indices])
        self.transformer_y_les.fit(self.data_y_les[self.filtered_indices])

    def _print_init_info(self):
        """Print dataset initialization information"""
        print("\n╒══════════════════════════════════════════════╕")
        print("│ Dataset Initialization Info                 │")
        print("╞══════════════════════════════════════════════╡")
        print(f"│ File: {self.hdf5_file:<35} │")
        print(f"│ Total samples: {len(self.data_q_saxs):<26} │")
        print(f"│ Samples filtered: {len(self.filtered_indices):<23} │")
        print(f"│ Requested fraction: {self.sample_frac:<22} │")
        print(f"│ Fractioned samples: {len(self.filtered_indices):<22} │")
        print(f"│ Requested metadata: {len(self.requested_metadata):<22} │")

    def _validate_requested_metadata(self, requested, available):
        """Validate and filter requested metadata columns"""
        if requested is None:
            return available

        valid = [col for col in requested if col in available]
        missing = set(requested) - set(valid)

        if missing:
            warnings.warn(f"Missing requested metadata columns: {missing}")

        return valid

    def _validate_frac(self, sample_frac):
        """Validate data fraction parameter"""
        if not (0 < sample_frac <= 1):
            raise ValueError("Data fraction must be between 0 and 1")

    def _load_conversion_dict(self, path):
        """Load JSON conversion dictionary for categorical metadata"""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _apply_metadata_filters(self):
        """Vectorized metadata filtering using numpy operations"""
        if not self.metadata_filters:
            return list(range(len(self.data_q_saxs)))

        mask = np.ones(len(self.data_q_saxs), dtype=bool)

        for key, allowed_values in tqdm(self.metadata_filters.items(),
                                        desc="Applying filters"):
            if key not in self.metadata_datasets:
                mask[:] = False
                break

            data = self.metadata_datasets[key][...]

            if key in self.conversion_dict:
                converted_allowed = [self.conversion_dict[key].get(str(v), -1)
                                     for v in allowed_values]
                key_mask = np.isin(data, converted_allowed)
            else:
                key_mask = np.isin(data, allowed_values)

            mask &= key_mask

        return np.where(mask)[0]

    def _apply_data_fraction(self, sample_frac):
        """Select a fraction of the filtered indices in sorted order"""
        num_samples = int(len(self.filtered_indices) * sample_frac)
        self.filtered_indices = self.filtered_indices[:num_samples]

    def __len__(self):
        return len(self.filtered_indices)

    def _get_metadata(self, idx):
        """Preprocess requested metadata to tensors during initialization"""
        metadata = {}
        for col in self.requested_metadata:
            data = self.metadata_datasets[col][idx]
            metadata[col] = data
        return metadata

    def __getitem__(self, idx):
        """Optimized data loading with timing and progress tracking"""
        # Get original dataset index
        original_idx = self.filtered_indices[idx]

        # Load main data
        data_q_saxs = self.data_q_saxs[original_idx]
        data_y_saxs = self.data_y_saxs[original_idx]
        data_q_les = self.data_q_les[original_idx]
        data_y_les = self.data_y_les[original_idx]

        # Get preprocessed metadata
        metadata = self._get_metadata(original_idx)
        metadata = {k: torch.tensor(v) for k, v in metadata.items()}

        # Data processing
        data_q_saxs = self.transformer_q_saxs.transform(data_q_saxs)
        data_y_saxs = self.transformer_y_saxs.transform(data_y_saxs)
        data_q_les = self.transformer_q_les.transform(data_q_les)
        data_y_les = self.transformer_y_les.transform(data_y_les)

        # Convert to tensors if needed
        data_q_saxs = torch.as_tensor(data_q_saxs, dtype=torch.float32)
        data_y_saxs = torch.as_tensor(data_y_saxs, dtype=torch.float32)
        data_q_les = torch.as_tensor(data_q_les, dtype=torch.float32)
        data_y_les = torch.as_tensor(data_y_les, dtype=torch.float32)

        # return data_q, data_y, metadata, self.csv_index[original_idx]
        return {"data_q_saxs": data_q_saxs.unsqueeze(0), "data_y_saxs": data_y_saxs.unsqueeze(0),
                "data_q_les": data_q_les.unsqueeze(0), "data_y_les": data_y_les.unsqueeze(0),
                "metadata": metadata, "csv_index": self.csv_index[original_idx]}

    def close(self):
        """Close the HDF5 file"""
        self.hdf.close()

    def transforms_to_dict(self):
        """Convert the transformations to a dictionary format"""
        return {
            "q_saxs": self.transformer_q_saxs.to_dict(),
            "y_saxs": self.transformer_y_saxs.to_dict(),
            "q_les": self.transformer_q_les.to_dict(),
            "y_les": self.transformer_y_les.to_dict()
        }