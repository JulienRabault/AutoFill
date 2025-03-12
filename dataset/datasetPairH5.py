import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

class PairHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, pad_size=None, metadata_filters=None,
                 conversion_dict_path=None, frac=1, enable_timing=False,
                 requested_metadata=[], to_normalize=[]):
        """
        Optimized PyTorch Dataset for HDF5 files with selective metadata preprocessing
        """
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r', swmr=True)
        self.pad_size = pad_size
        self.enable_timing = enable_timing
        self.to_normalize = to_normalize

        # Initialize timing statistics
        self.timing_stats = defaultdict(list)
        self.samples_processed = 0
        self.data_q = self.hdf['data_q']
        self.data_y = self.hdf['data_y']
        self.csv_index = self.hdf['csv_index']

        all_metadata_cols = [col for col in self.hdf.keys() if col not in
                             ['data_q', 'data_y', 'len', 'csv_index']]
        self.metadata_datasets = {col: self.hdf[col] for col in all_metadata_cols}

        self.requested_metadata = self._validate_requested_metadata(requested_metadata, all_metadata_cols)

        self.conversion_dict = self._load_conversion_dict(conversion_dict_path)

        self.metadata_filters = metadata_filters or {}
        self.filtered_indices = self._apply_metadata_filters()
        self.len_after_filter = len(self.filtered_indices)

        # Validate and apply data fraction
        self._validate_frac(frac)
        self.frac = frac
        if 0 < frac < 1:
            self._apply_data_fraction(frac)

        # Preprocess only requested metadata
        self.metadata_tensors = self._preprocess_metadata()

        # Print initialization info
        self._print_init_info()

    def _print_init_info(self):
        """Print dataset initialization information"""
        print("\n╒══════════════════════════════════════════════╕")
        print("│ Dataset Initialization Info                 │")
        print("╞══════════════════════════════════════════════╡")
        print(f"│ File: {self.hdf5_file:<35} │")
        print(f"│ Total samples: {len(self.data_q):<26} │")
        print(f"│ Samples filtered: {self.len_after_filter:<23} │")
        print(f"│ Requested fraction: {self.frac:<22} │")
        print(f"│ Fractioned samples: {len(self.filtered_indices):<22} │")
        print(f"│ Pad size: {str(self.pad_size) if self.pad_size else 'None':<30} │")
        print(f"│ To normalization: {str(self.to_normalize):<28} │")

        # Calcul des normes de data_y
        data_y_tensor = torch.tensor(self.data_y[:])  # Conversion en tensor PyTorch
        norm_l1 = torch.norm(data_y_tensor, p=1).item()
        norm_l2 = torch.norm(data_y_tensor, p=2).item()
        norm_max = torch.norm(data_y_tensor, p=float('inf')).item()

        print(f"│ Norme L1 de data_y: {norm_l1:<24.4f} │")
        print(f"│ Norme L2 de data_y: {norm_l2:<24.4f} │")
        print(f"│ Norme max de data_y: {norm_max:<22.4f} │")

        print("╘══════════════════════════════════════════════╛\n")

    def _validate_requested_metadata(self, requested, available):
        """Validate and filter requested metadata columns"""
        if requested is None:
            return available

        valid = [col for col in requested if col in available]
        missing = set(requested) - set(valid)

        if missing:
            warnings.warn(f"Missing requested metadata columns: {missing}")

        return valid

    def _validate_frac(self, frac):
        """Validate data fraction parameter"""
        if not (0 < frac <= 1):
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
            return list(range(len(self.data_q)))

        mask = np.ones(len(self.data_q), dtype=bool)

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

    def _apply_data_fraction(self, frac):
        """Select a fraction of the filtered indices in sorted order"""
        num_samples = int(len(self.filtered_indices) * frac)
        self.filtered_indices = self.filtered_indices[:num_samples]

    def _preprocess_metadata(self):
        """Preprocess requested metadata to tensors during initialization"""
        if not self.requested_metadata:
            return torch.empty((len(self.filtered_indices), 0))

        metadata_columns = []
        for col in tqdm(self.requested_metadata, desc="Preprocessing metadata"):
            data = self.metadata_datasets[col][self.filtered_indices]

            if col in self.conversion_dict:
                conv_dict = self.conversion_dict[col]
                vectorized_map = np.vectorize(lambda x: conv_dict.get(str(x), -1))
                converted = vectorized_map(data)
                metadata_columns.append(converted.astype(float))
            else:
                metadata_columns.append(data.astype(float))

        return torch.tensor(np.column_stack(metadata_columns), dtype=torch.float32)

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        """Optimized data loading with timing and progress tracking"""
        timers = {}
        if self.enable_timing:
            timers['start'] = time.perf_counter()

        # Get original dataset index
        original_idx = self.filtered_indices[idx]
        if self.enable_timing:
            timers['indexing'] = time.perf_counter()

        # Load main data
        data_q = self.data_q[original_idx]
        data_y = self.data_y[original_idx]
        if self.enable_timing:
            timers['data_load'] = time.perf_counter()

        # Get preprocessed metadata
        metadata = self.metadata_tensors[idx]
        if self.enable_timing:
            timers['metadata'] = time.perf_counter()

        # Data processing
        if self.to_normalize:
            data_q, data_y = self._normalize_data(data_q, data_y)
        if self.enable_timing:
            timers['processing'] = time.perf_counter()
        if self.pad_size:
            data_q, data_y = self._pad_data(data_q, data_y)
        if self.enable_timing:
            timers['end'] = time.perf_counter()

        # Convert to tensors if needed
        data_q = torch.as_tensor(data_q, dtype=torch.float32)
        data_y = torch.as_tensor(data_y, dtype=torch.float32)

        if self.enable_timing:
            self._update_timing_stats(timers, idx)

        # return data_q, data_y, metadata, self.csv_index[original_idx]
        return {"data_q": data_q, "data_y": data_y, "metadata": metadata, "csv_index": self.csv_index[original_idx]}

    def _update_timing_stats(self, timers, idx):
        """Update timing statistics and display averages when complete"""
        t = timers

        # Calculate timings
        stages = [
            ('Indexing', t.get('indexing', t['start']) - t['start']),
            ('Data Load', t['data_load'] - t.get('indexing', t['start'])),
            ('Metadata', t['metadata'] - t['data_load']),
            ('Processing', time.perf_counter() - t['metadata'])
        ]

        # Update statistics
        for name, duration in stages:
            self.timing_stats[name].append(duration)

        self.samples_processed += 1

        if self.samples_processed >= (len( self.filtered_indices) // os.cpu_count()) * 0.85:
            self._print_timing_averages()
            self.samples_processed=0

    def _print_timing_averages(self):
        """Print average timing statistics"""
        print("\n╒══════════════════════════════════════════════╕")
        print("│ Average Timings (per sample)                │")
        print("╞══════════════════════════════════════════════╡")

        total = 0.0
        for name, times in self.timing_stats.items():
            avg = np.mean(times)
            total += avg
            print(f"│ {name.ljust(20)} {avg:.6f} s {' ' * 12} │")

        print(f"│ {'TOTAL'.ljust(20)} {total:.6f} s {' ' * 12} │")
        print("╘══════════════════════════════════════════════╛\n")

    def _normalize_data(self, data_q, data_y):
        """
        Normalise data_q et data_y avec la formule (data - min) / (max - min)
        si spécifié dans self.to_normalize.
        """

        def minmax_norm(x):
            min_x, max_x = torch.min(x), torch.max(x)
            return (x - min_x) / (max_x - min_x) if max_x > min_x else torch.zeros_like(x)

        if not isinstance(data_q, torch.Tensor):
            data_q = torch.tensor(data_q, dtype=torch.float32)
        if not isinstance(data_y, torch.Tensor):
            data_y = torch.tensor(data_y, dtype=torch.float32)

        if "data_q" in self.to_normalize:
            data_q = minmax_norm(data_q)

        if "data_y" in self.to_normalize:
            data_y = minmax_norm(data_y)

        return data_q, data_y

    def _pad_data(self, data_q, data_y):
        def pad(data, pad_size):
            # Si les données ne sont pas déjà des tensors, les convertir en tensors
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)

            num_rows = data.size(0)
            if num_rows < pad_size:
                padding_rows = pad_size - num_rows
                pad_tensor = torch.zeros(padding_rows, dtype=torch.float32)
                return torch.cat([data, pad_tensor], dim=0)
            return data[:pad_size]

        # Appliquer le padding aux deux données
        padded_q = pad(data_q, self.pad_size)
        padded_y = pad(data_y, self.pad_size)

        return padded_q, padded_y

    def close(self):
        """Close the HDF5 file"""
        self.hdf.close()
