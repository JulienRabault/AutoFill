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

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, pad_size=None, normalize=True, metadata_filters=None,
                 conversion_dict_path=None, frac=1, enable_timing=False,
                 requested_metadata=[]):
        """
        Optimized PyTorch Dataset for HDF5 files with selective metadata preprocessing
        """
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r', swmr=True)
        self.pad_size = pad_size
        self.normalize = normalize
        self.enable_timing = enable_timing

        # Initialize timing statistics
        self.timing_stats = defaultdict(list)
        self.samples_processed = 0
        self.data_q = self.hdf['data_q']
        self.data_y = self.hdf['data_y']
        self.csv_index = self.hdf['csv_index']

        # Identify available metadata columns
        all_metadata_cols = [col for col in self.hdf.keys() if col not in
                             ['data_q', 'data_y', 'len', 'csv_index']]
        self.metadata_datasets = {col: self.hdf[col] for col in all_metadata_cols}

        # Handle requested metadata
        self.requested_metadata = self._validate_requested_metadata(requested_metadata, all_metadata_cols)

        # Load conversion dictionary
        self.conversion_dict = self._load_conversion_dict(conversion_dict_path)

        # Apply metadata filters using vectorized operations
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
        # print(f"│ Filter dict metadata: {self.metadata_filters:<15} │")
        print(f"│ Samples filtered: {self.len_after_filter:<23} │")
        print(f"│ Requested fraction: {self.frac:<22} │")
        print(f"│ Fractioned samples: {len(self.filtered_indices):<22} │")
        print(
            f"│ Requested metadata: {', '.join(self.requested_metadata) if self.requested_metadata else 'None':<15} │")
        print(f"│ Pad size: {str(self.pad_size) if self.pad_size else 'None':<30} │")
        print(f"│ Normalization: {str(self.normalize):<28} │")
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

        # print("Start of sorted_indices")
        # sorted_indices = sorted(self.filtered_indices)
        # print("End of sorted_indices")
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
        if self.normalize:
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

        return data_q, data_y, metadata, self.csv_index[original_idx]

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
        if not isinstance(data_q, torch.Tensor):
            data_q = torch.tensor(data_q, dtype=torch.float32)
        if not isinstance(data_y, torch.Tensor):
            data_y = torch.tensor(data_y, dtype=torch.float32)

        # Normalisation de data_q
        min_q = torch.min(data_q)
        max_q = torch.max(data_q)
        if max_q > min_q:
            data_q = (data_q - min_q) / (max_q - min_q)
        else:
            data_q = torch.zeros_like(data_q)

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



def load_data_from_file(file_path):
    data_q, data_y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', 'Q', 'q', 'Q;', 'q;')):
                continue
            tokens = line.replace(';', ' ').replace(',', ' ').split()
            try:
                values = [float(token) if token.lower() != 'nan' else float('nan') for token in tokens]
            except ValueError:
                continue
            if len(values) == 2:
                values = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in values]
                data_q.append(values[0])
                data_y.append(values[1])
    return np.array(data_q), np.array(data_y)

if __name__ == "__main__":
    hdf5_file = 'data.h5'
    csv_file = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'
    data_dir = '../AUTOFILL_data/datav2/Base_de_donnee'
    pad_size = 2000
    conversion_dict_path = 'conversion_dict.json'
    metadata_filters = {"technique": ["les"]}

    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    if not os.path.exists(conversion_dict_path):
        raise FileNotFoundError(f"Conversion dictionary file not found: {conversion_dict_path}")

    df = pd.read_csv(csv_file)
    dataset = HDF5Dataset(hdf5_file=hdf5_file, pad_size=pad_size, metadata_filters=metadata_filters, conversion_dict_path=conversion_dict_path)
    indices_to_test = [0, 1, 2, 3, 4]

    for idx in indices_to_test:
        data_q_h5, data_y_h5, metadata, csv_file_index = dataset[idx]
        file_path = os.path.join(data_dir, df.iloc[int(csv_file_index.item())]['path'])
        normalized_path = os.path.normpath(file_path.replace('\\', os.sep).replace('/', os.sep))

        if not os.path.exists(normalized_path):
            print(f"Fichier introuvable : {normalized_path}")
            continue

        data_q_txt, data_y_txt = load_data_from_file(normalized_path)

        match_q = np.allclose(data_q_h5.numpy()[:len(data_q_txt)], data_q_txt, atol=1e-6)
        match_y = np.allclose(data_y_h5.numpy()[:len(data_y_txt)], data_y_txt, atol=1e-6)

        print(f"Index {idx}:")
        print(f"Index csv: {int(csv_file_index.item())}")
        print(f"Path: {normalized_path}")
        print(f"data_q_h5: {data_q_h5}")
        print(f"data_q_txt: {data_q_txt}")
        print(f"data_y_h5: {data_y_h5}")
        print(f"data_y_txt: {data_y_txt}")
        print(f"Match Q: {match_q}")
        print(f"Match Y: {match_y}")
        print("=" * 40)

    dataset.close()
