import pickle

import numpy as np


class StrictlyPositiveTransformer:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, data):
        # No fitting necessary; included for consistency with the API
        return self

    def transform(self, data):
        data = np.asarray(data)
        transformed = np.where(data <= 0, self.epsilon, data)
        return transformed

    def batch_transform(self, data):
        data = np.asarray(data)
        transformed = np.where(data <= 0, self.epsilon, data)
        return transformed


class LogTransformer:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, data):
        # No fitting necessary; included for consistency with the API
        return self

    def transform(self, data):
        data = np.asarray(data)
        transformed = np.log(data + self.epsilon)
        return transformed

    def batch_transform(self, data):
        data = np.asarray(data)
        transformed = np.log(data + self.epsilon)
        return transformed


class LogPlusTransformer:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, data):
        # No fitting necessary; included for consistency with the API
        return self

    def transform(self, data):
        data = np.asarray(data)
        transformed = np.log(data + 1 + self.epsilon)
        return transformed

    def batch_transform(self, data):
        data = np.asarray(data)
        transformed = np.log(data + 1 + self.epsilon)
        return transformed


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        self.mean_ = np.mean(data)
        self.std_ = np.std(data)
        return self

    def transform(self, data):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transformation.")
        if self.std_ == 0:
            return np.zeros_like(data)
        return (data - self.mean_) / self.std_

    def batch_transform(self, batch_data):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transformation.")
        if self.std_ == 0:
            return np.zeros_like(batch_data)
        return (batch_data - self.mean_) / self.std_


class MinMaxNormalizer:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        self.min_ = np.min(data)
        self.max_ = np.max(data)
        return self

    def transform(self, data):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Le normaliseur doit être ajusté (fit) avant la transformation.")
        if self.max_ == self.min_:
            return np.zeros_like(data)
        return (data - self.min_) / (self.max_ - self.min_)

    def batch_transform(self, batch_data):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Le normaliseur doit être ajusté (fit) avant la transformation.")
        if self.max_ == self.min_:
            return np.zeros_like(batch_data)
        return (batch_data - self.min_) / (self.max_ - self.min_)


class PaddingTransformer:
    def __init__(self, pad_size, value=0):
        self.pad_size = pad_size
        self.value = value

    def fit(self, data):
        return self

    def transform(self, data):
        if len(data) >= self.pad_size:
            return data[:self.pad_size]
        return np.pad(data, (0, self.pad_size - len(data)), constant_values=self.value)

    def batch_transform(self, batch_data):
        transformed_batch = []
        for data in batch_data:
            if len(data) >= self.pad_size:
                transformed = data[:self.pad_size]
            else:
                transformed = np.pad(data, (0, self.pad_size - len(data)), constant_values=self.value)
            transformed_batch.append(transformed)
        return np.array(transformed_batch)


class PreprocessingSAXS:
    def __init__(self, pad_size=54, value=0):
        self.pipeline = SequentialTransformer(config={
            "PaddingTransformer": {"pad_size": pad_size, "value": value},
            "StrictlyPositiveTransformer": {},
            "LogTransformer": {},
            "MinMaxNormalizer": {}
        })

    def fit(self, data):
        self.pipeline.fit(data)
        return self

    def transform(self, data):
        return self.pipeline.transform(data)

    def batch_transform(self, batch_data):
        transformed_batch = []
        for data in batch_data:
            transformed_batch.append(self.transform(data))
        return np.array(transformed_batch)

    def save(self, filepath):
        self.pipeline.save(filepath)

    @staticmethod
    def load(filepath):
        preprocessing = PreprocessingSAXS()
        preprocessing.pipeline = SequentialTransformer.load(filepath)
        return preprocessing


class PreprocessingLES:
    def __init__(self, pad_size, value=0):
        self.pipeline = SequentialTransformer(config={
            "PaddingTransformer": {"pad_size": pad_size, "value": value},
            "MinMaxNormalizer": {}
        })

    def fit(self, data):
        self.pipeline.fit(data)
        return self

    def transform(self, data):
        return self.pipeline.transform(data)

    def batch_transform(self, batch_data):
        transformed = []
        for d in batch_data:
            transformed.append(self.transform(d))
        return np.array(transformed)

    def save(self, filepath):
        self.pipeline.save(filepath)

    @staticmethod
    def load(filepath):
        instance = PreprocessingLES(pad_size=0)
        instance.pipeline = SequentialTransformer.load(filepath)
        return instance


class PreprocessingQ:
    def __init__(self, pad_size, value=0):
        self.pipeline = SequentialTransformer(config={
            "PaddingTransformer": {"pad_size": pad_size, "value": value},
        })

    def fit(self, data):
        self.pipeline.fit(data)
        return self

    def transform(self, data):
        return self.pipeline.transform(data)

    def batch_transform(self, batch_data):
        transformed = []
        for d in batch_data:
            transformed.append(self.transform(d))
        return np.array(transformed)

    def save(self, filepath):
        self.pipeline.save(filepath)

    @staticmethod
    def load(filepath):
        instance = PreprocessingLES(pad_size=0)
        instance.pipeline = SequentialTransformer.load(filepath)
        return instance


#################################################################################

class SequentialTransformer:
    def __init__(self, config=None):
        if config is not None:
            self.transformers = self.parse_config(config)
        else:
            self.transformers = []

    def parse_config(self, config):
        transformer_map = {
            "StandardScaler": StandardScaler,
            "LogTransformer": LogTransformer,
            "LogPlusTransformer": LogPlusTransformer,
            "StrictlyPositiveTransformer": StrictlyPositiveTransformer,
            "MinMaxNormalizer": MinMaxNormalizer,
            "PaddingTransformer": PaddingTransformer,
            "PreprocessingSAXS": PreprocessingSAXS,
            "PreprocessingLES": PreprocessingLES,
            "PreprocessingQ": PreprocessingQ,
        }
        transformers = []
        for transform_name, params in config.items():
            if transform_name in transformer_map:
                transformers.append(transformer_map[transform_name](**params))
            else:
                raise ValueError(f"Transformateur inconnu : {transform_name}")
        return transformers

    def fit(self, all_data):
        fitted_transformers = []
        for transformer in self.transformers:
            fitted_transformer = transformer.fit(all_data)
            fitted_transformers.append(fitted_transformer)
            all_data = fitted_transformer.batch_transform(all_data)
        self.transformers = fitted_transformers

    def transform(self, data):
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
