from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    name = None

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def batch_transform(self, batch_data):
        pass

    def to_dict(self):
        config = {
            k: getattr(self, k)
            for k in vars(self)
            if not k.startswith('_')
        }
        return {self.name or self.__class__.__name__: config}

class MinMaxNormalizer(BaseTransformer):
    name = "MinMaxNormalizer"

    def __init__(self, min_val=None, max_val=None):
        self.no_fit = True
        if min_val is not None and max_val is not None:
            if min_val >= max_val:
                raise ValueError("min_val must be less than max_val.")
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data):
        if not self.no_fit:
            self.min_val = np.min(data)
            self.max_val = np.max(data)
        return self

    def transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Le normaliseur doit être ajusté (fit) avant la transformation.")
        if self.max_val == self.min_val:
            return np.zeros_like(data)
        return (data - self.min_val) / (self.max_val - self.min_val)

    def batch_transform(self, batch_data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Le normaliseur doit être ajusté (fit) avant la transformation.")
        if self.max_val == self.min_val:
            return np.zeros_like(batch_data)
        return (batch_data - self.min_val) / (self.max_val - self.min_val)

    def invert_transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Le normaliseur doit être ajusté (fit) avant l'inversion.")
        return data * (self.max_val - self.min_val) + self.min_val

class PaddingTransformer(BaseTransformer):
    name = "PaddingTransformer"

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

class StrictlyPositiveTransformer(BaseTransformer):
    name = "StrictlyPositiveTransformer"

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, data):
        return self

    def transform(self, data):
        data = np.asarray(data)
        return np.where(data <= 0, self.epsilon, data)

    def batch_transform(self, data):
        data = np.asarray(data)
        return np.where(data <= 0, self.epsilon, data)

class LogTransformer(BaseTransformer):
    name = "LogTransformer"

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, data):
        return self

    def transform(self, data):
        data = np.asarray(data)
        return np.log(data + self.epsilon)

    def batch_transform(self, data):
        data = np.asarray(data)
        return np.log(data + self.epsilon)

    def invert_transform(self, data):
        data = np.asarray(data)
        return np.exp(data) - self.epsilon

#################################################################################
import numpy as np

class BasePreprocessing(BaseTransformer):
    def __init__(self, config):
        self.pipeline = SequentialTransformer(config=config)

    def fit(self, data):
        self.pipeline.fit(data)
        return self

    def transform(self, data):
        return self.pipeline.transform(data)

    def batch_transform(self, batch_data):
        return np.array([self.transform(d) for d in batch_data])


class PreprocessingSAXS(BasePreprocessing):
    def __init__(self, pad_size=54, value=0):
        config = {
            "PaddingTransformer": {"pad_size": pad_size, "value": value},
            "StrictlyPositiveTransformer": {},
            "LogTransformer": {},
            "MinMaxNormalizer": {}
        }
        super().__init__(config=config)


class PreprocessingLES(BasePreprocessing):
    def __init__(self, pad_size, value=0):
        config = {
            "PaddingTransformer": {"pad_size": pad_size, "value": value},
            "MinMaxNormalizer": {}
        }
        super().__init__(config=config)


class PreprocessingQ(BasePreprocessing):
    def __init__(self, pad_size, value=0):
        config = {
            "PaddingTransformer": {"pad_size": pad_size, "value": value}
        }
        super().__init__(config=config)

#################################################################################

class SequentialTransformer:
    def __init__(self, config=None):
        if config is not None:
            self.transformers = self.parse_config(config)
        else:
            self.transformers = []

    def parse_config(self, config):
        transformer_map = {
            "LogTransformer": LogTransformer,
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

    def invert_transform(self, data):
        for transformer in reversed(self.transformers):
            data = transformer.invert_transform(data)
        return data

