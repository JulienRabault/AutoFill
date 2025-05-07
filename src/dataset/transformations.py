import numpy as np


class LogTransformer:
    def __init__(self):
        pass

    def transform(self, data):
        if np.any(data <= 0):
            raise ValueError("Tous les éléments doivent être strictement positifs pour appliquer le log.")
        return np.log(data)

    def fit_transform(self, data):
        return self.transform(data)


class LogTransformer2:
    def __init__(self):
        pass

    def transform(self, data):
        data = data - np.min(data) + 1e-9
        if np.any(data <= 0):
            raise ValueError("Tous les éléments doivent être strictement positifs pour appliquer le log.")
        return np.log(data)

    def fit_transform(self, data):
        return self.transform(data)


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

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class PaddingTransformer:
    def __init__(self, pad_size, value=0):
        self.pad_size = pad_size
        self.value = value

    def transform(self, data):
        if len(data) >= self.pad_size:
            return data[:self.pad_size]
        return np.pad(data, (0, self.pad_size - len(data)), constant_values=self.value)

    def fit_transform(self, data):
        return self.transform(data)


class SequentialTransformer:
    def __init__(self, config=None):
        if config is not None:
            self.transformers = self.parse_config(config)
        else:
            self.transformers = []

    def parse_config(self, config):
        transformer_map = {
            "LogTransformer": LogTransformer,
            "MinMaxNormalizer": MinMaxNormalizer,
            "PaddingTransformer": PaddingTransformer
        }
        transformers = []
        for transform_name, params in config.items():
            if transform_name in transformer_map:
                transformers.append(transformer_map[transform_name](**params))
            else:
                raise ValueError(f"Transformateur inconnu : {transform_name}")
        return transformers

    def fit_transform(self, data):
        for transformer in self.transformers:
            data = transformer.fit_transform(data)
        return data
