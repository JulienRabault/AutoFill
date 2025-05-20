from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

# region Base Transformer
class Transformer(ABC):
    name: str

    @abstractmethod
    def fit(self, data: np.ndarray) -> "Transformer":
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def invert(self, data: np.ndarray) -> np.ndarray:
        pass

    def get_config(self) -> Dict[str, Any]:
        init_params = self.__init__.__code__.co_varnames[1:self.__init__.__code__.co_argcount]
        return {self.name: {k: getattr(self, k) for k in init_params if hasattr(self, k)}}
# endregion

# region Basic Transformers
class MinMaxScaler(Transformer):
    name = "MinMaxScaler"

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self._fitted = False
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        if self.min_val is None or self.max_val is None:
            self.min_val = float(np.min(data))
            self.max_val = float(np.max(data))
        if self.min_val >= self.max_val:
            raise ValueError("min_val must be less than max_val.")
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform.")
        if self.max_val == self.min_val:
            return np.zeros_like(data)
        return (data - self.min_val) / (self.max_val - self.min_val)

    def invert(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Scaler must be fitted before invert.")
        return data * (self.max_val - self.min_val) + self.min_val

class Padding(Transformer):
    name = "PaddingTransformer"

    def __init__(self, pad_size: int, value: float = 0):
        self.pad_size = pad_size
        self.value = value

    def fit(self, data: np.ndarray) -> "Padding":
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        length = data.shape[-1]
        if length >= self.pad_size:
            return data[..., : self.pad_size]
        pad_width = [(0, 0)] * data.ndim
        pad_width[-1] = (0, self.pad_size - length)
        return np.pad(data, pad_width, constant_values=self.value)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return data

class EnsurePositive(Transformer):
    name = "StrictlyPositiveTransformer"

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon

    def fit(self, data: np.ndarray) -> "EnsurePositive":
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.where(data <= 0, self.epsilon, data)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return data

class Log(Transformer):
    name = "LogTransformer"

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon

    def fit(self, data: np.ndarray) -> "Log":
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.log(data + self.epsilon)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return np.exp(data) - self.epsilon
# endregion

# region Preprocessing Base
class PreprocessingBase(Transformer):
    def __init__(self, pad_size: int, value: float = 0):
        self.pad_size = pad_size
        self.value = value

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.pipeline.transform(data)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return self.pipeline.invert(data)

    def fit(self, data: np.ndarray):
        self.pipeline.fit(data)
        return self

# endregion

# region Preprocessing Pipelines
class PreprocessingLES(PreprocessingBase):
    name = "PreprocessingLES"

    def __init__(self, pad_size: int, value: float = 0):
        super().__init__(pad_size, value)
        self.pipeline = Pipeline([
            Padding(pad_size, value),
            MinMaxScaler(),
        ])

class PreprocessingSAXS(PreprocessingBase):
    name = "PreprocessingSAXS"

    def __init__(self, pad_size: int = 54, value: float = 0):
        super().__init__(pad_size, value)
        self.pipeline = Pipeline([
            Padding(pad_size, value),
            EnsurePositive(),
            Log(),
            MinMaxScaler(),
        ])

class PreprocessingQ(PreprocessingBase):
    name = "PreprocessingQ"

    def __init__(self, pad_size: int, value: float = 0):
        super().__init__(pad_size, value)
        self.pipeline = Pipeline([
            Padding(pad_size, value),
        ])

# endregion

# region Pipeline
class Pipeline:
    transformer_map = {
        MinMaxScaler.name: MinMaxScaler,
        Padding.name: Padding,
        EnsurePositive.name: EnsurePositive,
        Log.name: Log,
        PreprocessingLES.name: PreprocessingLES,
        PreprocessingSAXS.name: PreprocessingSAXS,
        PreprocessingQ.name: PreprocessingQ,
    }

    def __init__(self, config_or_steps: Union[Sequence[Transformer], Dict[str, Dict[str, Any]]]= {}):
        if isinstance(config_or_steps, dict):
            steps: List[Transformer] = []
            for name, params in config_or_steps.items():
                if name not in self.transformer_map:
                    raise ValueError(f"Unknown transformer: {name}")
                steps.append(self.transformer_map[name](**params))
            self.steps = steps
        else:
            self.steps = list(config_or_steps)

    def fit(self, data: Union[np.ndarray, Sequence[np.ndarray]]) -> "Pipeline":
        array = np.array(data)
        for step in self.steps:
            step.fit(array)
            array = step.transform(array)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        array = data
        for step in self.steps:
            array = step.transform(array)
        return array

    def batch_transform(self, batch: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([self.transform(item) for item in batch])

    def invert(self, data: np.ndarray) -> np.ndarray:
        array = data
        for step in reversed(self.steps):
            array = step.invert(array)
        return array

    def to_dict(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for step in self.steps:
            merged.update(step.get_config())
        return merged
# endregion
