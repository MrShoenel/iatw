from abc import ABC
from enum import Enum, auto
from math import isnan
from typing import Callable, Tuple
from typing_extensions import Self
from strongtyping.strong_typing import match_typing
from iatw.Computable import Computable
from iatw.Parameter import Parameter



class IntervalType(Enum):
    CONSTANT = auto()
    MIN_MAX = auto()
    FLEXIBLE = auto()
    RELATIVE = auto()




class Interval(ABC):
    def __init__(self, interval_type: IntervalType, model: Computable, ref_length: float) -> None:
        from iatw.AlignmentModel import AlignmentModel
        super().__init__()
        if not type(model) is AlignmentModel:
            raise Exception('Model needs to be of type AlignmentModel')
        if isnan(ref_length) or ref_length <= 0.0:
            raise Exception('The reference length must be greater than 0')
        self.interval_type = interval_type
        self.model: AlignmentModel = model
        self._ref_length = ref_length
        self._params: list[Parameter] = []

    @property
    def params(self) -> list[Parameter]:
        return self._params.copy() # return a copy of this list
    
    @property
    def ref_length(self) -> float:
        return self._ref_length

    @ref_length.setter
    def ref_length(self, value: float) -> Self:
        if isnan(value) or value <= 0.0:
            raise Exception('The reference length must be greater than 0')
        self._ref_length = value
        return self


class IntervalWithLength(Interval):
    def __init__(self, interval_type: IntervalType, model: Computable, ref_length: float, length: float=None) -> None:
        super().__init__(interval_type=interval_type, ref_length=ref_length, model=model)
        self._params.append(Parameter(name='len', is_trainable=False))
        if not length is None and not isnan(length):
            self.length(length=length)
    
    @property
    def length(self):
        return self._params[0].value
    
    @length.setter
    def length(self, value: float) -> Self:
        if isnan(value) or value < 0.0:
            raise Exception('You must not pass NaN or a negative value for the length')
        p = self._params[0]
        p.value = value
        return self
    
    @property
    def offset(self):
        return self.model.interval_offset(interval=self)



class ConstantLengthInterval(IntervalWithLength):
    def __init__(self, model: Computable, ref_length: float, length: float=None) -> None:
        super().__init__(interval_type=IntervalType.CONSTANT, model=model, ref_length=ref_length, length=length)
        self._params[0].is_trainable = False
    
    def length(self, value: float) -> Self:
        super().length = value
        p = self._params[0]
        p.lower_bound = p.upper_bound = value
        return self

class MinMaxLengthInterval(IntervalWithLength):
    def __init__(self, model: Computable, ref_length: float, length: float = None, min_max_lengths: Tuple[float, float] = None) -> None:
        """
        Parameter min_max_lengths are the box bounds. However, the length should
        actually be controlled by some inequality constraints, which are handled
        by the optimizer, not the model!
        """
        super().__init__(interval_type=IntervalType.MIN_MAX, model=model, ref_length=ref_length, length=length)
        self._params[0].is_trainable = True
        if not min_max_lengths is None:
            self.min_max_lengths(min_max_lengths=min_max_lengths)
    
    @property
    def min_max_lengths(self) -> Tuple[float, float]:
        p = self._params[0]
        return (p.lower_bound, p.upper_bound)
    
    @min_max_lengths.setter
    @match_typing
    def min_max_lengths(self, min_max_lengths: Tuple[float, float]) -> Self:
        mm = min_max_lengths
        if isnan(mm[0]) or isnan(mm[1]):
            raise Exception('Neither length must be NaN')
        if mm[0] >= mm[1]:
            raise Exception('Minimum is larger than or equal to maximum')
        
        l = self.length()
        if l < mm[0] or l > mm[1]:
            raise Exception('Current length is outside bounds')
        
        p = self._params[0]
        p.lower_bound = mm[0]
        p.upper_bound = mm[1]
        return self


class RelativeLengthInterval(IntervalWithLength):
    def __init__(self, model: Computable, ref_length: float, rel_length: float = None) -> None:
        super().__init__(interval_type=IntervalType.RELATIVE, model=model, ref_length=ref_length, length=rel_length)
        self._params[0].name = 'rel_len'
        self._params[0].is_trainable = True
    
    @property
    def rel_length(self) -> float:
        return self.params[0].value
    
    def length(self):
        """
        A relative-length interval determines its length using a ratio
        of the available width.
        """
        m = self.model
        p = self._params[0]
        return (p.value + m.mu_R()) * m.rho / sum(m.vartheta_R)

class FlexibleInterval(IntervalWithLength):
    def __init__(self, model: Computable, ref_length: float) -> None:
        super().__init__(interval_type=IntervalType.FLEXIBLE, model=model, ref_length=ref_length)
        self._params.clear() # We don't have params.
    
    def length(self):
        return self.model.available_length # Takes up all of it.
    
    def length(self, value: float):
        raise Exception('A flexible interval cannot have an explicit length')
