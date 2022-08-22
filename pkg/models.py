from abc import ABC, abstractmethod
from numpy import NaN, isnan
from nptyping import NDArray, Shape
from typing import Callable, Any, Iterable, Optional, Tuple
from typing_extensions import Self
from sys import maxsize
from strongtyping.strong_typing import match_typing


from dataclasses import dataclass

@dataclass
class Parameter:
    name: str
    value: float = None
    lower_bound: float = None
    upper_bound: float = None
    is_trainable: bool = True

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.lower_bound, self.upper_bound)
    
    @bounds.setter
    def bounds(self, value: tuple[float, float]) -> Self:
        self.lower_bound = value[0]
        self.upper_bound = value[1]
        return self

    @property
    def is_within_bounds(self) -> bool:
        l = [self.value, self.lower_bound, self.upper_bound]
        if any(isnan(l)):
            raise Exception('Cannot perform check if any value is NaN')
        return l[0] >= l[1] and l[0] <= l[2]
    




class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self) -> float:
        """
        Compute this model, that is, calculate the objective, given all
        defined losses and current parameter settings.
        """
        pass

    @abstractmethod
    def gradient(self) -> NDArray[Shape['Dim'], float]:
        """
        Compute all 1st-order partial derivatives of this model
        """
        pass

    def hessian(self) -> NDArray[Shape['Dim, Dim'], float]:
        pass

    @abstractmethod
    @property
    def params(self) -> list[Parameter]:
        pass

    @abstractmethod
    @property
    def params_as_dict(self) -> dict[str, float]:
        pass



class Problem(ABC):
    def __init__(self) -> None:
        super().__init__()









from enum import Enum, auto

class IntervalType(Enum):
    CONSTANT = auto()
    MIN_MAX = auto()
    FLEXIBLE = auto()
    RELATIVE = auto()


class Interval(ABC):
    def __init__(self, interval_type: IntervalType, model: Model, ref_length: float) -> None:
        super().__init__()
        if not type(model) is AlignmentModel:
            raise Exception('Model needs to be of type AlignmentModel')
        if isnan(ref_length) or ref_length <= 0.0:
            raise Exception('The reference length must be greater than 0')
        self.interval_type = interval_type
        self.model = model
        self._ref_length = ref_length
        self._params: list[Parameter] = []

    @property
    def params(self) -> list[Parameter]:
        return self._params[:] # return a copy of this list
    
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
    def __init__(self, interval_type: IntervalType, model: Model, ref_length: float, length: float=None) -> None:
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


class ConstantLengthInterval(IntervalWithLength):
    def __init__(self, model: Model, ref_length: float, length: float=None) -> None:
        super().__init__(interval_type=IntervalType.CONSTANT, model=model, ref_length=ref_length, length=length)
        self._params[0].is_trainable = False
    
    def length(self, value: float) -> Self:
        super().length = value
        p = self._params[0]
        p.lower_bound = p.upper_bound = value
        return self

class MinMaxLengthInterval(IntervalWithLength):
    def __init__(self, model: Model, ref_length: float, length: float = None, min_max_lengths: Tuple[float, float] = None) -> None:
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
    
    @match_typing
    @min_max_lengths.setter
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


class RelativeLengthInterval(IntervalWithLength):
    def __init__(self, model: Model, ref_length: float, rel_length: float = None) -> None:
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
        m: AlignmentModel = self.model
        p = self._params[0]
        return (p.value + m.mu_R()) * m.rho / sum(m.vartheta_R)

class FlexibleInterval(IntervalWithLength):
    def __init__(self, model: Model, ref_length: float) -> None:
        super().__init__(interval_type=IntervalType.FLEXIBLE, model=model, ref_length=ref_length)
        self._params.clear() # We don't have params.
    
    def length(self):
        m: AlignmentModel = self.model
        return m.available_length() # Takes up all of it.
    
    def length(self, value: float):
        raise Exception('A flexible interval cannot have an explicit length')




class IntervalKernel(ABC):
    def __init__(self, interval: IntervalWithLength) -> None:
        super().__init__()
        self.interval = interval
        self.model = interval.model

    @property
    def interval_idx(self) -> int:
        list(self.model.intervals()).index(self.interval)

    @property
    def support(self) -> tuple[float, float]:
        idx = self.interval_idx
        tb = self.model.theta_b
        return (tb[idx], tb[idx + 1])

    @property
    def target_begin(self) -> float:
        return self.support[0]
    
    # Alias:
    t_b = target_begin
    
    @property
    def target_end(self) -> float:
        return self.support[1]
    
    # Alias:
    t_e = target_end

    @property
    def delta_t(self) -> float:
        s = self.support
        return s[1] - s[0]


class LinearWarpingKernel(IntervalKernel):
    def __init__(self, interval: IntervalWithLength) -> None:
        super().__init__(interval)
    
    @property
    def phi(self) -> float:
        idx = self.interval_idx
        if idx == 0:
            return 0.0
        return sum(map(lambda i: self.model._intervals[i].length, range(idx)))
    
    @property
    def source_begin(self) -> float:
        self.model.beta_L + self.phi
    
    # Alias:
    s_b = source_begin
    
    def __call__(self, x) -> float:
        s = self.support
        if x < s[0] or x > s[1]:
            raise Exception('x is outside bounds for this interval')
        return self.model.f_qry(self.s_b + (self.interval.length * (x - self.t_b) / self.delta_t))







from collections import deque

class AlignmentModel(Model):

    @match_typing
    def __init__(self,
        f_qry: Callable[[float], float],
        ref_begin: Parameter,
        qry_begin: Parameter,
        qry_end: Parameter,
        gamma_d: float=None
    ) -> None:
        super().__init__()
        self.f_qry = f_qry

        self._ref_begin: Parameter = None
        self.ref_begin = ref_begin
        
        self._intervals: deque[IntervalWithLength] = deque()

        if qry_begin.lower_bound >= qry_end.upper_bound:
            # begin-upper and end-lower are essentially ignored, as
            # gamma_b is begin-lower and gamme_e is end-upper
            raise Exception('begin/end bounds misconfigured')
        self.qry_begin = qry_begin
        self.qry_begin.name = 'begin'
        self.qry_end = qry_end
        self.qry_end.name = 'end'
        self.gamma_d = gamma_d



    @property
    def ref_supp(self) -> tuple[float, float]:
        return self._ref_begin.bounds
    
    @ref_supp.setter
    @match_typing
    def ref_supp(self, value: tuple[float, float]) -> Self:
        if any(isnan(value)) or value[0] >= value[1]:
            raise Exception('The reference support is ill-defined')
        self._ref_begin.bounds = value
        return self
    
    @property
    def ref_begin(self) -> Parameter:
        return self._ref_begin

    @ref_begin.setter
    @match_typing
    def ref_begin(self, value: Parameter) -> Self:
        if value.value is None or isnan(value.value):
            raise Exception('A concrete begin is required')
        if not value.is_within_bounds:
            raise Exception('The begin is not within bounds')
        self._ref_begin = value
        self.ref_supp = value.bounds
        return self
    
    def intervals(self, type: Optional[IntervalType]=None) -> Iterable[IntervalWithLength]:
        if type is None:
            return list(self._intervals)

        def f(i: Interval):
            return i.interval_type == type
        return filter(f, self._intervals)
    

    @property
    def has_flexible_interval(self) -> bool:
        return len(list(self.intervals(type=IntervalType.FLEXIBLE))) == 1
    

    @match_typing
    def add_interval(self, interval: IntervalWithLength, idx: int=maxsize) -> Self:
        if self.has_flexible_interval and interval.interval_type == IntervalType.FLEXIBLE:
            raise Exception('A model can only have zero to one flexible intervals')

        # Check reference support
        ref_length = self._ref_supp[1] - self._ref_supp[0]
        def f(i: IntervalWithLength):
            return i.ref_length
        used_ref_length = sum(map(f, self._intervals))
        if used_ref_length + interval.ref_length > ref_length:
            raise Exception(f'Cannot add interval as reference-length would exceed the available length of {ref_length} by {used_ref_length + interval.ref_length - ref_length}')
        
        # Check query support
        # TODO: More checking here before we actually add the intervals
        
        self._intervals.insert(idx, interval)
        return self
    
    @property
    def is_open_begin(self) -> bool:
        return self.qry_begin.is_trainable
    
    @property
    def is_open_end(self) -> bool:
        return self.qry_end.is_trainable
    
    @property
    def P(self) -> int:
        """
        Returns the total number of intervals.
        """
        return len(self._intervals)
    
    @property
    def reference_boundaries(self) -> list[float]:
        t = [self.ref_begin.value]
        for idx, interval in enumerate(self.intervals()):
            t.append(t[idx] + interval.ref_length)
        return t
    
    # Alias for paper
    theta_b = reference_boundaries
    
    def P_for_x(self, x: float) -> int:
        t = self.reference_boundaries
        if x < t[0] or x > t[-1]:
            raise Exception(f'x={x} is out of range, needs to be {t[0]} <= x <= {t[-1]}.')
        
        for idx in range(len(self._intervals) - 1):
            if x >= t[idx] and x < t[idx + 1]:
                return idx
        
        return len(self._intervals) - 1
    
    # Alias for paper
    kappa_of_x = P_for_x

    @property
    def beta_L(self) -> float:
        b = self.qry_begin
        e = self.qry_end
        if self.gamma_d is None:
            # Handled by optimizer + external constraints
            if b.value > e.value:
                raise Exception('Illegal model state, end comes before begin; check inequality constraints')
            return b.value
        
        # Otherwise, handle this within the model:
        return min(e.upper_bound - self.gamma_d, max(b.lower_bound, min(b.value, e.value)))
        

    @property
    def beta_U(self) -> float:
        b = self.qry_begin
        e = self.qry_end
        if self.gamma_d is None:
            if b.value > e.value:
                raise Exception('Illegal model state, end comes before begin; check inequality constraints')
            return e.value
        
        return max(b.lower_bound + self.gamma_d, min(e.upper_bound, max(b.value, e.value)))

    @property
    def available_length(self) -> float:
        def l(i: Interval):
            i.length()
        
        return self.beta_U() - self.getBetaL() -\
            sum([l(i) for i in self.intervals(type=IntervalType.CONSTANT)]) -\
            sum([l(i) for i in self.intervals(type=IntervalType.MIN_MAX)])

    rho = available_length

    @property
    def relative_lengths(self) -> Iterable[float]:
        intervals: Iterable[RelativeLengthInterval] = self.intervals(type=IntervalType.RELATIVE)
        return [i.rel_length for i in intervals]
    
    # Alias
    r = relative_lengths

    @property
    def mu_R(self) -> float:
        if sum(self.relative_lengths) == 0.0:
            return 1.0 / float(len(self._intervals_r))
        return float(0.0)
    
    @property
    def vartheta_R(self) -> Iterable[float]:
        return map(lambda elem: elem + self.mu_R, self.relative_lengths)
    

    @property
    def trainable_params(self) -> Iterable[Parameter]:
        if self.qry_begin.is_trainable:
            yield self.qry_begin
        if self.qry_end.is_trainable:
            yield self.qry_end
        for interval in self.intervals():
            for param in filter(lambda p: p.is_trainable, interval.params):
                yield param

    @property
    def params_as_dict(self) -> dict[str, float]:
        return { f'{p.name}_{idx + 1}':p.value for idx, p in enumerate(self.trainable_params) }
    
    @property
    def lower_bounds_as_dict(self) -> dict[str, float]:
        return { f'{p.name}_{idx + 1}':p.lower_bound for idx, p in enumerate(self.trainable_params) }

    @property
    def upper_bounds_as_dict(self) -> dict[str, float]:
        return { f'{p.name}_{idx + 1}':p.upper_bound for idx, p in enumerate(self.trainable_params) }

    
    def __call__(self, x: float) -> float:
        """
        \hat{f}^{(qry)}
        """
        return super().__call__(*args, **kwds)

