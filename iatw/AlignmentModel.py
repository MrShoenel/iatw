from enum import Enum, auto
from math import isnan
#from nptyping import NDArray, Shape, Int as npt_int, Float as npt_float
from typing import Callable, Iterable, Optional
from typing_extensions import Self
from sys import maxsize

from nptyping import NDArray, Shape, Float as npt_float
from iatw.Intervals import Interval, IntervalType, IntervalWithLength, RelativeLengthInterval
from iatw.loss.Loss import Loss
from iatw.Parameterized import Parameterized
from strongtyping.strong_typing import match_typing

from iatw.Parameter import Parameter
from collections import deque


class SignalType(Enum):
    REFERENCE = auto()
    QUERY = auto()



class AlignmentModel(Parameterized):

    @match_typing
    def __init__(self,
        f_ref: Callable[[float], float],
        f_qry: Callable[[float], float],
        ref_begin: Parameter,
        qry_begin: Parameter,
        qry_end: Parameter,
        gamma_d: float=None
    ) -> None:
        super().__init__()
        self.f_ref = f_ref
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
    def qry_supp(self) -> tuple[float, float]:
        return (self.qry_begin.lower_bound, self.qry_end.upper_bound)


    @property
    def ref_supp(self) -> tuple[float, float]:
        return self._ref_begin.bounds
    
    @ref_supp.setter
    @match_typing
    def ref_supp(self, value: tuple[float, float]) -> Self:
        if isnan(value[0]) or isnan(value[1]) or value[0] >= value[1]:
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
        ref_length = self.ref_supp[1] - self.ref_supp[0]
        used_ref_length = sum([i.ref_length for i in self.intervals()])
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

    def interval_offset(self, interval: Interval) -> float:
        offset = self.beta_L
        intervals = self.intervals()
        for i in intervals:
            if i == interval:
                break
            offset += i.length
        
        return offset
    
    # Alias for paper
    phi = interval_offset

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
        return self.beta_U() - self.getBetaL() -\
            sum([i.length for i in self.intervals(type=IntervalType.CONSTANT)]) -\
            sum([i.length for i in self.intervals(type=IntervalType.MIN_MAX)])

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
    def params(self) -> Iterable[Parameter]:
        yield self.qry_begin
        yield self.qry_end
        yield from [[p for p in l.params] for l in self.losses]
        yield from [[p for p in r.params] for r in self.regularizers]
        yield from [[p for p in i.params] for i in self.intervals()]

    @property
    def trainable_params_as_dict(self) -> dict[str, float]:
        return { f'{p.name}_{idx + 1}':p.value for idx, p in enumerate(self.trainable_params) }
    
    @property
    def lower_bounds_as_dict(self) -> dict[str, float]:
        return { f'{p.name}_{idx + 1}':p.lower_bound for idx, p in enumerate(self.trainable_params) }

    @property
    def upper_bounds_as_dict(self) -> dict[str, float]:
        return { f'{p.name}_{idx + 1}':p.upper_bound for idx, p in enumerate(self.trainable_params) }
    
    # def unit_function(self, interval_index: int, signalType: SignalType) -> Callable[[float], float]:
    #     if (interval_index + 1) > len(self.reference_boundaries):
    #         raise Exception(f'Interval index of {interval_index} is out of bounds')
        
    #     offset = 0
    #     length = 0
    #     func: Callable[[float], float]
    #     if signalType == SignalType.REFERENCE:
    #         offset = self.reference_boundaries[interval_index]
    #         length = self.reference_boundaries[interval_index + 1] - self.reference_boundaries[interval_index]
    #         func = self.f_ref
    #     else:
    #         interval = self._intervals[interval_index]
    #         offset = interval.offset
    #         length = interval.length
    #         func = self.f_qry
        
    #     return lambda x: func(x * length + offset)

    
    # def __call__(self, x: float) -> float:
    #     """
    #     \hat{f}^{(qry)}
    #     """
    #     return super().__call__(*args, **kwds)

