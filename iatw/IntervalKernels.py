from typing import Iterable
from iatw.Computable import Computable
from iatw.Intervals import IntervalWithLength
from iatw.Parameter import Parameter


class IntervalKernel(Computable):
    def __init__(self, interval: IntervalWithLength, ref_idx: int=None) -> None:
        super().__init__()
        self.interval = interval
        self.model = interval.model
        # Essentially allow the Kernel to refer to another reference interval.
        self._ref_idx = self.qry_idx if ref_idx is None else ref_idx
    
    @property
    def params(self) -> Iterable[Parameter]:
        yield from ()
    
    @property
    def ref_idx(self) -> int:
        return self._ref_idx

    @property
    def qry_idx(self) -> int:
        list(self.model.intervals()).index(self.interval)
    
    @property
    def source_begin(self) -> float:
        return self.model.beta_L + self.interval.offset
    
    # Alias:
    s_b = source_begin

    @property
    def ref_support(self) -> tuple[float, float]:
        idx = self.ref_idx
        tb = self.model.theta_b
        return (tb[idx], tb[idx + 1])
    
    @property
    def qry_support(self) -> tuple[float, float]:
        b = self.source_begin
        return (b, b + self.interval.length)

    @property
    def target_begin(self) -> float:
        return self.ref_support[0]
    
    # Alias:
    t_b = target_begin
    
    @property
    def target_end(self) -> float:
        return self.ref_support[1]
    
    # Alias:
    t_e = target_end

    @property
    def delta_t(self) -> float:
        s = self.ref_support
        return s[1] - s[0]


class LinearWarpingKernel(IntervalKernel):
    def __init__(self, interval: IntervalWithLength, ref_idx: int=None) -> None:
        super().__init__(interval=interval, ref_idx=ref_idx)
    
    @property
    def qry_support(self) -> tuple[float, float]:
        # For a linear warping kernel, both supports are the same!
        return super().ref_support

    def evaluate(self, x) -> list[float]:
        return super().evaluate()
    
    def __call__(self, x) -> float:
        s = self.ref_support
        if x < s[0] or x > s[1]:
            raise Exception('x is outside bounds for this interval')
        return self.model.f_qry(self.s_b + (self.interval.length * (x - self.t_b) / self.delta_t))