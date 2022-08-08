from math import isnan
from typing_extensions import Self
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
    @match_typing
    def bounds(self, value: tuple[float, float]) -> Self:
        self.lower_bound = value[0]
        self.upper_bound = value[1]
        return self

    @property
    def is_within_bounds(self) -> bool:
        lb, ub = self.bounds
        v = self.value
        if lb is None or ub is None or v is None:
            raise Exception('Cannot check if within bounds if they are not set or value is not present')
        
        if any([isnan(x) for x in [lb, ub, v]]):
            raise Exception('Cannot perform check if any value is NaN')
        return v >= lb and v <= ub
