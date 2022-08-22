
from typing import Iterable
from iatw.Computable import Computable
from iatw.Parameter import Parameter


class Loss(Computable):
    
    def __init__(self, weight: Parameter=None) -> None:
        super().__init__()
        self.weight = weight
    
    @property
    def params(self) -> Iterable[Parameter]:
        yield self.weight
