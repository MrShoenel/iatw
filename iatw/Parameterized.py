from abc import ABC, abstractmethod
from typing import Iterable
from iatw.Parameter import Parameter


class Parameterized(ABC):

    @property
    @abstractmethod
    def params(self) -> Iterable[Parameter]:
        pass

    @property
    def trainable_params(self) -> Iterable[Parameter]:
        for param in self.params:
            if param.is_trainable:
                yield param
