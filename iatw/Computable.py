from abc import abstractmethod
from typing import Any, Union
from nptyping import Shape, Float, NDArray
from jax._src.device_array import DeviceArray
from iatw.Parameterized import Parameterized

FloatLike = Union[float, Float, NDArray[Shape["*"], Float], DeviceArray]



class Computable(Parameterized):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, *args: FloatLike) -> FloatLike:
        """
        Compute this computable, that is, perform all necessary calculations
        and return the numeric outcome(s).
        """
        pass

    def __call__(self) -> FloatLike:
        """
        Wrapper for evaluate().
        """
        return self.evaluate()
