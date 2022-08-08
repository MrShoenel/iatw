from abc import abstractmethod
from iatw.Parameterized import Parameterized


class Computable(Parameterized):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self) -> list[float]:#NDArray[Shape["Dim", npt_float]]:
        """
        Compute this computable, that is, perform all necessary calculations
        and return the numeric outcome(s).
        """
        pass

    def __call__(self) -> list[float]:#NDArray[Shape["Dim", npt_float]]:
        """
        Wrapper for evaluate().
        """
        return self.evaluate()

    # @abstractmethod
    # def gradient(self) -> NDArray[Shape['Dim'], npt_float]:
    #     """
    #     Compute all 1st-order partial derivatives of this model
    #     """
    #     pass

    # @abstractmethod
    # def hessian(self) -> NDArray[Shape['Dim, Dim'], npt_float]:
    #     pass
