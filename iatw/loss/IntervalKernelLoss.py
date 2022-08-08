from iatw.IntervalKernels import IntervalKernel
from iatw.Parameter import Parameter
from iatw.loss.Loss import Loss


class IntervalKernelLoss(Loss):
    def __init__(self, kernel: IntervalKernel, weight: Parameter = None) -> None:
        super().__init__(weight)
        self.kernel = kernel
    
    def evaluate(self) -> list[float]:
        return super().evaluate()