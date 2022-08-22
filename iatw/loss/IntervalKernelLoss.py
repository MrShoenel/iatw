from iatw.kernel.IntervalKernels import IntervalKernel
from iatw.Parameter import Parameter
from iatw.loss.Loss import Loss


class IntervalKernelLoss(Loss):
    def __init__(self, kernel: IntervalKernel, weight: Parameter = None) -> None:
        super().__init__(weight)
        self.kernel = kernel
