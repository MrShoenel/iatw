


from collections import deque
from typing import Iterable
from typing_extensions import Self
from warnings import warn
from iatw.model.AlignmentModel import AlignmentModel
from iatw.Computable import Computable
from iatw.kernel.IntervalKernels import IntervalKernel
from iatw.loss.Loss import Loss


class AlignmentProblem(Computable):
    
    def __init__(self, model: AlignmentModel) -> None:
        super().__init__()
        self.model = model

        self._kernels: list[IntervalKernel] = []

        self._losses: deque[Loss] = deque()
        self._regularizers: deque[Loss] = deque()
    
    def add_interval_pair_kernel(self, kernel: IntervalKernel) -> Self:
        """
        In this AlignmentProblem, we do not allow to have more than one Kernel operating on a single
        query interval. It is allowed, however, to use composite Kernels and MultiReference-Kernels.
        """
        if any(filter(lambda k: k.interval == kernel.interval, self._kernels)):
            raise Exception('Another Kernel for the enclosed interval was added previously.')

        self._kernels.append(kernel)
        return self

    def add_interval_pair_loss(self, ref_idx: int, qry_idx: int, loss: Loss) -> Self:
        pass


    
    @property
    def losses(self) -> Iterable[Loss]:
        return self._losses.copy()
    
    @property
    def regularizers(self) -> Iterable[Loss]:
        return self._regularizers.copy()
    
    def add_loss(self, loss: Loss) -> Self:
        self._losses.append(loss)
        return self
    
    def remove_loss(self, loss: Loss) -> Self:
        self._losses.remove(loss)
        return self
    
    def add_regularizer(self, reg: Loss) -> Self:
        self._regularizers.append(reg)
        return self
    
    def remove_regularizer(self, reg: Loss) -> Self:
        self._regularizers.remove(reg)
        return self