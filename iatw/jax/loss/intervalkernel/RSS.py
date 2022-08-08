


import numpy as np
from functools import partial
from iatw.model.AlignmentModel import AlignmentModel
from iatw.kernel.IntervalKernels import IntervalKernel
from iatw.Parameter import Parameter
from iatw.loss.IntervalKernelLoss import IntervalKernelLoss
from jax import jit, grad, numpy as jnp


class RSS(IntervalKernelLoss):
    def __init__(self, kernel: IntervalKernel, weight: Parameter = None) -> None:
        super().__init__(kernel=kernel, weight=weight)

        # We need to jit ref and query.
        m = kernel.interval.model
        ref_supp = m.ref_supp
        x_ref = jnp.linspace(start=ref_supp[0], stop=ref_supp[1], num=10_000)
        y_ref = np.vectorize(m.f_ref)(x_ref) # list([m.f_ref(x) for x in x_ref])
        self.f_ref = jit(lambda x: jnp.interp(x, x_ref, y_ref))
        
        qry_supp = m.qry_supp
        x_qry = jnp.linspace(start=qry_supp[0], stop=qry_supp[1], num=10_000)
        y_qry = np.vectorize(m.f_qry)(x_qry) # list([m.f_qry(x) for x in x_qry])
        self.f_qry = jit(lambda x: jnp.interp(x, x_qry, y_qry))
    

    @partial(jit, static_argnums=(0,))
    def evaluate(self) -> list[float]:
        k = self.kernel
        
        ref_supp = k.ref_support
        x_ref = jnp.linspace(start=ref_supp[0], stop=ref_supp[1], num=1000)
        y_ref = self.f_ref(x_ref)
        qry_supp = k.qry_support
        x_qry = jnp.linspace(start=qry_supp[0], stop=qry_supp[1], num=1000)
        y_qry = self.f_qry(x_qry)

        return [jnp.sum(jnp.square(y_ref - y_qry))]
