from functools import partial
from jax import jit
import jax.numpy as jnp


@partial(jit, static_argnums=(0,))
def simpson_3_8(f, a, b):
    return (b-a)/8.0 * (f(a) + 3.0*f((2.0*a+b)/3.0) + 3.0*f((a+2.0*b)/3.0) + f(b))


@partial(jit, static_argnums=(0,3))
def simpson_n(f, a, b, n=1000):
    k = jnp.arange(start=1, stop=n)
    return (b-a)/n * (f(a)/2.0 + jnp.sum(f(a+k*(b-a)/n)) + f(b)/2.0)
