from tests.ExtendedTestCase import ExtendedTestCase
from os import environ, cpu_count




class JaxTestCase(ExtendedTestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        environ['JAX_ENABLE_X64'] = 'True'
        environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count()}'
        environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
