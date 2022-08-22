from typing import Any, Callable, Type
from unittest import TestCase


class ExtendedTestCase(TestCase):

    def assertDoesNotRaise(self, fn: Callable[[], Any], exType: Type=Exception):
        try:
            fn()
        except Exception as e:
            if isinstance(e, exType):
                raise AssertionError(f'The function raises: {e}')
