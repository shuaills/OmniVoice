"""Minimal pytest shim for gate runs in the pytest-less donor venv.
Only what our test files use: mark.parametrize, raises, skip.
Never shadow a real pytest: only reachable via tools/pytest_shim on PYTHONPATH."""


class _Mark:
    def parametrize(self, argnames, argvalues):
        def deco(fn):
            fn._param = (argnames, list(argvalues))
            return fn
        return deco

    def __getattr__(self, _name):
        def deco(fn):
            return fn
        return deco


mark = _Mark()


class raises:
    def __init__(self, exc, match=None):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), \
            f"expected {self.exc}, got {et}"
        return True


def skip(reason=""):
    raise AssertionError(f"pytest.skip called in shim: {reason}")
