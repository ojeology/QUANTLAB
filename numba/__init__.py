"""
Minimal numba stub for Python 3.13 compatibility.
shap imports njit from numba for JIT-compiled helpers; on Python 3.13
real numba cannot be installed.  This stub makes njit/jit a no-op so
shap's TreeExplainer loads and runs correctly (it does not actually call
the numba-decorated clustering helpers at inference time).
"""


def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    def decorator(func):
        return func
    return decorator


def jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    def decorator(func):
        return func
    return decorator


# shap/links.py uses:  numba.extending.overload, numba.types, numba.float64 etc.
# Provide enough stubs to survive import without error.
float64 = float
float32 = float
int32   = int
int64   = int
boolean = bool
void    = type(None)

prange = range


class _StubModule:
    """Catches any attribute access and returns a no-op."""
    def __getattr__(self, name):
        return _StubModule()
    def __call__(self, *a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        def deco(fn):
            return fn
        return deco


extending = _StubModule()
types      = _StubModule()
core       = _StubModule()
