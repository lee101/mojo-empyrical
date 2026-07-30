from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOEMPYRICAL_LIB") or os.path.join(
    ROOT, "dist", "libmojo-empyrical.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mep_simple_returns": ([I, I, I, I], None),
    "mep_cum_returns": ([I, I, I, I, F], None),
    "mep_cum_final": ([I, I, I, F], F),
    "mep_annual_return": ([I, I, I, F], F),
    "mep_annual_volatility": ([I, I, I, F, F], F),
    "mep_max_drawdown": ([I, I, I], F),
    "mep_sharpe": ([I, I, I, F, F], F),
    "mep_downside_risk": ([I, I, I, F, F], F),
    "mep_sortino": ([I, I, I, F, F], F),
    "mep_omega": ([I, I, I, F, F], F),
    "mep_stability": ([I, I, I], F),
    "mep_beta": ([I, I, I, I, I], F),
    "mep_alpha": ([I, I, I, I, I, F, F, F], F),
    "mep_excess_sharpe": ([I, I, I, I, I], F),
    "mep_roll_sharpe": ([I, I, I, I, F, F], None),
    "mep_roll_volatility": ([I, I, I, I, F, F], None),
    "mep_roll_sortino": ([I, I, I, I, F, F], None),
    "mep_roll_max_drawdown": ([I, I, I, I], None),
    "mep_roll_alpha_beta": ([I, I, I, I, I, F, F], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOEMPYRICAL_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
            return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build`")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        details = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
        raise BuildError(details[:4000] or "Mojo build failed without diagnostic output")
    return LIB


_LIBRARY = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def f64(values) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind in "cO":
        raise TypeError("complex and object arrays cannot be converted safely to float64")
    if source.dtype.kind == "f" and source.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("floating-point inputs wider than float64 would lose precision")
    if source.dtype.kind in "iu" and source.size:
        exact_limit = 1 << 53
        if source.dtype.kind == "u":
            unsafe = np.max(source) > exact_limit
        else:
            unsafe = np.min(source) < -exact_limit or np.max(source) > exact_limit
        if unsafe:
            raise TypeError("integer input contains values that float64 cannot represent exactly")
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(values: np.ndarray) -> int:
    if values.size == 0 or values.ctypes.data == 0:
        raise ValueError("cannot pass an empty or null NumPy buffer across the FFI boundary")
    return int(values.ctypes.data)
