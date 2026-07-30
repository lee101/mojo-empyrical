"""Benchmark Mojo kernels against empyrical 0.5.5 on identical arrays."""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

if not hasattr(np, "NINF"):
    np.NINF = -np.inf

import empyrical as reference  # noqa: E402
import mojoempyrical as mojo  # noqa: E402


def timeit(function, repeats=5):
    best = math.inf
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in stream
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(7)
    n = 5_000_000
    returns = np.ascontiguousarray(rng.normal(0.0, 0.005, n))
    factor = np.ascontiguousarray(rng.normal(0.0, 0.004, n))
    prices = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.001, n)))
    rolling_returns = returns[:100_000]
    rolling_factor = factor[:100_000]

    cases = [
        (
            "simple_returns, 5M",
            lambda: mojo.simple_returns(prices),
            lambda: reference.simple_returns(prices),
        ),
        (
            "cum_returns, 5M",
            lambda: mojo.cum_returns(returns),
            lambda: reference.cum_returns(returns),
        ),
        (
            "max_drawdown, 5M",
            lambda: mojo.max_drawdown(returns),
            lambda: reference.max_drawdown(returns),
        ),
        (
            "Sharpe ratio, 5M",
            lambda: mojo.sharpe_ratio(returns),
            lambda: reference.sharpe_ratio(returns),
        ),
        (
            "Sortino ratio, 5M",
            lambda: mojo.sortino_ratio(returns),
            lambda: reference.sortino_ratio(returns),
        ),
        (
            "alpha_beta, 5M",
            lambda: mojo.alpha_beta(returns, factor),
            lambda: reference.alpha_beta(returns, factor),
        ),
        (
            "rolling Sharpe, 100k x 252",
            lambda: mojo.roll_sharpe_ratio(rolling_returns, 252),
            lambda: reference.roll_sharpe_ratio(rolling_returns, 252),
        ),
        (
            "rolling alpha_beta, 100k x 252",
            lambda: mojo.roll_alpha_beta(rolling_returns, rolling_factor, 252),
            lambda: reference.roll_alpha_beta(rolling_returns, rolling_factor, 252),
        ),
    ]

    mojo.max_drawdown(returns[:10])
    print(f"Machine: {machine()}")
    print()
    print("| metric | mojo-empyrical | empyrical 0.5.5 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours_time = timeit(ours)
        theirs_time = timeit(theirs)
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{theirs_time * 1000:.2f} ms | {theirs_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
