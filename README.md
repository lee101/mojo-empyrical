# mojo-empyrical

`mojo-empyrical` is a standalone Mojo port of the compute-heavy risk and
performance metrics in [empyrical](https://github.com/quantopian/empyrical).
It keeps the familiar Python function names and signatures while moving the
array scans and rolling calculations into a compiled shared library.

The package is imported as `mojoempyrical` (the alias `mojo_empyrical` is also
provided):

```python
import numpy as np
import mojoempyrical as ep

rng = np.random.default_rng(0)
returns = rng.normal(0.0004, 0.01, 5_000)
benchmark = rng.normal(0.0002, 0.008, 5_000)

print(ep.sharpe_ratio(returns))
print(ep.max_drawdown(returns))
print(ep.alpha_beta(returns, benchmark))
print(ep.roll_sharpe_ratio(returns, window=252)[-1])
```

## Coverage

The Mojo-backed API covers:

- return transforms: `simple_returns`, `cum_returns`, `cum_returns_final`;
- annualized metrics: `annual_return`/`cagr`, `annual_volatility`,
  `calmar_ratio`;
- risk and reward metrics: `max_drawdown`, `sharpe_ratio`, `sortino_ratio`,
  `downside_risk`, `omega_ratio`, `stability_of_timeseries`;
- factor metrics: `alpha`, `beta`, `alpha_beta`, their aligned forms,
  `excess_sharpe`, capture ratios, filtered alpha/beta, and beta fragility;
- rolling kernels: Sharpe, Sortino, annual volatility, max drawdown, alpha,
  beta, and alpha/beta;
- distribution metrics: `tail_ratio`, `value_at_risk`, and
  `conditional_value_at_risk`;
- pandas calendar aggregation through `aggregate_returns`.

The array APIs accept one-dimensional NumPy arrays and pandas Series. The
unary non-rolling metrics also accept two-dimensional arrays and DataFrames,
preserving upstream return conventions and labels. NaNs, `out=` buffers,
annualization periods, short inputs, and Series alignment are parity-tested
against empyrical 0.5.5.

Of the public callable API exported by empyrical 0.5.5, this port does not
implement `compute_exposures`, `perf_attrib`, `gpd_risk_estimates`, or
`gpd_risk_estimates_aligned`. Rolling alpha/beta currently accepts
one-dimensional inputs. Calendar
aggregation and percentile/partition operations stay in pandas/NumPy because
those libraries already execute their heavy work in compiled code.

## Install and run

The repository pins the tested Mojo nightly and installs the real upstream
empyrical package for parity testing:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` creates `dist/libmojo-empyrical.so`. The Python loader also
rebuilds a missing or stale library when Mojo is available. Set
`MOJOEMPYRICAL_LIB` to load a prebuilt library from another location.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14. Inputs are C-contiguous float64
arrays; timings are the best of five runs and include Python wrapper and
output allocation time.

| metric | mojo-empyrical | empyrical 0.5.5 | speedup |
| --- | ---: | ---: | ---: |
| simple_returns, 5M | 17.36 ms | 35.04 ms | 2.02x |
| cum_returns, 5M | 28.56 ms | 52.56 ms | 1.84x |
| max_drawdown, 5M | 12.33 ms | 447.20 ms | 36.27x |
| Sharpe ratio, 5M | 36.42 ms | 108.34 ms | 2.97x |
| Sortino ratio, 5M | 44.21 ms | 885.47 ms | 20.03x |
| alpha_beta, 5M | 36.87 ms | 2920.08 ms | 79.21x |
| rolling Sharpe, 100k x 252 | 0.98 ms | 5482.09 ms | 5616.97x |
| rolling alpha_beta, 100k x 252 | 112.46 ms | 24108.96 ms | 214.39x |

Simple returns uses native-width float64 SIMD and parallelizes independent
chunks for large inputs. Large one-dimensional cumulative returns uses a
four-chunk prefix scan followed by a SIMD-adjusted output pass; smaller inputs
remain serial to avoid thread-launch overhead. Drawdown and factor metrics benefit
from fused scans that avoid large intermediate NumPy arrays. Rolling Sharpe
maintains the entering and leaving sum and squared sum in constant space;
upstream evaluates a two-dimensional rolling view and creates large
temporaries.

No GPU path is provided. The targeted transform and prefix kernels perform far
less than two floating-point operations per byte moved, while the rolling
kernels already use constant-space updates and are strongly ahead on CPU. Their
arithmetic intensity does not justify host/device transfer and launch overhead.

## How it works

All numerical kernels live in one Mojo compilation unit. Python converts
inputs only when needed to C-contiguous float64 storage, then ctypes passes
buffer addresses as 64-bit integers through a C ABI. Mojo reconstructs
`UnsafePointer[Float64, AnyOrigin[mut=True]]` values from those addresses.
The caller owns every input and output allocation, so no allocator or object
ownership crosses the FFI boundary.

Matrices use row-major C order. Matrix transforms run all columns in one Mojo
call; scalar column reductions pass a base address and row stride without
copying each column. pandas index alignment, output labels, calendar grouping,
and API validation remain in the Python layer.

MIT licensed.
