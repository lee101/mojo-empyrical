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
| simple_returns, 5M | 25.11 ms | 62.70 ms | 2.50x |
| cum_returns, 5M | 39.32 ms | 65.21 ms | 1.66x |
| max_drawdown, 5M | 11.59 ms | 198.43 ms | 17.12x |
| Sharpe ratio, 5M | 38.79 ms | 140.95 ms | 3.63x |
| Sortino ratio, 5M | 47.72 ms | 174.63 ms | 3.66x |
| alpha_beta, 5M | 48.01 ms | 474.38 ms | 9.88x |
| rolling Sharpe, 100k x 252 | 1.48 ms | 764.43 ms | 515.80x |
| rolling alpha_beta, 100k x 252 | 135.80 ms | 2459.76 ms | 18.11x |

Simple returns uses native-width float64 SIMD and parallelizes independent
chunks for large inputs. Drawdown and factor metrics benefit
from fused scans that avoid large intermediate NumPy arrays. Rolling Sharpe
maintains the entering and leaving sum and squared sum in constant space;
upstream evaluates a two-dimensional rolling view and creates large
temporaries.

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
