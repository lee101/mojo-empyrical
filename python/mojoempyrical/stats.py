from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._lib import addr, f64, lib

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
QUARTERLY = "quarterly"
YEARLY = "yearly"
ANNUALIZATION_FACTORS = {
    DAILY: 252,
    WEEKLY: 52,
    MONTHLY: 12,
    QUARTERLY: 4,
    YEARLY: 1,
}

__all__ = [
    "DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY",
    "aggregate_returns", "alpha", "alpha_aligned", "alpha_beta",
    "alpha_beta_aligned", "annual_return", "annual_volatility", "beta",
    "beta_aligned", "beta_fragility_heuristic",
    "beta_fragility_heuristic_aligned", "cagr", "calmar_ratio", "capture",
    "conditional_value_at_risk", "cum_returns", "cum_returns_final",
    "down_alpha_beta", "down_capture", "downside_risk", "excess_sharpe",
    "max_drawdown", "omega_ratio", "roll_alpha", "roll_alpha_aligned",
    "roll_alpha_beta", "roll_alpha_beta_aligned", "roll_annual_volatility",
    "roll_beta", "roll_beta_aligned", "roll_down_capture",
    "roll_max_drawdown", "roll_sharpe_ratio", "roll_sortino_ratio",
    "roll_up_capture", "roll_up_down_capture", "sharpe_ratio",
    "simple_returns", "sortino_ratio", "stability_of_timeseries",
    "tail_ratio", "up_alpha_beta", "up_capture", "up_down_capture",
    "value_at_risk",
]


def annualization_factor(period, annualization):
    if annualization is not None:
        return annualization
    try:
        return ANNUALIZATION_FACTORS[period]
    except KeyError:
        choices = "', '".join(ANNUALIZATION_FACTORS)
        raise ValueError(f"Period cannot be '{period}'. Can be '{choices}'.") from None


def _values(data) -> np.ndarray:
    values = f64(data)
    if values.ndim not in (1, 2):
        raise ValueError("returns must be one- or two-dimensional")
    return values


def _scalar_columns(data, function, *args) -> float | np.ndarray:
    values = _values(data)
    if values.shape[0] == 0:
        if values.ndim == 1:
            return float("nan")
        return np.full(values.shape[1], np.nan)
    if values.ndim == 1:
        return float(function(addr(values), len(values), 1, *args))
    rows, columns = values.shape
    result = np.empty(columns)
    for column in range(columns):
        result[column] = function(
            addr(values) + column * values.itemsize, rows, columns, *args
        )
    return result


def _return_out(result, out):
    if out is None:
        return result
    out[()] = result
    if np.ndim(result) == 0:
        return out.item()
    return out


def _frame_series(result, original, columns=False, default_index=False):
    if isinstance(original, pd.DataFrame):
        index = original.columns if columns else None
        return pd.Series(result, index=index if not default_index else None)
    return result


def _aligned(returns, factor_returns):
    if (
        isinstance(returns, np.ndarray)
        and isinstance(factor_returns, np.ndarray)
        and len(returns) == len(factor_returns)
    ):
        return returns, factor_returns
    left = returns if isinstance(returns, (pd.Series, pd.DataFrame)) else pd.Series(returns)
    right = (
        factor_returns
        if isinstance(factor_returns, (pd.Series, pd.DataFrame))
        else pd.Series(factor_returns)
    )
    joined = pd.concat([left, right], axis=1)
    return joined.iloc[:, 0], joined.iloc[:, 1]


def simple_returns(prices):
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        return prices.pct_change().iloc[1:]
    values = _values(prices)
    rows = values.shape[0]
    columns = 1 if values.ndim == 1 else values.shape[1]
    matrix = values.reshape(rows, columns)
    result = np.empty((max(rows - 1, 0), columns))
    if rows > 1:
        lib().mep_simple_returns(addr(matrix), addr(result), rows, columns)
    return result[:, 0] if values.ndim == 1 else result


def cum_returns(returns, starting_value=0, out=None):
    if len(returns) < 1:
        return returns.copy()
    values = _values(returns)
    rows = values.shape[0]
    columns = 1 if values.ndim == 1 else values.shape[1]
    matrix = values.reshape(rows, columns)
    allocated = out is None
    result = np.empty_like(values) if allocated else np.asarray(out)
    if (
        result.dtype != np.float64
        or not result.flags.c_contiguous
        or not result.flags.writeable
    ):
        raise TypeError("out must be a writable C-contiguous float64 array")
    if result.shape != values.shape:
        raise ValueError("out must have the same shape as returns")
    lib().mep_cum_returns(
        addr(matrix), addr(result), rows, columns, float(starting_value)
    )
    if allocated and isinstance(returns, pd.Series):
        return pd.Series(result, index=returns.index)
    if allocated and isinstance(returns, pd.DataFrame):
        return pd.DataFrame(result, index=returns.index, columns=returns.columns)
    return result


def cum_returns_final(returns, starting_value=0):
    if len(returns) == 0:
        return np.nan
    values = _values(returns)
    result = _scalar_columns(
        values, lib().mep_cum_final, float(starting_value)
    )
    if isinstance(returns, pd.DataFrame):
        return pd.Series(result, index=returns.columns)
    return result


def aggregate_returns(returns, convert_to):
    def cumulative(group):
        return cum_returns(group).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month / 3.0))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError("convert_to must be weekly, monthly or yearly")
    return returns.groupby(grouping).apply(cumulative)


def annual_return(returns, period=DAILY, annualization=None):
    factor = annualization_factor(period, annualization)
    result = _scalar_columns(returns, lib().mep_annual_return, float(factor))
    return _frame_series(result, returns, columns=True)


def cagr(returns, period=DAILY, annualization=None):
    return annual_return(returns, period, annualization)


def annual_volatility(
    returns, period=DAILY, alpha=2.0, annualization=None, out=None
):
    factor = annualization_factor(period, annualization)
    result = _scalar_columns(
        returns, lib().mep_annual_volatility, float(factor), float(alpha)
    )
    return _return_out(result, out)


def max_drawdown(returns, out=None):
    result = _scalar_columns(returns, lib().mep_max_drawdown)
    result = _frame_series(result, returns, default_index=True)
    return _return_out(result, out)


def calmar_ratio(returns, period=DAILY, annualization=None):
    drawdown = max_drawdown(returns)
    if np.ndim(drawdown):
        result = np.where(
            np.asarray(drawdown) < 0,
            np.asarray(annual_return(returns, period, annualization))
            / np.abs(np.asarray(drawdown)),
            np.nan,
        )
        result[np.isinf(result)] = np.nan
        return result
    if drawdown >= 0:
        return np.nan
    result = annual_return(returns, period, annualization) / abs(drawdown)
    return np.nan if np.isinf(result) else result


def sharpe_ratio(
    returns, risk_free=0, period=DAILY, annualization=None, out=None
):
    factor = annualization_factor(period, annualization)
    result = _scalar_columns(
        returns, lib().mep_sharpe, float(risk_free), float(factor)
    )
    return _return_out(result, out)


def downside_risk(
    returns, required_return=0, period=DAILY, annualization=None, out=None
):
    factor = annualization_factor(period, annualization)
    result = _scalar_columns(
        returns,
        lib().mep_downside_risk,
        float(required_return),
        float(factor),
    )
    result = _frame_series(result, returns, columns=True)
    return _return_out(result, out)


def sortino_ratio(
    returns,
    required_return=0,
    period=DAILY,
    annualization=None,
    out=None,
    _downside_risk=None,
):
    factor = annualization_factor(period, annualization)
    if _downside_risk is None:
        result = _scalar_columns(
            returns,
            lib().mep_sortino,
            float(required_return),
            float(factor),
        )
    else:
        adjusted = _values(returns) - required_return
        result = np.nanmean(adjusted, axis=0) * factor / _downside_risk
    result = _frame_series(result, returns, default_index=True)
    return _return_out(result, out)


def omega_ratio(
    returns, risk_free=0.0, required_return=0.0, annualization=252
):
    if annualization == 1:
        threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        threshold = (1 + required_return) ** (1.0 / annualization) - 1
    return _scalar_columns(
        returns, lib().mep_omega, float(risk_free), float(threshold)
    )


def stability_of_timeseries(returns):
    return _scalar_columns(returns, lib().mep_stability)


def tail_ratio(returns):
    values = np.asanyarray(returns)
    if len(values) < 1:
        return np.nan
    values = values[~np.isnan(values)]
    if len(values) < 1:
        return np.nan
    return abs(np.percentile(values, 95)) / abs(np.percentile(values, 5))


def value_at_risk(returns, cutoff=0.05):
    return np.percentile(returns, 100 * cutoff)


def conditional_value_at_risk(returns, cutoff=0.05):
    values = np.asanyarray(returns)
    cutoff_index = int((len(values) - 1) * cutoff)
    return np.mean(np.partition(values, cutoff_index)[: cutoff_index + 1])


def _binary_columns(returns, factor_returns, name, *args):
    left, right = _aligned(returns, factor_returns)
    a, b = _values(left), _values(right)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if len(a) != len(b):
        raise ValueError("returns and factor_returns must have the same length")
    columns = max(a.shape[1], b.shape[1])
    if a.shape[1] not in (1, columns) or b.shape[1] not in (1, columns):
        raise ValueError("incompatible column counts")
    if len(a) == 0:
        result = np.full(columns, np.nan)
        return result.item() if columns == 1 else result
    function = getattr(lib(), name)
    result = np.empty(columns)
    for column in range(columns):
        ai = min(column, a.shape[1] - 1)
        bi = min(column, b.shape[1] - 1)
        call_args = [
            float(np.asarray(value).reshape(-1)[column])
            if np.ndim(value)
            else value
            for value in args
        ]
        result[column] = function(
            addr(a) + ai * 8,
            addr(b) + bi * 8,
            len(a),
            a.shape[1],
            b.shape[1],
            *call_args,
        )
    return result.item() if columns == 1 else result


def beta(returns, factor_returns, risk_free=0.0, out=None):
    result = _binary_columns(returns, factor_returns, "mep_beta")
    return _return_out(result, out)


def beta_aligned(returns, factor_returns, risk_free=0.0, out=None):
    result = _binary_columns(returns, factor_returns, "mep_beta")
    return _return_out(result, out)


def alpha(
    returns,
    factor_returns,
    risk_free=0.0,
    period=DAILY,
    annualization=None,
    out=None,
    _beta=None,
):
    left, right = _aligned(returns, factor_returns)
    return alpha_aligned(
        left, right, risk_free, period, annualization, out, _beta
    )


def alpha_aligned(
    returns,
    factor_returns,
    risk_free=0.0,
    period=DAILY,
    annualization=None,
    out=None,
    _beta=None,
):
    factor = annualization_factor(period, annualization)
    beta_value = beta_aligned(returns, factor_returns, risk_free) if _beta is None else _beta
    result = _binary_columns(
        returns,
        factor_returns,
        "mep_alpha",
        float(risk_free),
        float(factor),
        beta_value,
    )
    return _return_out(result, out)


def alpha_beta(
    returns,
    factor_returns,
    risk_free=0.0,
    period=DAILY,
    annualization=None,
    out=None,
):
    left, right = _aligned(returns, factor_returns)
    return alpha_beta_aligned(
        left, right, risk_free, period, annualization, out
    )


def alpha_beta_aligned(
    returns,
    factor_returns,
    risk_free=0.0,
    period=DAILY,
    annualization=None,
    out=None,
):
    b = beta_aligned(returns, factor_returns, risk_free)
    a = alpha_aligned(
        returns, factor_returns, risk_free, period, annualization, _beta=b
    )
    result = np.stack([a, b], axis=-1)
    return _return_out(result, out)


def excess_sharpe(returns, factor_returns, out=None):
    result = _binary_columns(
        returns, factor_returns, "mep_excess_sharpe"
    )
    return _return_out(result, out)


def capture(returns, factor_returns, period=DAILY):
    return annual_return(returns, period) / annual_return(factor_returns, period)


def up_capture(returns, factor_returns, **kwargs):
    left, right = _aligned(returns, factor_returns)
    mask = np.asarray(right) > 0
    return capture(np.asarray(left)[mask], np.asarray(right)[mask], **kwargs)


def down_capture(returns, factor_returns, **kwargs):
    left, right = _aligned(returns, factor_returns)
    mask = np.asarray(right) < 0
    return capture(np.asarray(left)[mask], np.asarray(right)[mask], **kwargs)


def up_down_capture(returns, factor_returns, **kwargs):
    return up_capture(returns, factor_returns, **kwargs) / down_capture(
        returns, factor_returns, **kwargs
    )


def up_alpha_beta(returns, factor_returns, **kwargs):
    left, right = _aligned(returns, factor_returns)
    mask = np.asarray(right) > 0
    return alpha_beta_aligned(np.asarray(left)[mask], np.asarray(right)[mask], **kwargs)


def down_alpha_beta(returns, factor_returns, **kwargs):
    left, right = _aligned(returns, factor_returns)
    mask = np.asarray(right) < 0
    return alpha_beta_aligned(np.asarray(left)[mask], np.asarray(right)[mask], **kwargs)


def beta_fragility_heuristic(returns, factor_returns):
    left, right = _aligned(returns, factor_returns)
    if len(left) < 3 or len(right) < 3:
        return np.nan
    pairs = np.column_stack([left, right])
    pairs = pairs[~np.isnan(pairs).any(axis=1)]
    pairs = pairs[np.argsort(pairs[:, 1], kind="stable")]
    middle = int(np.around(len(pairs) / 2, 0))
    start, mid, end = pairs[0], pairs[middle], pairs[-1]
    factor_range = end[1] - start[1]
    start_weight = end_weight = 0.5
    if factor_range != 0:
        start_weight = (mid[1] - start[1]) / factor_range
        end_weight = (end[1] - mid[1]) / factor_range
    return start_weight * start[0] + end_weight * end[0] - mid[0]


beta_fragility_heuristic_aligned = beta_fragility_heuristic


def _rolling_size(values, window):
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(values) == 0:
        return 0, window
    actual = min(len(values), window)
    return len(values) - actual + 1, actual


def _finish_rolling(result, original, out):
    if out is not None:
        out[()] = result
        return out
    if isinstance(original, pd.Series):
        index = original.index[-len(result) :]
        if np.ndim(result) == 2:
            return pd.DataFrame(result, index=index)
        return pd.Series(result, index=index)
    return result


def _roll_unary(arr, window, symbol, out, *args):
    values = _values(arr)
    if values.ndim == 2:
        columns = [
            np.asarray(_roll_unary(values[:, i], window, symbol, None, *args))
            for i in range(values.shape[1])
        ]
        result = np.column_stack(columns) if columns else np.empty((0, 0))
        return _finish_rolling(result, arr, out)
    size, actual = _rolling_size(values, window)
    result = np.empty(size)
    if size:
        getattr(lib(), symbol)(addr(values), addr(result), len(values), actual, *args)
    return _finish_rolling(result, arr, out)


def roll_sharpe_ratio(arr, window, out=None, **kwargs):
    factor = annualization_factor(
        kwargs.pop("period", DAILY), kwargs.pop("annualization", None)
    )
    return _roll_unary(
        arr,
        window,
        "mep_roll_sharpe",
        out,
        float(kwargs.pop("risk_free", 0)),
        float(factor),
    )


def roll_annual_volatility(arr, window, out=None, **kwargs):
    factor = annualization_factor(
        kwargs.pop("period", DAILY), kwargs.pop("annualization", None)
    )
    return _roll_unary(
        arr,
        window,
        "mep_roll_volatility",
        out,
        float(factor),
        float(kwargs.pop("alpha", 2.0)),
    )


def roll_sortino_ratio(arr, window, out=None, **kwargs):
    factor = annualization_factor(
        kwargs.pop("period", DAILY), kwargs.pop("annualization", None)
    )
    return _roll_unary(
        arr,
        window,
        "mep_roll_sortino",
        out,
        float(kwargs.pop("required_return", 0)),
        float(factor),
    )


def roll_max_drawdown(arr, window, out=None, **kwargs):
    return _roll_unary(arr, window, "mep_roll_max_drawdown", out)


def roll_alpha_beta_aligned(lhs, rhs, window, out=None, **kwargs):
    left, right = _values(lhs), _values(rhs)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("rolling alpha/beta currently expects one-dimensional inputs")
    if len(left) != len(right):
        raise ValueError("returns and factor_returns must have the same length")
    size, actual = _rolling_size(left, window)
    result = np.empty((size, 2))
    factor = annualization_factor(
        kwargs.pop("period", DAILY), kwargs.pop("annualization", None)
    )
    if size:
        lib().mep_roll_alpha_beta(
            addr(left),
            addr(right),
            addr(result),
            len(left),
            actual,
            float(kwargs.pop("risk_free", 0.0)),
            float(factor),
        )
    return _finish_rolling(result, lhs, out)


def roll_alpha_beta(returns, factor_returns, window=10, **kwargs):
    left, right = _aligned(returns, factor_returns)
    return roll_alpha_beta_aligned(left, right, window, **kwargs)


def roll_alpha_aligned(lhs, rhs, window, out=None, **kwargs):
    result = np.asarray(
        roll_alpha_beta_aligned(lhs, rhs, window, None, **kwargs)
    )[:, 0]
    return _finish_rolling(result, lhs, out)


def roll_beta_aligned(lhs, rhs, window, out=None, **kwargs):
    kwargs.pop("period", None)
    kwargs.pop("annualization", None)
    result = np.asarray(
        roll_alpha_beta_aligned(lhs, rhs, window, None, **kwargs)
    )[:, 1]
    return _finish_rolling(result, lhs, out)


def roll_alpha(lhs, rhs, window, out=None, **kwargs):
    left, right = _aligned(lhs, rhs)
    return roll_alpha_aligned(left, right, window, out, **kwargs)


def roll_beta(lhs, rhs, window, out=None, **kwargs):
    left, right = _aligned(lhs, rhs)
    return roll_beta_aligned(left, right, window, out, **kwargs)


def _roll_capture(function, returns, factor_returns, window, **kwargs):
    left, right = _aligned(returns, factor_returns)
    size, actual = _rolling_size(left, window)
    result = np.array(
        [
            function(
                np.asarray(left)[i : i + actual],
                np.asarray(right)[i : i + actual],
                **kwargs,
            )
            for i in range(size)
        ]
    )
    return _finish_rolling(result, left, None)


def roll_up_capture(returns, factor_returns, window=10, **kwargs):
    return _roll_capture(up_capture, returns, factor_returns, window, **kwargs)


def roll_down_capture(returns, factor_returns, window=10, **kwargs):
    return _roll_capture(down_capture, returns, factor_returns, window, **kwargs)


def roll_up_down_capture(returns, factor_returns, window=10, **kwargs):
    return _roll_capture(up_down_capture, returns, factor_returns, window, **kwargs)
