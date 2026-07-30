import numpy as np
import pandas as pd
import pytest

import empyrical as upstream
import mojoempyrical as mojo

rng = np.random.default_rng(20260730)
returns = rng.normal(0.0004, 0.012, 2003)
factor = rng.normal(0.0002, 0.009, 2003)
returns[::137] = np.nan
factor[::181] = np.nan


def same(actual, expected, rtol=2e-9, atol=2e-12):
    assert np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def test_simple_returns_ndarray_parity():
    prices = 100 * np.cumprod(1 + np.nan_to_num(returns))
    same(mojo.simple_returns(prices), upstream.simple_returns(prices))


def test_simple_returns_matrix_parity():
    prices = np.column_stack(
        [100 * np.cumprod(1 + np.nan_to_num(returns)),
         50 * np.cumprod(1 + np.nan_to_num(factor))]
    )
    same(mojo.simple_returns(prices), upstream.simple_returns(prices))


def test_simple_returns_simd_tail_and_unaligned_input():
    storage = np.linspace(99.0, 101.0, 19)
    prices = storage[1:]
    assert prices.flags.c_contiguous
    same(mojo.simple_returns(prices), upstream.simple_returns(prices))


def test_inputs_reject_silent_float64_narrowing():
    with pytest.raises(TypeError, match="wider than float64"):
        mojo.simple_returns(np.array([1, 2], dtype=np.longdouble))
    with pytest.raises(TypeError, match="cannot represent exactly"):
        mojo.simple_returns(np.array([1 << 54, (1 << 54) + 1], dtype=np.int64))
    with pytest.raises(TypeError, match="complex"):
        mojo.simple_returns(np.array([1 + 0j, 2 + 0j]))


@pytest.mark.parametrize("output_size", [1_048_575, 1_048_579])
def test_simple_returns_parallel_threshold(output_size):
    prices = np.linspace(50.0, 150.0, output_size + 1)
    same(mojo.simple_returns(prices), upstream.simple_returns(prices))


def test_simple_returns_series_preserves_index():
    values = pd.Series(
        [100.0, 101.0, 99.0, 103.0],
        index=pd.date_range("2025-01-01", periods=4),
    )
    pd.testing.assert_series_equal(
        mojo.simple_returns(values), upstream.simple_returns(values)
    )


@pytest.mark.parametrize("starting_value", [0, 1, 100])
def test_cum_returns_parity(starting_value):
    same(
        mojo.cum_returns(returns, starting_value),
        upstream.cum_returns(returns, starting_value),
    )


def test_cum_returns_matrix_parity():
    matrix = np.column_stack([returns, factor])
    same(mojo.cum_returns(matrix), upstream.cum_returns(matrix))


def test_cum_returns_pandas_labels():
    frame = pd.DataFrame(
        {"strategy": returns[:30], "benchmark": factor[:30]},
        index=pd.date_range("2025-01-01", periods=30),
    )
    pd.testing.assert_frame_equal(mojo.cum_returns(frame), upstream.cum_returns(frame))


@pytest.mark.parametrize("frequency", ["weekly", "monthly", "quarterly", "yearly"])
def test_aggregate_returns_parity(frequency):
    series = pd.Series(
        returns[:500], index=pd.bdate_range("2023-01-02", periods=500)
    )
    pd.testing.assert_series_equal(
        mojo.aggregate_returns(series, frequency),
        upstream.aggregate_returns(series, frequency),
    )


def test_cum_returns_out_buffer():
    actual = np.empty_like(returns)
    returned = mojo.cum_returns(returns, out=actual)
    assert returned is actual
    same(actual, upstream.cum_returns(returns))


def test_cum_returns_rejects_unsafe_out_buffers():
    with pytest.raises(ValueError, match="same shape"):
        mojo.cum_returns(returns, out=np.empty(1, dtype=np.float64))
    readonly = np.empty_like(returns)
    readonly.flags.writeable = False
    with pytest.raises(TypeError, match="writable"):
        mojo.cum_returns(returns, out=readonly)
    with pytest.raises(TypeError, match="float64"):
        mojo.cum_returns(returns, out=np.empty_like(returns, dtype=np.float32))


@pytest.mark.parametrize("starting_value", [0, 1, 10])
def test_cum_returns_final_parity(starting_value):
    same(
        mojo.cum_returns_final(returns, starting_value),
        upstream.cum_returns_final(returns, starting_value),
    )


def test_cum_returns_final_dataframe():
    frame = pd.DataFrame({"a": returns, "b": factor})
    pd.testing.assert_series_equal(
        mojo.cum_returns_final(frame), upstream.cum_returns_final(frame)
    )


@pytest.mark.parametrize(
    "period,annualization",
    [("daily", None), ("weekly", None), ("monthly", None), ("daily", 365)],
)
def test_annual_return_parity(period, annualization):
    same(
        mojo.annual_return(returns, period, annualization),
        upstream.annual_return(returns, period, annualization),
    )


def test_cagr_alias_parity():
    same(mojo.cagr(returns), upstream.cagr(returns))


@pytest.mark.parametrize("alpha", [1.5, 2.0, 3.0])
def test_annual_volatility_parity(alpha):
    same(
        mojo.annual_volatility(returns, alpha=alpha),
        upstream.annual_volatility(returns, alpha=alpha),
    )


def test_max_drawdown_parity_and_known_vector():
    same(mojo.max_drawdown(returns), upstream.max_drawdown(returns))
    known = np.array([0.10, -0.20, 0.05])
    assert mojo.max_drawdown(known) == pytest.approx(-0.20)


def test_calmar_ratio_parity():
    same(mojo.calmar_ratio(returns), upstream.calmar_ratio(returns))


@pytest.mark.parametrize("risk_free", [0.0, 0.0001])
def test_sharpe_ratio_parity(risk_free):
    same(
        mojo.sharpe_ratio(returns, risk_free),
        upstream.sharpe_ratio(returns, risk_free),
    )


@pytest.mark.parametrize("required_return", [0.0, 0.0002])
def test_downside_risk_parity(required_return):
    same(
        mojo.downside_risk(returns, required_return),
        upstream.downside_risk(returns, required_return),
    )


@pytest.mark.parametrize("required_return", [0.0, 0.0002])
def test_sortino_ratio_parity(required_return):
    same(
        mojo.sortino_ratio(returns, required_return),
        upstream.sortino_ratio(returns, required_return),
    )


def test_sortino_precomputed_downside_parity():
    downside = upstream.downside_risk(returns)
    same(
        mojo.sortino_ratio(returns, _downside_risk=downside),
        upstream.sortino_ratio(returns, _downside_risk=downside),
    )


@pytest.mark.parametrize(
    "risk_free,required_return,annualization",
    [(0, 0, 252), (0.0001, 0.05, 252), (0, 0.001, 1)],
)
def test_omega_ratio_parity(risk_free, required_return, annualization):
    same(
        mojo.omega_ratio(returns, risk_free, required_return, annualization),
        upstream.omega_ratio(returns, risk_free, required_return, annualization),
    )


def test_stability_parity():
    same(
        mojo.stability_of_timeseries(returns),
        upstream.stability_of_timeseries(returns),
    )


def test_tail_ratio_parity():
    same(mojo.tail_ratio(returns), upstream.tail_ratio(returns))


@pytest.mark.parametrize("cutoff", [0.01, 0.05, 0.2])
def test_value_at_risk_parity(cutoff):
    clean = returns[~np.isnan(returns)]
    same(mojo.value_at_risk(clean, cutoff), upstream.value_at_risk(clean, cutoff))


@pytest.mark.parametrize("cutoff", [0.01, 0.05, 0.2])
def test_conditional_value_at_risk_parity(cutoff):
    clean = returns[~np.isnan(returns)]
    same(
        mojo.conditional_value_at_risk(clean, cutoff),
        upstream.conditional_value_at_risk(clean, cutoff),
    )


def test_beta_parity():
    same(mojo.beta(returns, factor), upstream.beta(returns, factor))


def test_alpha_parity():
    same(
        mojo.alpha(returns, factor, risk_free=0.0001),
        upstream.alpha(returns, factor, risk_free=0.0001),
    )


def test_alpha_beta_parity():
    same(mojo.alpha_beta(returns, factor), upstream.alpha_beta(returns, factor))


def test_aligned_factor_api_parity():
    for function in ("alpha", "beta", "alpha_beta"):
        same(
            getattr(mojo, f"{function}_aligned")(returns, factor),
            getattr(upstream, f"{function}_aligned")(returns, factor),
        )
    same(
        mojo.beta_fragility_heuristic_aligned(returns, factor),
        upstream.beta_fragility_heuristic_aligned(returns, factor),
    )


def test_excess_sharpe_parity():
    same(
        mojo.excess_sharpe(returns, factor),
        upstream.excess_sharpe(returns, factor),
    )


def test_binary_series_alignment_parity():
    index = pd.date_range("2024-01-01", periods=100)
    strategy = pd.Series(returns[:100], index=index)
    benchmark = pd.Series(factor[10:100], index=index[10:])
    same(
        mojo.alpha_beta(strategy, benchmark),
        upstream.alpha_beta(strategy, benchmark),
    )


def test_capture_ratio_parity():
    for function in ("capture", "up_capture", "down_capture", "up_down_capture"):
        same(
            getattr(mojo, function)(returns, factor),
            getattr(upstream, function)(returns, factor),
            rtol=2e-8,
        )


def test_filtered_alpha_beta_parity():
    same(
        mojo.up_alpha_beta(returns, factor),
        upstream.up_alpha_beta(returns, factor),
    )
    same(
        mojo.down_alpha_beta(returns, factor),
        upstream.down_alpha_beta(returns, factor),
    )


def test_beta_fragility_parity():
    same(
        mojo.beta_fragility_heuristic(returns, factor),
        upstream.beta_fragility_heuristic(returns, factor),
    )


@pytest.mark.parametrize("window", [20, 63, 5000])
def test_roll_sharpe_parity(window):
    same(
        mojo.roll_sharpe_ratio(returns, window),
        upstream.roll_sharpe_ratio(returns, window),
        rtol=2e-8,
    )


def test_roll_annual_volatility_parity():
    same(
        mojo.roll_annual_volatility(returns, 63),
        upstream.roll_annual_volatility(returns, 63),
        rtol=2e-8,
    )


def test_roll_sortino_parity():
    same(
        mojo.roll_sortino_ratio(returns, 63),
        upstream.roll_sortino_ratio(returns, 63),
        rtol=2e-8,
    )


def test_roll_max_drawdown_parity():
    same(
        mojo.roll_max_drawdown(returns, 63),
        upstream.roll_max_drawdown(returns, 63),
    )


def test_roll_beta_parity():
    same(mojo.roll_beta(returns, factor, 63), upstream.roll_beta(returns, factor, 63))


def test_roll_alpha_parity():
    same(
        mojo.roll_alpha(returns, factor, 63),
        upstream.roll_alpha(returns, factor, 63),
        rtol=2e-8,
    )


def test_roll_alpha_beta_parity():
    same(
        mojo.roll_alpha_beta(returns, factor, 63),
        upstream.roll_alpha_beta(returns, factor, 63),
        rtol=2e-8,
    )


def test_aligned_rolling_factor_api_parity():
    for function in ("roll_alpha", "roll_beta", "roll_alpha_beta"):
        same(
            getattr(mojo, f"{function}_aligned")(returns, factor, 63),
            getattr(upstream, f"{function}_aligned")(returns, factor, 63),
            rtol=2e-8,
        )


def test_roll_alpha_beta_aligned_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        mojo.roll_alpha_beta_aligned(returns, factor[:-1], 63)


def test_rolling_capture_parity():
    sample_returns = returns[:300]
    sample_factor = factor[:300]
    for function in ("roll_up_capture", "roll_down_capture", "roll_up_down_capture"):
        same(
            getattr(mojo, function)(sample_returns, sample_factor, 40),
            getattr(upstream, function)(sample_returns, sample_factor, 40),
            rtol=2e-8,
        )


def test_rolling_series_index_and_type():
    series = pd.Series(returns[:200], index=pd.date_range("2025-01-01", periods=200))
    expected = upstream.roll_sharpe_ratio(series, 30)
    actual = mojo.roll_sharpe_ratio(series, 30)
    pd.testing.assert_index_equal(actual.index, expected.index)
    same(actual, expected)


def test_dataframe_metrics_parity():
    frame = pd.DataFrame({"strategy": returns, "benchmark": factor})
    for function in (
        "cum_returns_final",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "downside_risk",
        "sortino_ratio",
    ):
        same(getattr(mojo, function)(frame), getattr(upstream, function)(frame))


@pytest.mark.parametrize(
    "function",
    ["annual_volatility", "max_drawdown", "sharpe_ratio", "downside_risk", "sortino_ratio"],
)
def test_scalar_out_buffers(function):
    actual = np.empty(())
    expected = np.empty(())
    actual_result = getattr(mojo, function)(returns, out=actual)
    expected_result = getattr(upstream, function)(returns, out=expected)
    same(actual, expected)
    same(actual_result, expected_result)


def test_empty_and_short_inputs():
    empty = np.array([], dtype=float)
    assert np.isnan(mojo.annual_return(empty))
    assert np.isnan(mojo.max_drawdown(empty))
    assert np.isnan(mojo.sharpe_ratio(np.array([0.01])))
    assert mojo.cum_returns(empty).size == 0
    assert np.isnan(mojo.beta(empty, empty))


def test_invalid_period_matches_upstream_error():
    with pytest.raises(ValueError):
        mojo.sharpe_ratio(returns, period="hourly")
