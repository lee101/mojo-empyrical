"""Risk and performance kernels for the Python C ABI."""

from max.algorithm import parallelize
from std.math import abs, isnan, log1p, pow, sqrt
from std.sys.info import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_THRESHOLD = 1_048_576
comptime PARALLEL_WORKERS = 4


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


def mean_std(
    values: Ptr, n: Int, stride: Int, adjustment: Float64
) -> Tuple[Float64, Float64, Int]:
    var count = 0
    var mean = 0.0
    var moment2 = 0.0
    for i in range(n):
        var value = values.unsafe_load(i * stride)
        if isnan(value):
            continue
        value -= adjustment
        count += 1
        var delta = value - mean
        mean += delta / Float64(count)
        moment2 += delta * (value - mean)
    var std = nan_value()
    if count > 1:
        std = sqrt(moment2 / Float64(count - 1))
    return mean, std, count


def cumulative_final(
    values: Ptr, n: Int, stride: Int, starting_value: Float64
) -> Float64:
    if n == 0:
        return nan_value()
    var wealth = 1.0
    for i in range(n):
        var value = values.unsafe_load(i * stride)
        if not isnan(value):
            wealth *= 1.0 + value
    if starting_value == 0.0:
        return wealth - 1.0
    return wealth * starting_value


def annual_return_impl(
    values: Ptr, n: Int, stride: Int, annualization: Float64
) -> Float64:
    if n == 0:
        return nan_value()
    var ending = cumulative_final(values, n, stride, 1.0)
    return pow(ending, annualization / Float64(n)) - 1.0


def max_drawdown_impl(values: Ptr, n: Int, stride: Int) -> Float64:
    if n == 0:
        return nan_value()
    var wealth = 100.0
    var peak = 100.0
    var worst = 0.0
    for i in range(n):
        var value = values.unsafe_load(i * stride)
        if not isnan(value):
            wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        var drawdown = (wealth - peak) / peak
        if drawdown < worst:
            worst = drawdown
    return worst


def downside_impl(
    values: Ptr, n: Int, stride: Int, required: Float64, annualization: Float64
) -> Float64:
    if n == 0:
        return nan_value()
    var total = 0.0
    var count = 0
    for i in range(n):
        var value = values.unsafe_load(i * stride)
        if isnan(value):
            continue
        var diff = value - required
        if diff > 0.0:
            diff = 0.0
        total += diff * diff
        count += 1
    if count == 0:
        return nan_value()
    return sqrt(total / Float64(count)) * sqrt(annualization)


def beta_impl(
    returns: Ptr, factor: Ptr, n: Int, return_stride: Int, factor_stride: Int
) -> Float64:
    var count = 0
    var factor_mean = 0.0
    for i in range(n):
        var rv = returns.unsafe_load(i * return_stride)
        var fv = factor.unsafe_load(i * factor_stride)
        if isnan(rv) or isnan(fv):
            continue
        factor_mean += fv
        count += 1
    if count == 0:
        return nan_value()
    factor_mean /= Float64(count)
    var covariance = 0.0
    var variance = 0.0
    for i in range(n):
        var rv = returns.unsafe_load(i * return_stride)
        var fv = factor.unsafe_load(i * factor_stride)
        if isnan(rv) or isnan(fv):
            continue
        var residual = fv - factor_mean
        covariance += residual * rv
        variance += residual * residual
    if variance / Float64(count) < 1.0e-30:
        return nan_value()
    return covariance / variance


def alpha_impl(
    returns: Ptr,
    factor: Ptr,
    n: Int,
    return_stride: Int,
    factor_stride: Int,
    risk_free: Float64,
    annualization: Float64,
    beta_value: Float64,
) -> Float64:
    if n < 2:
        return nan_value()
    var total = 0.0
    var count = 0
    for i in range(n):
        var rv = returns.unsafe_load(i * return_stride)
        var fv = factor.unsafe_load(i * factor_stride)
        if isnan(rv) or isnan(fv):
            continue
        total += (rv - risk_free) - beta_value * (fv - risk_free)
        count += 1
    if count == 0:
        return nan_value()
    return pow(1.0 + total / Float64(count), annualization) - 1.0


def simple_returns_range(
    values: Ptr, result: Ptr, columns: Int, start: Int, end: Int
):
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var previous = values.unsafe_load[width=W](i)
        var following = values.unsafe_load[width=W](i + columns)
        result.unsafe_store(i, (following - previous) / previous)
        i += W
    while i < end:
        var previous = values.unsafe_load(i)
        result.unsafe_store(
            i, (values.unsafe_load(i + columns) - previous) / previous
        )
        i += 1


def cumulative_local(values: Ptr, result: Ptr, start: Int, end: Int):
    var wealth = 1.0
    for i in range(start, end):
        var value = values.unsafe_load(i)
        if not isnan(value):
            wealth *= 1.0 + value
        result.unsafe_store(i, wealth)


def cumulative_adjust(
    result: Ptr,
    start: Int,
    end: Int,
    multiplier: Float64,
    starting_value: Float64,
):
    var i = start
    var vector_end = end - (end - start) % W
    if starting_value == 0.0:
        while i < vector_end:
            result.unsafe_store(
                i, result.unsafe_load[width=W](i) * multiplier - 1.0
            )
            i += W
        while i < end:
            result.unsafe_store(
                i, result.unsafe_load(i) * multiplier - 1.0
            )
            i += 1
    else:
        var scale = multiplier * starting_value
        while i < vector_end:
            result.unsafe_store(
                i, result.unsafe_load[width=W](i) * scale
            )
            i += W
        while i < end:
            result.unsafe_store(i, result.unsafe_load(i) * scale)
            i += 1


@export("mep_simple_returns")
def mep_simple_returns(src: Int, dst: Int, rows: Int, columns: Int) abi("C"):
    var values = p(src)
    var result = p(dst)
    var n = (rows - 1) * columns

    @__parameter
    def process(worker: Int):
        var start = worker * n // PARALLEL_WORKERS
        var end = (worker + 1) * n // PARALLEL_WORKERS
        simple_returns_range(values, result, columns, start, end)

    if n >= PARALLEL_THRESHOLD:
        parallelize[process](PARALLEL_WORKERS, PARALLEL_WORKERS)
    else:
        simple_returns_range(values, result, columns, 0, n)


@export("mep_cum_returns")
def mep_cum_returns(
    src: Int, dst: Int, rows: Int, columns: Int, starting_value: Float64
) abi("C"):
    var values = p(src)
    var result = p(dst)
    if columns == 1 and rows >= PARALLEL_THRESHOLD:

        @__parameter
        def scan(worker: Int):
            var start = worker * rows // PARALLEL_WORKERS
            var end = (worker + 1) * rows // PARALLEL_WORKERS
            cumulative_local(values, result, start, end)

        parallelize[scan](PARALLEL_WORKERS, PARALLEL_WORKERS)
        var multiplier1 = result.unsafe_load(rows // 4 - 1)
        var multiplier2 = (
            multiplier1 * result.unsafe_load(rows // 2 - 1)
        )
        var multiplier3 = (
            multiplier2 * result.unsafe_load(3 * rows // 4 - 1)
        )

        @__parameter
        def adjust(worker: Int):
            var start = worker * rows // PARALLEL_WORKERS
            var end = (worker + 1) * rows // PARALLEL_WORKERS
            var multiplier = 1.0
            if worker == 1:
                multiplier = multiplier1
            elif worker == 2:
                multiplier = multiplier2
            elif worker == 3:
                multiplier = multiplier3
            cumulative_adjust(
                result, start, end, multiplier, starting_value
            )

        parallelize[adjust](PARALLEL_WORKERS, PARALLEL_WORKERS)
        return
    for column in range(columns):
        var wealth = 1.0
        for row in range(rows):
            var i = row * columns + column
            var value = values.unsafe_load(i)
            if not isnan(value):
                wealth *= 1.0 + value
            result.unsafe_store(
                i,
                wealth - 1.0 if starting_value
                == 0.0 else wealth * starting_value,
            )


@export("mep_cum_final")
def mep_cum_final(
    src: Int, n: Int, stride: Int, starting_value: Float64
) abi("C") -> Float64:
    return cumulative_final(p(src), n, stride, starting_value)


@export("mep_annual_return")
def mep_annual_return(
    src: Int, n: Int, stride: Int, annualization: Float64
) abi("C") -> Float64:
    return annual_return_impl(p(src), n, stride, annualization)


@export("mep_annual_volatility")
def mep_annual_volatility(
    src: Int, n: Int, stride: Int, annualization: Float64, alpha: Float64
) abi("C") -> Float64:
    if n < 2:
        return nan_value()
    var _, std, _ = mean_std(p(src), n, stride, 0.0)
    return std * pow(annualization, 1.0 / alpha)


@export("mep_max_drawdown")
def mep_max_drawdown(src: Int, n: Int, stride: Int) abi("C") -> Float64:
    return max_drawdown_impl(p(src), n, stride)


@export("mep_sharpe")
def mep_sharpe(
    src: Int, n: Int, stride: Int, risk_free: Float64, annualization: Float64
) abi("C") -> Float64:
    if n < 2:
        return nan_value()
    var mean, std, _ = mean_std(p(src), n, stride, risk_free)
    return mean / std * sqrt(annualization)


@export("mep_downside_risk")
def mep_downside_risk(
    src: Int,
    n: Int,
    stride: Int,
    required_return: Float64,
    annualization: Float64,
) abi("C") -> Float64:
    return downside_impl(p(src), n, stride, required_return, annualization)


@export("mep_sortino")
def mep_sortino(
    src: Int,
    n: Int,
    stride: Int,
    required_return: Float64,
    annualization: Float64,
) abi("C") -> Float64:
    if n < 2:
        return nan_value()
    var mean, _, _ = mean_std(p(src), n, stride, required_return)
    return (
        mean
        * annualization
        / downside_impl(p(src), n, stride, required_return, annualization)
    )


@export("mep_omega")
def mep_omega(
    src: Int, n: Int, stride: Int, risk_free: Float64, threshold: Float64
) abi("C") -> Float64:
    if n < 2:
        return nan_value()
    var numerator = 0.0
    var denominator = 0.0
    var values = p(src)
    for i in range(n):
        var value = values.unsafe_load(i * stride) - risk_free - threshold
        if isnan(value):
            continue
        if value > 0.0:
            numerator += value
        elif value < 0.0:
            denominator -= value
    if denominator <= 0.0:
        return nan_value()
    return numerator / denominator


@export("mep_stability")
def mep_stability(src: Int, n: Int, stride: Int) abi("C") -> Float64:
    if n < 2:
        return nan_value()
    var values = p(src)
    var count = 0
    var cumulative = 0.0
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_x2 = 0.0
    var sum_y2 = 0.0
    var sum_xy = 0.0
    for i in range(n):
        var value = values.unsafe_load(i * stride)
        if isnan(value):
            continue
        cumulative += log1p(value)
        var x = Float64(count)
        count += 1
        sum_x += x
        sum_y += cumulative
        sum_x2 += x * x
        sum_y2 += cumulative * cumulative
        sum_xy += x * cumulative
    if count < 2:
        return nan_value()
    var count_f = Float64(count)
    var covariance = count_f * sum_xy - sum_x * sum_y
    var variance_x = count_f * sum_x2 - sum_x * sum_x
    var variance_y = count_f * sum_y2 - sum_y * sum_y
    if variance_x <= 0.0 or variance_y <= 0.0:
        return nan_value()
    return covariance * covariance / (variance_x * variance_y)


@export("mep_beta")
def mep_beta(
    returns: Int,
    factor: Int,
    n: Int,
    return_stride: Int,
    factor_stride: Int,
) abi("C") -> Float64:
    return beta_impl(p(returns), p(factor), n, return_stride, factor_stride)


@export("mep_alpha")
def mep_alpha(
    returns: Int,
    factor: Int,
    n: Int,
    return_stride: Int,
    factor_stride: Int,
    risk_free: Float64,
    annualization: Float64,
    beta_value: Float64,
) abi("C") -> Float64:
    return alpha_impl(
        p(returns),
        p(factor),
        n,
        return_stride,
        factor_stride,
        risk_free,
        annualization,
        beta_value,
    )


@export("mep_excess_sharpe")
def mep_excess_sharpe(
    returns: Int, factor: Int, n: Int, return_stride: Int, factor_stride: Int
) abi("C") -> Float64:
    if n < 2:
        return nan_value()
    var r = p(returns)
    var f = p(factor)
    var count = 0
    var mean = 0.0
    var moment2 = 0.0
    for i in range(n):
        var rv = r.unsafe_load(i * return_stride)
        var fv = f.unsafe_load(i * factor_stride)
        if isnan(rv) or isnan(fv):
            continue
        var value = rv - fv
        count += 1
        var delta = value - mean
        mean += delta / Float64(count)
        moment2 += delta * (value - mean)
    var tracking_error = 0.0
    if count > 1:
        tracking_error = sqrt(moment2 / Float64(count - 1))
    return mean / tracking_error


@export("mep_roll_sharpe")
def mep_roll_sharpe(
    src: Int,
    dst: Int,
    n: Int,
    window: Int,
    risk_free: Float64,
    annualization: Float64,
) abi("C"):
    var values = p(src)
    var result = p(dst)
    var total = 0.0
    var total2 = 0.0
    var count = 0
    for i in range(n):
        var value = values.unsafe_load(i)
        if not isnan(value):
            value -= risk_free
            total += value
            total2 += value * value
            count += 1
        if i >= window:
            var old = values.unsafe_load(i - window)
            if not isnan(old):
                old -= risk_free
                total -= old
                total2 -= old * old
                count -= 1
        if i + 1 >= window:
            var mean = total / Float64(count)
            var variance = (total2 - total * total / Float64(count)) / Float64(
                count - 1
            )
            result.unsafe_store(
                i + 1 - window, mean / sqrt(variance) * sqrt(annualization)
            )


@export("mep_roll_volatility")
def mep_roll_volatility(
    src: Int,
    dst: Int,
    n: Int,
    window: Int,
    annualization: Float64,
    alpha: Float64,
) abi("C"):
    var values = p(src)
    var result = p(dst)
    var total = 0.0
    var total2 = 0.0
    var count = 0
    for i in range(n):
        var value = values.unsafe_load(i)
        if not isnan(value):
            total += value
            total2 += value * value
            count += 1
        if i >= window:
            var old = values.unsafe_load(i - window)
            if not isnan(old):
                total -= old
                total2 -= old * old
                count -= 1
        if i + 1 >= window:
            if count > 1:
                var variance = (
                    total2 - total * total / Float64(count)
                ) / Float64(count - 1)
                result.unsafe_store(
                    i + 1 - window,
                    sqrt(variance) * pow(annualization, 1.0 / alpha),
                )
            else:
                result.unsafe_store(i + 1 - window, nan_value())


@export("mep_roll_sortino")
def mep_roll_sortino(
    src: Int,
    dst: Int,
    n: Int,
    window: Int,
    required_return: Float64,
    annualization: Float64,
) abi("C"):
    var values = p(src)
    var result = p(dst)
    var total = 0.0
    var downside2 = 0.0
    var count = 0
    for i in range(n):
        var value = values.unsafe_load(i)
        if not isnan(value):
            var diff = value - required_return
            total += diff
            if diff < 0.0:
                downside2 += diff * diff
            count += 1
        if i >= window:
            var old = values.unsafe_load(i - window)
            if not isnan(old):
                var diff = old - required_return
                total -= diff
                if diff < 0.0:
                    downside2 -= diff * diff
                count -= 1
        if i + 1 >= window:
            result.unsafe_store(
                i + 1 - window,
                (total / Float64(count))
                * annualization
                / (sqrt(downside2 / Float64(count)) * sqrt(annualization)),
            )


@export("mep_roll_max_drawdown")
def mep_roll_max_drawdown(src: Int, dst: Int, n: Int, window: Int) abi("C"):
    var values = p(src)
    var result = p(dst)
    for i in range(n - window + 1):
        result.unsafe_store(
            i, max_drawdown_impl(values.unsafe_offset(i), window, 1)
        )


@export("mep_roll_alpha_beta")
def mep_roll_alpha_beta(
    returns: Int,
    factor: Int,
    dst: Int,
    n: Int,
    window: Int,
    risk_free: Float64,
    annualization: Float64,
) abi("C"):
    var r = p(returns)
    var f = p(factor)
    var result = p(dst)
    for i in range(n - window + 1):
        var returns_window = r.unsafe_offset(i)
        var factor_window = f.unsafe_offset(i)
        var b = beta_impl(returns_window, factor_window, window, 1, 1)
        result.unsafe_store(
            2 * i,
            alpha_impl(
                returns_window,
                factor_window,
                window,
                1,
                1,
                risk_free,
                annualization,
                b,
            ),
        )
        result.unsafe_store(2 * i + 1, b)
