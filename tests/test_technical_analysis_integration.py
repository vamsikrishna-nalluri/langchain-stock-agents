"""
Integration tests for get_technical_analysis.

These call the REAL Yahoo Finance API through yfinance (no mocks), so they:
  * need network access,
  * never assert exact prices (those change daily); they assert invariants,
    value ranges and cross-field consistency instead,
  * are marked `integration` so they can be excluded from fast unit-test runs.

Run:
    pytest -m integration -v                 # only these tests
    TA_STRICT=1 pytest -m integration -v     # fail (instead of skip) when Yahoo is unreachable
    pytest -m integration -k print_sample    # just print/save a sample report (see test at the bottom)

The tool is imported from tools/stock_data.py (run pytest from the project root).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tools.stock_data import get_technical_analysis

pytestmark = pytest.mark.integration

STOCKS = ["RELIANCE.NS", "TCS.NS", "AAPL"]
INDEX = "^NSEI"
TIMEFRAMES = ["daily", "weekly", "monthly"]
NA = "NA"
TOL = 1e-6


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fetch():
    """Calls the tool once per ticker per test session (avoids Yahoo rate limits)."""
    cache: dict[str, dict] = {}

    def _fetch(ticker: str) -> dict:
        if ticker not in cache:
            cache[ticker] = json.loads(get_technical_analysis.invoke({"ticker": ticker}))
        return cache[ticker]

    return _fetch


def report_for(fetch, ticker: str) -> dict:
    """Fetch a report for a known-good ticker; skip (or fail in strict mode) if Yahoo is down."""
    data = fetch(ticker)
    if "error" in data:
        msg = f"{ticker}: tool returned error {data['error']!r} (errors={data.get('errors')})"
        if os.getenv("TA_STRICT"):
            pytest.fail(msg)
        pytest.skip(msg)
    return data


def num(x):
    """float if x is a real number, None if it is 'NA'."""
    return None if x == NA else float(x)


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


# --------------------------------------------------------------------------
# 1. Contract: shape of the response
# --------------------------------------------------------------------------
def test_tool_is_registered_as_langchain_tool():
    assert get_technical_analysis.name == "get_technical_analysis"
    assert "ticker" in get_technical_analysis.args


@pytest.mark.parametrize("ticker", STOCKS)
def test_top_level_sections_present(fetch, ticker):
    data = report_for(fetch, ticker)
    for key in ("meta", "data_quality", "price_snapshot", "performance", "relative_strength",
                "recent_daily_candles", "timeframes", "multi_timeframe_summary", "errors", "notes"):
        assert key in data, f"missing top-level section {key!r}"
    assert set(data["timeframes"]) == set(TIMEFRAMES)


@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_every_timeframe_has_all_sections(fetch, ticker, tf):
    t = report_for(fetch, ticker)["timeframes"][tf]
    assert isinstance(t, dict), f"{tf} analysis failed entirely: {t!r}"
    for key in ("bars", "last_bar_date", "last_bar_is_partial", "last_bar", "trend", "momentum",
                "volatility", "volume", "support_resistance", "candlestick", "signals"):
        assert key in t, f"{tf}: missing section {key!r}"
    for key in ("moving_averages", "ma_alignment", "ma_cross", "adx_14", "supertrend_10_3"):
        assert key in t["trend"]
    for key in ("rsi_14", "macd_12_26_9", "stochastic_14_3_3", "williams_r_14", "cci_20", "roc_12_pct"):
        assert key in t["momentum"]


# --------------------------------------------------------------------------
# 2. "NA" convention: never None / NaN / inf anywhere in the output
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS + [INDEX])
def test_no_none_nan_or_inf_values(fetch, ticker):
    data = report_for(fetch, ticker)
    for path, v in leaves(data):
        assert v is not None, f"None leaked at {path}"
        assert isinstance(v, (int, float, str, bool)), f"unexpected type {type(v)} at {path}"
        if isinstance(v, float):
            assert math.isfinite(v), f"non-finite float at {path}: {v}"


# --------------------------------------------------------------------------
# 3. Data quality / freshness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS)
def test_meta_and_freshness(fetch, ticker):
    data = report_for(fetch, ticker)
    assert data["meta"]["stock_symbol"] == ticker
    last = datetime.strptime(data["meta"]["last_daily_bar"], "%Y-%m-%d")
    assert datetime.now() - last <= timedelta(days=7), "last daily bar older than 7 days (stale feed?)"


@pytest.mark.parametrize("ticker", STOCKS)
def test_data_quality_block(fetch, ticker):
    dq = report_for(fetch, ticker)["data_quality"]
    assert dq["status"] in {"OK", "OK_WITH_WARNINGS", "DEGRADED"}
    assert isinstance(dq["warnings"], list)
    daily, weekly, monthly = (dq[tf]["bars"] for tf in TIMEFRAMES)
    assert daily >= 1000, "expected years of daily history for a large-cap"
    # Resampling sanity: ~5 trading days per week, ~21 per month (loose bounds for holidays)
    assert daily / 6 <= weekly <= daily / 4
    assert daily / 30 <= monthly <= daily / 15


@pytest.mark.parametrize("ticker", STOCKS)
def test_bar_labels_match_timeframe(fetch, ticker):
    tfs = report_for(fetch, ticker)["timeframes"]
    daily_last = pd.Timestamp(tfs["daily"]["last_bar_date"])
    weekly_last = pd.Timestamp(tfs["weekly"]["last_bar_date"])
    monthly_last = pd.Timestamp(tfs["monthly"]["last_bar_date"])
    assert weekly_last.weekday() == 4, "weekly bars must be labelled on Friday"
    assert monthly_last.is_month_end, "monthly bars must be labelled on month-end"
    assert weekly_last >= daily_last and monthly_last >= daily_last


# --------------------------------------------------------------------------
# 4. Price consistency across sections
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS)
def test_price_is_consistent_across_sections(fetch, ticker):
    data = report_for(fetch, ticker)
    snap = data["price_snapshot"]
    close = snap["close"]
    assert close > 0
    assert close == data["timeframes"]["daily"]["last_bar"]["Close"]
    assert close == data["recent_daily_candles"][-1]["Close"]
    if snap["high_52w"] != NA:
        assert snap["low_52w"] - TOL <= close <= snap["high_52w"] + TOL
        assert 0 <= snap["position_in_52w_range_pct"] <= 100
    # change_pct must agree with close / previous_close
    expected = (close / snap["previous_close"] - 1) * 100
    assert snap["change_pct"] == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("ticker", STOCKS)
def test_recent_candles_are_valid_ohlc(fetch, ticker):
    candles = report_for(fetch, ticker)["recent_daily_candles"]
    assert len(candles) == 10
    dates = [c["date"] for c in candles]
    assert dates == sorted(dates) and len(set(dates)) == len(dates)
    for c in candles:
        assert c["Low"] <= min(c["Open"], c["Close"]) + TOL
        assert c["High"] >= max(c["Open"], c["Close"]) - TOL
        assert c["Volume"] >= 0


# --------------------------------------------------------------------------
# 5. Indicator value ranges (theoretical bounds)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_bounded_indicators_within_range(fetch, ticker, tf):
    t = report_for(fetch, ticker)["timeframes"][tf]
    m, trend = t["momentum"], t["trend"]
    bounds = [
        (m["rsi_14"]["value"], 0, 100, "RSI"),
        (m["stochastic_14_3_3"]["percent_k"], 0, 100, "%K"),
        (m["stochastic_14_3_3"]["percent_d"], 0, 100, "%D"),
        (m["williams_r_14"]["value"], -100, 0, "Williams %R"),
        (trend["adx_14"]["adx"], 0, 100, "ADX"),
        (trend["adx_14"]["plus_di"], 0, 100, "+DI"),
        (trend["adx_14"]["minus_di"], 0, 100, "-DI"),
    ]
    if isinstance(t["volume"], dict):
        bounds += [(t["volume"]["mfi_14"]["value"], 0, 100, "MFI"), (t["volume"]["cmf_20"], -1, 1, "CMF")]
    for value, lo, hi, name in bounds:
        if value != NA:
            assert lo <= value <= hi, f"{tf} {name}={value} outside [{lo}, {hi}]"


@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_volatility_values_are_sane(fetch, ticker, tf):
    v = report_for(fetch, ticker)["timeframes"][tf]["volatility"]
    if v["atr_14"] != NA:
        assert v["atr_14"] > 0
        assert 0 < v["atr_pct_of_price"] < 100
    if v["historical_volatility_20_annualised_pct"] != NA:
        assert 0 < v["historical_volatility_20_annualised_pct"] < 500


# --------------------------------------------------------------------------
# 6. Structural relationships between indicators
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_bollinger_bands_are_ordered(fetch, ticker, tf):
    b = report_for(fetch, ticker)["timeframes"][tf]["volatility"]["bollinger_20_2"]
    if b["upper"] != NA:
        assert b["lower"] <= b["middle"] <= b["upper"]
        assert b["bandwidth_pct"] >= 0


@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_macd_histogram_equals_macd_minus_signal(fetch, ticker, tf):
    m = report_for(fetch, ticker)["timeframes"][tf]["momentum"]["macd_12_26_9"]
    if m["macd"] != NA and m["signal"] != NA:
        assert m["histogram"] == pytest.approx(m["macd"] - m["signal"], abs=0.002)


@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_pivots_are_ordered(fetch, ticker, tf):
    p = report_for(fetch, ticker)["timeframes"][tf]["support_resistance"]["classic_pivots"]
    if p != NA:
        seq = [p[k] for k in ("S3", "S2", "S1", "P", "R1", "R2", "R3")]
        assert seq == sorted(seq)


@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_swing_levels_sit_on_correct_side_of_price(fetch, ticker, tf):
    data = report_for(fetch, ticker)
    price = data["timeframes"][tf]["last_bar"]["Close"]
    sr = data["timeframes"][tf]["support_resistance"]
    if sr["swing_resistance_above"] != NA:
        assert all(x > price for x in sr["swing_resistance_above"])
    if sr["swing_support_below"] != NA:
        assert all(x < price for x in sr["swing_support_below"])


@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_fibonacci_levels_lie_between_swing_low_and_high(fetch, ticker, tf):
    fib = report_for(fetch, ticker)["timeframes"][tf]["support_resistance"]["fibonacci"]
    if fib != NA:
        assert fib["swing_low"] < fib["swing_high"]
        assert all(fib["swing_low"] <= v <= fib["swing_high"] for v in fib["levels"].values())


@pytest.mark.parametrize("ticker", STOCKS)
def test_moving_average_fields_are_self_consistent(fetch, ticker):
    t = report_for(fetch, ticker)["timeframes"]["daily"]
    price = t["last_bar"]["Close"]
    mas = t["trend"]["moving_averages"]
    # 10 years of history -> every daily MA (even SMA200) must be computable for a large-cap
    for p in (20, 50, 100, 200):
        assert mas[f"sma_{p}"]["value"] != NA, f"SMA{p} unexpectedly NA"
    for name, ma in mas.items():
        if ma["value"] == NA:
            continue
        expected_side = "above" if price > ma["value"] else "below" if price < ma["value"] else "at"
        assert ma["price_vs"] == expected_side, f"{name}: price_vs disagrees with values"
        assert ma["distance_pct"] == pytest.approx((price / ma["value"] - 1) * 100, abs=0.02)


# --------------------------------------------------------------------------
# 7. Signal summary arithmetic
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS)
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_signal_summary_matches_details(fetch, ticker, tf):
    sig = report_for(fetch, ticker)["timeframes"][tf]["signals"]
    details, s = sig["details"], sig["summary"]
    scores = [d["score"] for d in details]
    assert s["indicators_scored"] == len(scores)
    assert s["net_score"] == sum(scores)
    assert s["bullish"] == sum(1 for x in scores if x > 0)
    assert s["bearish"] == sum(1 for x in scores if x < 0)
    assert s["bullish"] + s["bearish"] + s["neutral"] == len(scores)
    assert all(x in (-1, 0, 1) for x in scores)
    if scores:
        ratio = sum(scores) / len(scores)
        expected = "bullish" if ratio >= 0.35 else "bearish" if ratio <= -0.35 else "neutral/mixed"
        assert s["bias"] == expected
    else:
        assert s["bias"] == NA


@pytest.mark.parametrize("ticker", STOCKS)
def test_multi_timeframe_alignment_matches_biases(fetch, ticker):
    mtf = report_for(fetch, ticker)["multi_timeframe_summary"]
    biases = list(mtf["bias_by_timeframe"].values())
    known = [b for b in biases if b != NA]
    if len(known) == 3 and len(set(known)) == 1:
        assert mtf["alignment"] == f"aligned_{known[0]}"
    elif known:
        assert mtf["alignment"] == "mixed"
    else:
        assert mtf["alignment"] == NA


# --------------------------------------------------------------------------
# 8. Performance and relative strength
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", STOCKS)
def test_performance_returns_are_plausible(fetch, ticker):
    returns = report_for(fetch, ticker)["performance"]["returns_pct"]
    for horizon in ("1d", "1w", "1m", "3m", "6m", "1y", "ytd"):
        assert returns[horizon] != NA, f"{horizon} return unexpectedly NA for a 10y-history stock"
        assert -100 < returns[horizon] < 1000


@pytest.mark.parametrize("ticker", STOCKS)
def test_relative_strength_outperformance_is_stock_minus_benchmark(fetch, ticker):
    rs = report_for(fetch, ticker)["relative_strength"]
    if rs == NA:
        pytest.skip("benchmark data unavailable")
    for period, r in rs["periods"].items():
        if NA not in (r["stock_pct"], r["benchmark_pct"]):
            assert r["outperformance_pp"] == pytest.approx(r["stock_pct"] - r["benchmark_pct"], abs=0.02)


def test_indian_stock_uses_nifty_benchmark(fetch):
    rs = report_for(fetch, "RELIANCE.NS")["relative_strength"]
    if rs == NA:
        pytest.skip("benchmark data unavailable")
    assert "^NSEI" in rs["benchmark"]


# --------------------------------------------------------------------------
# 9. Edge cases: bare symbol, index (no/odd volume), invalid ticker
# --------------------------------------------------------------------------
def test_bare_symbol_resolves_to_nse(fetch):
    assert report_for(fetch, "RELIANCE")["meta"]["stock_symbol"] == "RELIANCE.NS"


def test_index_handles_missing_volume_gracefully(fetch):
    data = report_for(fetch, INDEX)
    daily = data["timeframes"]["daily"]
    if data["data_quality"]["has_usable_volume"]:
        assert isinstance(daily["volume"], dict)
    else:
        assert daily["volume"] == NA, "volume indicators must be NA when the series has no volume"
    # price-based indicators must still work
    assert daily["momentum"]["rsi_14"]["value"] != NA


def test_invalid_ticker_returns_error_payload_not_exception(fetch):
    data = fetch("ZZZZNOTAREALTICKER123")
    assert "error" in data
    assert "timeframes" not in data


# --------------------------------------------------------------------------
# 10. Visibility: print and save a real report so you can inspect the format
# --------------------------------------------------------------------------
def _dig(data, dotted: str):
    """Navigate a dotted path such as 'timeframes.daily.momentum'."""
    for part in dotted.split("."):
        data = data[part]
    return data


def test_print_sample_report(fetch, capsys):
    """
    Prints the report to the terminal (even without -s) and saves it to
    tests/output/<ticker>_technical_analysis.json.

    Environment variables:
        TA_SAMPLE_TICKER   ticker to show           (default RELIANCE.NS)
        TA_PRINT_SECTION   dotted path to print only one part, e.g.
                           timeframes.daily.momentum   or   price_snapshot
    """
    ticker = os.getenv("TA_SAMPLE_TICKER", "LT.NS")
    section = os.getenv("TA_PRINT_SECTION", "")
    data = report_for(fetch, ticker)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{ticker.replace('^', '')}_technical_analysis.json"
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    shown = _dig(data, section) if section else data
    with capsys.disabled():
        print(f"\n{'=' * 78}\nTECHNICAL ANALYSIS SAMPLE: {ticker}"
              + (f"  [section: {section}]" if section else "") + f"\n{'=' * 78}")
        if not section:
            print("Top-level layout:")
            for key, val in data.items():
                kind = (f"dict ({len(val)} keys)" if isinstance(val, dict)
                        else f"list ({len(val)} items)" if isinstance(val, list) else type(val).__name__)
                print(f"  - {key:<26} {kind}")
            print()
        print(json.dumps(shown, indent=2))
        print(f"\nFull report saved to: {out_file}\n")

    assert out_file.exists() and out_file.stat().st_size > 1000