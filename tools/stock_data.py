"""
stock_tools.py

LangChain tools for fetching stock data via yfinance.

get_stock_snapshot returns a Screener-style snapshot where every field is
self-describing: value, unit, source ("yahoo" | "computed" | "approximate" |
"unavailable") and an optional note. Failures are isolated per data source,
so one broken endpoint never kills the whole response.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

CRORE = 1e7
MILLION = 1e6


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _num(x: Any) -> Optional[float]:
    """Coerce to a finite float, else None."""
    try:
        if x is None:
            return None
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _round(x: Any, n: int = 2) -> Optional[float]:
    v = _num(x)
    return None if v is None else round(v, n)


def _div(a: Any, b: Any) -> Optional[float]:
    a, b = _num(a), _num(b)
    if a is None or b in (None, 0):
        return None
    return a / b


def _pct(x: Any) -> Optional[float]:
    """Fraction (0.159) -> percent (15.9)."""
    v = _num(x)
    return None if v is None else round(v * 100, 2)


def _sorted_stmt(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Financial statements with the newest period first."""
    if df is None or df.empty:
        return None
    return df.sort_index(axis=1, ascending=False)


def _row(df: Optional[pd.DataFrame], *labels: str) -> Optional[float]:
    """Latest non-null value of the first matching row label."""
    if df is None:
        return None
    for label in labels:
        if label in df.index:
            s = df.loc[label].dropna()
            if not s.empty:
                return _num(s.iloc[0])
    return None


def _safe(errors: dict, name: str, fn: Callable[[], Any]) -> Any:
    """Run a fetch; record the error instead of raising."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - yfinance raises many types
        errors[name] = f"{type(e).__name__}: {e}"
        return None


def _resolve_symbol(ticker: str) -> str:
    """
    'RELIANCE' -> 'RELIANCE.NS' (NSE), falling back to BSE, then as-is.
    Symbols that already carry a suffix (or are indices) are left alone.
    """
    s = ticker.strip().upper()
    if "." in s or s.startswith("^"):
        return s
    for suffix in (".NS", ".BO", ""):
        try:
            if not yf.Ticker(s + suffix).history(period="5d").empty:
                return s + suffix
        except Exception:  # noqa: BLE001
            continue
    return s


def _money(value: Any, currency: str) -> tuple[Optional[float], str]:
    """INR -> Crore, anything else -> Million."""
    v = _num(value)
    if currency == "INR":
        return (None if v is None else round(v / CRORE, 2)), "INR Cr"
    return (None if v is None else round(v / MILLION, 2)), f"{currency} Mn"


# --------------------------------------------------------------------------
# Main tool
# --------------------------------------------------------------------------
@tool
def get_stock_snapshot(ticker: str) -> str:
    """
    Fetches a fundamentals + valuation + trading snapshot for a stock.

    Args:
        ticker: Symbol such as 'RELIANCE', 'TCS.NS' or 'AAPL'. A bare symbol is
            tried on NSE (.NS), then BSE (.BO), then as given.

    Returns:
        JSON string. Each entry in 'fields' has value, unit, source and an
        optional note. 'errors' lists data sources that failed and
        'unavailable' lists fields Yahoo Finance cannot provide.
    """
    symbol = _resolve_symbol(ticker)
    t = yf.Ticker(symbol)
    errors: dict[str, str] = {}

    # ---- fetch (each source isolated) -----------------------------------
    info = _safe(errors, "info", t.get_info) or {}
    hist = _safe(errors, "history", lambda: t.history(period="1y"))
    income = _safe(errors, "income_stmt", lambda: _sorted_stmt(t.get_income_stmt(freq="yearly")))
    balance = _safe(errors, "balance_sheet", lambda: _sorted_stmt(t.get_balance_sheet(freq="yearly")))
    cashflow = _safe(errors, "cashflow", lambda: _sorted_stmt(t.get_cashflow(freq="yearly")))
    divs = _safe(errors, "dividends", lambda: t.get_dividends())
    holders = _safe(errors, "major_holders", lambda: t.get_major_holders())

    has_hist = hist is not None and not hist.empty
    if not info and not has_hist:
        return json.dumps(
            {"error": f"No data found for '{ticker}' (tried '{symbol}'). "
                      "It may be invalid or delisted.", "errors": errors},
            indent=2,
        )

    currency = info.get("currency") or "INR"
    fields: dict[str, dict] = {}

    def add(key: str, value: Any, unit: str = "", source: str = "yahoo", note: Optional[str] = None):
        entry = {"value": value, "unit": unit, "source": source if value is not None else "unavailable"}
        if note:
            entry["note"] = note
        fields[key] = entry

    # ---- price & volume (computed from 1y history) ----------------------
    price = _num(info.get("currentPrice")) or (_num(hist["Close"].iloc[-1]) if has_hist else None)

    hi = _num(info.get("fiftyTwoWeekHigh")) or (_num(hist["High"].max()) if has_hist else None)
    lo = _num(info.get("fiftyTwoWeekLow")) or (_num(hist["Low"].min()) if has_hist else None)

    vol = hist["Volume"] if has_hist else None
    add("current_price", _round(price), currency)
    add("high_52w", _round(hi), currency)
    add("low_52w", _round(lo), currency)
    add("volume", _round(vol.iloc[-1], 0) if vol is not None else None, "shares", "yahoo",
        "Latest trading session")
    add("avg_volume_1y", _round(vol.mean(), 0) if vol is not None else None, "shares", "computed")
    add("avg_volume_1m", _round(vol.tail(21).mean(), 0) if vol is not None else None, "shares", "computed",
        "Last 21 trading sessions")
    add("avg_volume_1w", _round(vol.tail(5).mean(), 0) if vol is not None else None, "shares", "computed",
        "Last 5 trading sessions")

    # ---- size -----------------------------------------------------------
    mcap_val, mcap_unit = _money(info.get("marketCap"), currency)
    add("market_cap", mcap_val, mcap_unit)
    shares = _num(info.get("sharesOutstanding"))
    shares_val, _ = _money(shares, currency)
    add("equity_shares", shares_val, "Cr" if currency == "INR" else "Mn",
        note="Shares outstanding, in crore (INR) or million (others)")

    # ---- balance sheet items --------------------------------------------
    equity = _row(balance, "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
    total_debt = _row(balance, "Total Debt")
    total_assets = _row(balance, "Total Assets")
    cur_liab = _row(balance, "Current Liabilities", "Total Current Liabilities")
    ebit = _row(income, "EBIT", "Operating Income")
    net_income = _row(income, "Net Income", "Net Income Common Stockholders")

    book_value = _num(info.get("bookValue")) or _div(equity, shares)
    add("book_value", _round(book_value), f"{currency}/share", "yahoo" if info.get("bookValue") else "computed")

    eps = _num(info.get("trailingEps"))
    add("eps", _round(eps), f"{currency}/share", note="Trailing twelve months")

    pe = _num(info.get("trailingPE")) or _div(price, eps)
    add("stock_pe", _round(pe, 1), "x", "yahoo" if info.get("trailingPE") else "computed")

    pb = _num(info.get("priceToBook")) or _div(price, book_value)
    add("price_to_book", _round(pb), "x", "yahoo" if info.get("priceToBook") else "computed")
    add("pb_x_pe", _round(pb * pe, 0) if pb is not None and pe is not None else None, "", "computed",
        "Price-to-book multiplied by P/E")

    de = _div(total_debt, equity)
    add("debt_to_equity", _round(de), "x", "computed", "Total debt / stockholders' equity, latest annual balance sheet")

    # ---- returns --------------------------------------------------------
    capital_employed = (total_assets - cur_liab) if total_assets is not None and cur_liab is not None else None
    roce = _div(ebit, capital_employed)
    add("roce", _round(roce * 100, 1) if roce is not None else None, "%", "computed",
        "EBIT / (total assets - current liabilities), latest annual; may differ from Screener's method")

    roe = _pct(info.get("returnOnEquity"))
    if roe is None:
        r = _div(net_income, equity)
        roe = _round(r * 100, 1) if r is not None else None
        roe_src = "computed"
    else:
        roe_src = "yahoo"
    add("roe", roe, "%", roe_src)

    # ---- income / cash-flow items ---------------------------------------
    dep = _row(income, "Reconciled Depreciation") or _row(
        cashflow, "Depreciation And Amortization", "Depreciation Amortization Depletion")
    dep_val, dep_unit = _money(dep, currency)
    add("depreciation", dep_val, dep_unit, "computed", "Latest annual figure")

    add("ev_to_ebitda", _round(info.get("enterpriseToEbitda"), 1), "x")

    # ---- dividends ------------------------------------------------------
    div_yield = None
    if divs is not None and len(divs) and price:
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        ttm_div = _num(divs[divs.index >= cutoff].sum())
        div_yield = _round(ttm_div / price * 100) if ttm_div is not None else None
    elif divs is not None and price:
        div_yield = 0.0
    add("dividend_yield", div_yield, "%", "computed",
        "Dividends paid in last 365 days / current price (avoids Yahoo's inconsistent yield units)")

    # ---- holdings -------------------------------------------------------
    insiders = institutions = None
    if holders is not None and not holders.empty:
        col = holders.columns[0]
        insiders = _pct(holders[col].get("insidersPercentHeld"))
        institutions = _pct(holders[col].get("institutionsPercentHeld"))
    add("promoter_holding", insiders, "%", "approximate",
        "Yahoo's 'insiders' figure; closest available proxy for promoter holding")
    add("institutional_holding", institutions, "%", "yahoo",
        "FII + DII + others combined; Yahoo does not split them for Indian stocks")
    public = round(100 - insiders - institutions, 2) if insiders is not None and institutions is not None else None
    add("public_holding", public, "%", "approximate", "100 - insiders - institutions")

    # ---- not available from Yahoo ---------------------------------------
    unavailable = {
        "fii_holding": "Not provided by Yahoo; needs NSE/BSE shareholding-pattern data",
        "dii_holding": "Not provided by Yahoo; needs NSE/BSE shareholding-pattern data",
        "change_in_promoter_holding": "Needs quarterly shareholding-pattern history",
        "face_value": "Not in Yahoo data",
        "industry_pe": "Not provided; would need a peer-median calculation or another source",
    }
    for k, note in unavailable.items():
        add(k, None, "", "unavailable", note)

    result = {
        "stock_name": info.get("longName") or info.get("shortName"),
        "stock_symbol": symbol,
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "currency": currency,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": fields,
        "unavailable": list(unavailable),
        "errors": errors,
    }
    return json.dumps(result, indent=2, default=str)


# --------------------------------------------------------------------------
# Original tool (unchanged)
# --------------------------------------------------------------------------
@tool
def get_stock_history(ticker: str, days: int = 30) -> str:
    """
    Fetches historical daily closing prices and volume for a given stock ticker.
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        days: The number of past trading days to retrieve (default is 30).
    Returns:
        A string representation of the stock's historical data, or an error message.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        if hist.empty:
            return f"Error: No data found for ticker {ticker}. It might be invalid or delisted."
        essential_data = hist[["Close", "Volume"]]
        data_str = essential_data.to_string()
        return f"Historical Data for {ticker} over the last {days} days:\n{data_str}"
    except Exception as e:
        return f"An error occurred while fetching data for {ticker}: {str(e)}"


"""
technical_tools.py

LangChain tool: get_technical_analysis(ticker)

Pipeline
  1. Fetch 10y of daily bars (split-adjusted, NOT dividend-adjusted, to match
     charting platforms such as TradingView).
  2. Validate and clean (NaNs, duplicates, non-positive prices, OHLC
     inconsistencies, zero-volume series, suspicious jumps, stale data).
  3. Build weekly and monthly bars from the validated daily bars, so all three
     timeframes come from one consistent dataset.
  4. Compute indicators per timeframe, range-check them, and emit "NA" for
     anything that cannot be computed or fails verification.
  5. Add a rule-based signal summary per timeframe and across timeframes.

Indicator conventions: Wilder smoothing (RSI, ATR, ADX) and SMA-seeded EMAs
(MACD), the same as TA-Lib / StockCharts.

Requires stock_tools.py alongside it (for symbol resolution).
"""

import json
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


NA = "NA"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Per-timeframe settings: moving-average periods, annualisation factor and
# the lookback (bars) used for swing levels, Fibonacci and squeeze ranking.
TF_CONFIG = {
    "daily": {"ma": [20, 50, 100, 200], "ann": 252, "lookback": 120},
    "weekly": {"ma": [10, 20, 40], "ann": 52, "lookback": 52},
    "monthly": {"ma": [6, 12, 24], "ann": 12, "lookback": 36},
}


# ==========================================================================
# Generic helpers
# ==========================================================================
def na(x: Any, nd: int = 2):
    """Rounded float, or 'NA' when missing / non-finite."""
    v = _num(x)
    return NA if v is None else round(v, nd)


def _last(s: Optional[pd.Series]) -> Optional[float]:
    """Last value of a series as float, or None. Never forward-fills."""
    if s is None or len(s) == 0:
        return None
    return _num(s.iloc[-1])


def _date(ts: Any) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return NA


def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0


def _vs(price: Optional[float], level: Optional[float]) -> str:
    if price is None or level is None:
        return NA
    return "above" if price > level else "below" if price < level else "at"


def _bounded(name: str, v: Optional[float], lo: float, hi: float, warn: list, tf: str) -> Optional[float]:
    """Reject values outside their theoretical range (guards against bad maths/data)."""
    if v is None:
        return None
    if v < lo - 1e-6 or v > hi + 1e-6:
        warn.append(f"{tf}: {name}={v:.2f} outside [{lo}, {hi}], set to NA")
        return None
    return v


def _sanitize(o: Any) -> Any:
    """Final gate: numpy -> python, NaN/inf/None -> 'NA'."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if o is None:
        return NA
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return NA if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


# ==========================================================================
# Indicator library (pure functions on pandas objects)
# ==========================================================================
def _recursive_seeded(x: pd.Series, n: int, alpha: float) -> pd.Series:
    """
    y[seed] = SMA of first n valid values; then y[i] = y[i-1] + alpha * (x[i] - y[i-1]).
    alpha = 1/n  -> Wilder smoothing;  alpha = 2/(n+1) -> standard EMA.
    """
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.flatnonzero(~np.isnan(v))
    if len(valid) == 0:
        return pd.Series(out, index=x.index)
    s = int(valid[0])
    if s + n > len(v):
        return pd.Series(out, index=x.index)
    out[s + n - 1] = v[s:s + n].mean()
    for i in range(s + n, len(v)):
        out[i] = out[i - 1] + alpha * (v[i] - out[i - 1])
    return pd.Series(out, index=x.index)


def wilder(x: pd.Series, n: int) -> pd.Series:
    return _recursive_seeded(x, n, 1.0 / n)


def ema(x: pd.Series, n: int) -> pd.Series:
    return _recursive_seeded(x, n, 2.0 / (n + 1))


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    ag = wilder(d.clip(lower=0), n)
    al = wilder((-d).clip(lower=0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + ag / al)
    out = out.mask((al == 0) & (ag > 0), 100.0)
    out = out.mask((al == 0) & (ag == 0), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def stochastic(df: pd.DataFrame, k: int = 14, smooth: int = 3, d: int = 3):
    ll = df.Low.rolling(k).min()
    hh = df.High.rolling(k).max()
    rng = (hh - ll).where((hh - ll) != 0)
    raw = 100 * (df.Close - ll) / rng
    slow_k = raw.rolling(smooth).mean()
    return slow_k, slow_k.rolling(d).mean()


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    ll = df.Low.rolling(n).min()
    hh = df.High.rolling(n).max()
    rng = (hh - ll).where((hh - ll) != 0)
    return -100 * (hh - df.Close) / rng


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df.High + df.Low + df.Close) / 3
    m = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    return (tp - m) / (0.015 * mad.where(mad != 0))


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # needs a previous close
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14):
    up = df.High.diff()
    dn = -df.Low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    a = atr(df, n)
    pdi = 100 * wilder(plus_dm, n) / a
    mdi = 100 * wilder(minus_dm, n) / a
    s = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / s.where(s != 0)
    dx = dx.mask(s == 0, 0.0)
    return wilder(dx, n), pdi, mdi


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    a = atr(df, n)
    hl2 = (df.High + df.Low) / 2
    ub = (hl2 + mult * a).to_numpy()
    lb = (hl2 - mult * a).to_numpy()
    c = df.Close.to_numpy()
    N = len(df)
    st = np.full(N, np.nan)
    dr = np.full(N, np.nan)
    valid = np.flatnonzero(~np.isnan(ub))
    if len(valid) == 0:
        return pd.Series(st, index=df.index), pd.Series(dr, index=df.index)
    s = int(valid[0])
    fu, fl = ub.copy(), lb.copy()
    dr[s] = 1.0 if c[s] > (ub[s] + lb[s]) / 2 else -1.0
    st[s] = fl[s] if dr[s] == 1 else fu[s]
    for i in range(s + 1, N):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            dr[i] = 1.0
        elif c[i] < fl[i - 1]:
            dr[i] = -1.0
        else:
            dr[i] = dr[i - 1]
        st[i] = fl[i] if dr[i] == 1 else fu[i]
    return pd.Series(st, index=df.index), pd.Series(dr, index=df.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    width = (up - lo).where((up - lo) != 0)
    pct_b = (close - lo) / width
    bw = (up - lo) / mid * 100
    return up, mid, lo, pct_b, bw


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df.Close.diff()).fillna(0) * df.Volume).cumsum()


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df.High + df.Low + df.Close) / 3
    rmf = tp * df.Volume
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + pos / neg)
    out = out.mask((neg == 0) & (pos > 0), 100.0)
    out = out.mask((neg == 0) & (pos == 0), 50.0)
    return out.where(tp.rolling(n + 1).count() == n + 1)


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    rng = (df.High - df.Low).where((df.High - df.Low) != 0)
    mfm = (((df.Close - df.Low) - (df.High - df.Close)) / rng).fillna(0.0)
    vs = df.Volume.rolling(n).sum()
    return (mfm * df.Volume).rolling(n).sum() / vs.where(vs != 0)


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df.High + df.Low + df.Close) / 3
    vs = df.Volume.rolling(n).sum()
    return (tp * df.Volume).rolling(n).sum() / vs.where(vs != 0)


def classic_pivots(h: float, l: float, c: float) -> dict:
    p = (h + l + c) / 3
    return {
        "S3": l - 2 * (h - p), "S2": p - (h - l), "S1": 2 * p - h,
        "P": p,
        "R1": 2 * p - l, "R2": p + (h - l), "R3": h + 2 * (p - l),
    }


def swing_levels(df: pd.DataFrame, k: int = 5, lookback: int = 120):
    """Fractal swing highs/lows: a bar that is the extreme of +/- k neighbours."""
    d = df.tail(lookback)
    h, l = d.High.to_numpy(), d.Low.to_numpy()
    highs, lows = [], []
    for i in range(k, len(d) - k):
        if h[i] == h[i - k:i + k + 1].max():
            highs.append(float(h[i]))
        if l[i] == l[i - k:i + k + 1].min():
            lows.append(float(l[i]))
    return highs, lows


def _cluster(levels: list[float], price: float, tol: float = 0.003, maxn: int = 3) -> list[float]:
    """Nearest-first levels, merging ones within tol (0.3%) of an already kept level."""
    kept: list[float] = []
    for lv in sorted(set(levels), key=lambda x: abs(x - price)):
        if all(abs(lv / k - 1) > tol for k in kept):
            kept.append(lv)
        if len(kept) == maxn:
            break
    return kept


def fibonacci(df: pd.DataFrame, lookback: int) -> Optional[dict]:
    d = df.tail(lookback)
    if len(d) < 10:
        return None
    hi, lo = float(d.High.max()), float(d.Low.min())
    if hi <= lo:
        return None
    hi_pos = int(np.argmax(d.High.to_numpy()))
    lo_pos = int(np.argmin(d.Low.to_numpy()))
    up = lo_pos < hi_pos  # low came first -> measuring a pullback from an up-move
    ratios = {"23.6%": 0.236, "38.2%": 0.382, "50%": 0.5, "61.8%": 0.618, "78.6%": 0.786}
    levels = {k: (hi - r * (hi - lo)) if up else (lo + r * (hi - lo)) for k, r in ratios.items()}
    return {"swing_high": hi, "swing_low": lo,
            "leg": "up-move (retracement from high)" if up else "down-move (retracement from low)",
            "levels": levels}


def candle_patterns(df: pd.DataFrame) -> list[str]:
    if len(df) < 2:
        return []
    o, h, l, c = (float(df[x].iloc[-1]) for x in ("Open", "High", "Low", "Close"))
    po, pc = float(df.Open.iloc[-2]), float(df.Close.iloc[-2])
    rng = h - l
    if rng <= 0:
        return ["flat_bar"]
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    out: list[str] = []
    if body <= 0.1 * rng:
        out.append("doji")
    if body > 0 and lower >= 2 * body and upper <= 0.1 * rng:
        out.append("hammer_like (long lower shadow)")
    if body > 0 and upper >= 2 * body and lower <= 0.1 * rng:
        out.append("shooting_star_like (long upper shadow)")
    if body >= 0.9 * rng:
        out.append("bullish_marubozu" if c > o else "bearish_marubozu")
    if pc < po and c > o and c >= po and o <= pc:
        out.append("bullish_engulfing")
    if pc > po and c < o and o >= pc and c <= po:
        out.append("bearish_engulfing")
    return out


# ==========================================================================
# Data preparation and validation
# ==========================================================================
def clean_ohlcv(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and repair raw bars. Returns (clean_df, issues)."""
    missing = [c for c in OHLCV if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    issues: dict[str, Any] = {"rows_raw": int(len(raw))}
    df = raw[OHLCV].copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()

    dup = df.index.duplicated(keep="last")
    issues["dropped_duplicate_dates"] = int(dup.sum())
    df = df[~dup].sort_index()

    ohlc = ["Open", "High", "Low", "Close"]
    nan_rows = df[ohlc].isna().any(axis=1)
    issues["dropped_nan_ohlc"] = int(nan_rows.sum())
    df = df[~nan_rows]

    nonpos = (df[ohlc] <= 0).any(axis=1)
    issues["dropped_non_positive_price"] = int(nonpos.sum())
    df = df[~nonpos]

    hi, lo = df[ohlc].max(axis=1), df[ohlc].min(axis=1)
    viol = (df.High < hi) | (df.Low > lo)
    issues["repaired_ohlc_inconsistency"] = int(viol.sum())
    df.loc[viol, "High"] = hi[viol]
    df.loc[viol, "Low"] = lo[viol]

    issues["nan_volume_filled_with_zero"] = int(df.Volume.isna().sum())
    df["Volume"] = df.Volume.fillna(0.0).clip(lower=0.0)

    jumps = df.Close.pct_change().abs() > 0.40
    issues["jumps_over_40pct"] = [_date(d) for d in df.index[jumps.to_numpy()][:5]]
    issues["rows_clean"] = int(len(df))
    return df, issues


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    try:
        out = df.resample(rule).agg(agg)
    except ValueError:  # older pandas has no 'ME'
        out = df.resample("M" if rule == "ME" else rule).agg(agg)
    return out.dropna(subset=["Close"])


def _is_partial(tf: str, label: pd.Timestamp, last_daily: pd.Timestamp,
                today: pd.Timestamp, market_open: bool) -> bool:
    """Is the final bar of this timeframe still forming?"""
    if market_open and last_daily == today:
        return True
    if tf == "daily":
        return False
    if label <= last_daily:
        return False
    start = (last_daily + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (label + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return bool(np.busday_count(start, end) > 0)  # trading days still to come in the period


# ==========================================================================
# Per-timeframe analysis
# ==========================================================================
def analyze_timeframe(df: pd.DataFrame, tf: str, has_volume: bool, partial: bool, warn: list) -> dict:
    cfg = TF_CONFIG[tf]
    n = len(df)
    out: dict[str, Any] = {
        "bars": n,
        "last_bar_date": _date(df.index[-1]) if n else NA,
        "last_bar_is_partial": partial,
    }
    if n < 2:
        out["status"] = "insufficient_data"
        return out

    close, high, low, vol = df.Close, df.High, df.Low, df.Volume
    price = _last(close)
    out["last_bar"] = {k: na(df[k].iloc[-1], 2 if k != "Volume" else 0)
                       for k in OHLCV}

    sigs: list[dict] = []

    def sig(name: str, reading: Any, signal: str, score: int):
        sigs.append({"indicator": name, "reading": reading, "signal": signal, "score": score})

    # ---------------- Trend ----------------
    ma_block: dict[str, Any] = {}
    sma_vals: dict[int, Optional[float]] = {}
    for p in cfg["ma"]:
        sm, em = sma(close, p), ema(close, p)
        sv, ev = _last(sm), _last(em)
        sma_vals[p] = sv
        slope = _pct_diff(sv, _num(sm.iloc[-6])) if n > p + 5 else None
        ma_block[f"sma_{p}"] = {"value": na(sv), "price_vs": _vs(price, sv),
                                "distance_pct": na(_pct_diff(price, sv)), "slope_5bar_pct": na(slope)}
        ma_block[f"ema_{p}"] = {"value": na(ev), "price_vs": _vs(price, ev),
                                "distance_pct": na(_pct_diff(price, ev))}
        if sv is not None and price is not None:
            sig(f"Price vs SMA{p}", f"{price:.2f} vs {sv:.2f}", "bullish" if price > sv else "bearish",
                1 if price > sv else -1)

    vals = [sma_vals[p] for p in cfg["ma"]]  # short -> long
    if all(v is not None for v in vals):
        if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            alignment, sc = "bullish_stack (short above long)", 1
        elif all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            alignment, sc = "bearish_stack (short below long)", -1
        else:
            alignment, sc = "mixed", 0
        sig("SMA alignment", alignment, alignment.split("_")[0].split(" ")[0] if sc else "neutral", sc)
    else:
        alignment = NA

    ms, ml = cfg["ma"][1], cfg["ma"][-1]
    diff = (sma(close, ms) - sma(close, ml)).dropna()
    cross: dict[str, Any] = {"pair": f"SMA{ms}/SMA{ml}", "state": NA, "last_cross": NA}
    if len(diff):
        sgn = np.sign(diff)
        cross["state"] = "bullish (short above long)" if sgn.iloc[-1] > 0 else "bearish (short below long)"
        chg = (sgn != sgn.shift(1)) & sgn.shift(1).notna()
        if chg.any():
            li = chg[chg].index[-1]
            cross["last_cross"] = {"type": "golden" if sgn.loc[li] > 0 else "death", "date": _date(li),
                                   "bars_ago": int(len(diff) - 1 - diff.index.get_loc(li))}
        else:
            cross["last_cross"] = "none in available data"
        sig(f"SMA{ms}/SMA{ml} cross state", cross["state"], "bullish" if sgn.iloc[-1] > 0 else "bearish",
            1 if sgn.iloc[-1] > 0 else -1)

    adx_s, pdi_s, mdi_s = adx(df)
    adx_v = _bounded("ADX", _last(adx_s), 0, 100, warn, tf)
    pdi_v = _bounded("+DI", _last(pdi_s), 0, 100, warn, tf)
    mdi_v = _bounded("-DI", _last(mdi_s), 0, 100, warn, tf)
    adx_block: dict[str, Any] = {"adx": na(adx_v), "plus_di": na(pdi_v), "minus_di": na(mdi_v),
                                 "strength": NA, "direction": NA}
    if adx_v is not None:
        adx_block["strength"] = ("absent/weak (<20)" if adx_v < 20 else "emerging (20-25)" if adx_v < 25
                                 else "strong (25-50)" if adx_v < 50 else "very strong (>50)")
        if pdi_v is not None and mdi_v is not None:
            adx_block["direction"] = "up" if pdi_v > mdi_v else "down"
            if adx_v >= 20:
                sig("ADX/DI", f"ADX {adx_v:.1f}, +DI {pdi_v:.1f}, -DI {mdi_v:.1f}",
                    "bullish" if pdi_v > mdi_v else "bearish", 1 if pdi_v > mdi_v else -1)
            else:
                sig("ADX/DI", f"ADX {adx_v:.1f} (no clear trend)", "neutral", 0)

    st_s, st_d = supertrend(df)
    st_v, st_dir = _last(st_s), _last(st_d)
    st_block: dict[str, Any] = {"value": na(st_v), "direction": NA, "bars_since_flip": NA}
    if st_dir is not None:
        st_block["direction"] = "bullish" if st_dir > 0 else "bearish"
        dd = st_d.dropna()
        flips = dd != dd.shift(1)
        flips.iloc[0] = False
        st_block["bars_since_flip"] = int(len(dd) - 1 - dd.index.get_loc(flips[flips].index[-1])) if flips.any() else NA
        sig("Supertrend(10,3)", st_block["direction"], st_block["direction"], 1 if st_dir > 0 else -1)

    out["trend"] = {"moving_averages": ma_block, "ma_alignment": alignment, "ma_cross": cross,
                    "adx_14": adx_block, "supertrend_10_3": st_block}

    # ---------------- Momentum ----------------
    rsi_v = _bounded("RSI", _last(rsi(close)), 0, 100, warn, tf)
    rsi_zone = NA
    if rsi_v is not None:
        rsi_zone = "overbought (>=70)" if rsi_v >= 70 else "oversold (<=30)" if rsi_v <= 30 \
            else "bullish bias (50-70)" if rsi_v >= 50 else "bearish bias (30-50)"
        sig("RSI(14)", f"{rsi_v:.1f}", rsi_zone.split(" ")[0].split("_")[0],
            0 if (rsi_v >= 70 or rsi_v <= 30) else 1 if rsi_v >= 50 else -1)

    m_line, m_sig, m_hist = macd(close)
    ml_v, ms_v, mh_v = _last(m_line), _last(m_sig), _last(m_hist)
    macd_state = NA
    if ml_v is not None and ms_v is not None:
        macd_state = "bullish (MACD above signal)" if ml_v > ms_v else "bearish (MACD below signal)"
        sig("MACD(12,26,9)", f"MACD {ml_v:.2f}, signal {ms_v:.2f}", "bullish" if ml_v > ms_v else "bearish",
            1 if ml_v > ms_v else -1)
    macd_cross = NA
    mh = m_hist.dropna()
    if len(mh) > 1:
        sg = np.sign(mh)
        ch = (sg != sg.shift(1)) & sg.shift(1).notna() & (sg != 0)
        if ch.any():
            li = ch[ch].index[-1]
            macd_cross = {"type": "bullish" if sg.loc[li] > 0 else "bearish", "date": _date(li),
                          "bars_ago": int(len(mh) - 1 - mh.index.get_loc(li))}
        else:
            macd_cross = "none in available data"

    k_s, d_s = stochastic(df)
    k_v = _bounded("Stoch %K", _last(k_s), 0, 100, warn, tf)
    d_v = _bounded("Stoch %D", _last(d_s), 0, 100, warn, tf)
    stoch_zone = NA
    if k_v is not None:
        stoch_zone = "overbought (>80)" if k_v > 80 else "oversold (<20)" if k_v < 20 else "neutral zone"
        if d_v is not None:
            sig("Stochastic(14,3,3)", f"%K {k_v:.1f}, %D {d_v:.1f}", "bullish" if k_v > d_v else "bearish",
                1 if k_v > d_v else -1)

    wr_v = _bounded("Williams %R", _last(williams_r(df)), -100, 0, warn, tf)
    cci_v = _last(cci(df))
    roc_v = _last(close.pct_change(12) * 100)
    if cci_v is not None:
        sig("CCI(20)", f"{cci_v:.1f}", "bullish" if cci_v > 0 else "bearish", 1 if cci_v > 0 else -1)
    if roc_v is not None:
        sig("ROC(12)", f"{roc_v:.2f}%", "bullish" if roc_v > 0 else "bearish", 1 if roc_v > 0 else -1)

    out["momentum"] = {
        "rsi_14": {"value": na(rsi_v), "zone": rsi_zone},
        "macd_12_26_9": {"macd": na(ml_v, 3), "signal": na(ms_v, 3), "histogram": na(mh_v, 3),
                         "state": macd_state, "last_cross": macd_cross},
        "stochastic_14_3_3": {"percent_k": na(k_v), "percent_d": na(d_v), "zone": stoch_zone},
        "williams_r_14": {"value": na(wr_v),
                          "zone": NA if wr_v is None else "overbought (>-20)" if wr_v > -20
                          else "oversold (<-80)" if wr_v < -80 else "neutral zone"},
        "cci_20": {"value": na(cci_v),
                   "zone": NA if cci_v is None else "overbought (>100)" if cci_v > 100
                   else "oversold (<-100)" if cci_v < -100 else "neutral zone"},
        "roc_12_pct": na(roc_v),
    }

    # ---------------- Volatility ----------------
    atr_v = _last(atr(df))
    up_s, mid_s, lo_s, pb_s, bw_s = bollinger(close)
    up_v, mid_v, lo_v, pb_v, bw_v = (_last(s) for s in (up_s, mid_s, lo_s, pb_s, bw_s))
    if None not in (up_v, mid_v, lo_v) and not (lo_v <= mid_v <= up_v):
        warn.append(f"{tf}: Bollinger bands failed ordering check, set to NA")
        up_v = mid_v = lo_v = pb_v = bw_v = None
    bb_pos = NA
    if pb_v is not None:
        bb_pos = ("above upper band" if pb_v > 1 else "below lower band" if pb_v < 0
                  else "upper half" if pb_v >= 0.5 else "lower half")
    squeeze = NA
    bw_rank = NA
    bwt = bw_s.dropna().tail(cfg["lookback"])
    if bw_v is not None and len(bwt) >= 20:
        r = float((bwt <= bw_v).mean() * 100)
        bw_rank = round(r, 1)
        squeeze = bool(r <= 20)
    lr = np.log(close / close.shift(1))
    hv = _last(lr.rolling(20).std(ddof=1) * math.sqrt(cfg["ann"]) * 100)
    out["volatility"] = {
        "atr_14": na(atr_v), "atr_pct_of_price": na(None if atr_v is None or not price else atr_v / price * 100),
        "bollinger_20_2": {"upper": na(up_v), "middle": na(mid_v), "lower": na(lo_v),
                           "percent_b": na(pb_v, 3), "bandwidth_pct": na(bw_v),
                           "position": bb_pos, "bandwidth_percentile": bw_rank, "squeeze": squeeze},
        "historical_volatility_20_annualised_pct": na(hv),
    }

    # ---------------- Volume ----------------
    if has_volume:
        v20 = _last(vol.rolling(20).mean())
        vr = None if v20 in (None, 0) else _last(vol) / v20
        obv_s = obv(df)
        obv_chg = None
        obv_trend = NA
        confirm = NA
        if n > 21:
            obv_chg = float(obv_s.iloc[-1] - obv_s.iloc[-21])
            obv_trend = "rising" if obv_chg > 0 else "falling" if obv_chg < 0 else "flat"
            p_chg = float(close.iloc[-1] - close.iloc[-21])
            confirm = ("confirming" if (obv_chg > 0) == (p_chg > 0) else "diverging (price and OBV disagree)")
            if obv_chg != 0:
                sig("OBV (20-bar trend)", obv_trend, "bullish" if obv_chg > 0 else "bearish",
                    1 if obv_chg > 0 else -1)
        mfi_v = _bounded("MFI", _last(mfi(df)), 0, 100, warn, tf)
        cmf_v = _bounded("CMF", _last(cmf(df)), -1, 1, warn, tf)
        vw_v = _last(rolling_vwap(df))
        if cmf_v is not None:
            sig("CMF(20)", f"{cmf_v:.3f}", "bullish" if cmf_v > 0.05 else "bearish" if cmf_v < -0.05 else "neutral",
                1 if cmf_v > 0.05 else -1 if cmf_v < -0.05 else 0)
        out["volume"] = {
            "last_volume": na(vol.iloc[-1], 0), "avg_volume_20": na(v20, 0),
            "volume_ratio_vs_avg20": na(vr),
            "obv": na(_last(obv_s), 0), "obv_20bar_change": na(obv_chg, 0), "obv_trend": obv_trend,
            "obv_vs_price_20bar": confirm,
            "mfi_14": {"value": na(mfi_v),
                       "zone": NA if mfi_v is None else "overbought (>80)" if mfi_v > 80
                       else "oversold (<20)" if mfi_v < 20 else "neutral zone"},
            "cmf_20": na(cmf_v, 3),
            "rolling_vwap_20": {"value": na(vw_v), "price_vs": _vs(price, vw_v)},
        }
        if out["last_bar_is_partial"]:
            out["volume"]["note"] = "Last bar is still forming; its volume is incomplete, so volume_ratio is understated."
    else:
        out["volume"] = NA
        out["volume_note"] = "No usable volume in this series (e.g. index), volume indicators set to NA"

    # ---------------- Support / resistance ----------------
    sr: dict[str, Any] = {}
    # Pivots come from the last COMPLETED bar. If the final bar is still forming, use the one before it.
    pv = df.iloc[-2] if (partial and n >= 3) else df.iloc[-1]
    pv_date = df.index[-2] if (partial and n >= 3) else df.index[-1]
    piv = classic_pivots(float(pv.High), float(pv.Low), float(pv.Close))
    order = [piv[k] for k in ("S3", "S2", "S1", "P", "R1", "R2", "R3")]
    if all(order[i] <= order[i + 1] + 1e-9 for i in range(6)):
        sr["classic_pivots"] = {"based_on_bar": _date(pv_date), **{k: na(v) for k, v in piv.items()}}
    else:
        warn.append(f"{tf}: pivot ordering check failed, set to NA")
        sr["classic_pivots"] = NA
    highs, lows = swing_levels(df, 5, cfg["lookback"])
    if price is not None:
        sr["swing_resistance_above"] = [na(x) for x in _cluster([x for x in highs if x > price], price)] or NA
        sr["swing_support_below"] = [na(x) for x in _cluster([x for x in lows if x < price], price)] or NA
    sr["swing_note"] = f"Fractal swings (5 bars each side) within last {cfg['lookback']} bars; the latest 5 bars can't be confirmed yet."
    fib = fibonacci(df, cfg["lookback"])
    sr["fibonacci"] = NA if fib is None else {
        "lookback_bars": cfg["lookback"], "swing_high": na(fib["swing_high"]), "swing_low": na(fib["swing_low"]),
        "leg": fib["leg"], "levels": {k: na(v) for k, v in fib["levels"].items()}}
    out["support_resistance"] = sr

    # ---------------- Candlestick ----------------
    pats = candle_patterns(df)
    out["candlestick"] = {"last_bar_patterns": pats or ["none"],
                          "note": "Single/two-bar shapes only; interpret with trend context."}

    # ---------------- Signal summary ----------------
    scored = [s for s in sigs if s["score"] is not None]
    total = sum(s["score"] for s in scored)
    cnt = len(scored)
    ratio = total / cnt if cnt else None
    out["signals"] = {
        "details": sigs,
        "summary": {
            "bullish": sum(1 for s in scored if s["score"] > 0),
            "bearish": sum(1 for s in scored if s["score"] < 0),
            "neutral": sum(1 for s in scored if s["score"] == 0),
            "net_score": total, "indicators_scored": cnt,
            "score_ratio": na(ratio, 2),
            "bias": NA if ratio is None else "bullish" if ratio >= 0.35 else "bearish" if ratio <= -0.35 else "neutral/mixed",
        },
    }
    return out


# ==========================================================================
# Cross-timeframe sections (snapshot, performance, relative strength)
# ==========================================================================
def _ret_from(close: pd.Series, offset: pd.DateOffset | pd.Timedelta) -> Optional[float]:
    last_dt = close.index[-1]
    target = last_dt - offset
    if target < close.index[0]:
        return None
    base = _num(close.asof(target))
    return _pct_diff(_num(close.iloc[-1]), base)


HORIZONS = {
    "1w": pd.Timedelta(days=7), "1m": pd.DateOffset(months=1), "3m": pd.DateOffset(months=3),
    "6m": pd.DateOffset(months=6), "1y": pd.DateOffset(years=1), "3y": pd.DateOffset(years=3),
}


def build_snapshot(daily: pd.DataFrame, partial: bool) -> dict:
    last = daily.iloc[-1]
    prev = _num(daily.Close.iloc[-2]) if len(daily) > 1 else None
    px = _num(last.Close)
    snap: dict[str, Any] = {
        "date": _date(daily.index[-1]), "is_partial_bar": partial,
        "open": na(last.Open), "high": na(last.High), "low": na(last.Low), "close": na(px),
        "volume": na(last.Volume, 0), "previous_close": na(prev),
        "change": na(None if prev is None or px is None else px - prev),
        "change_pct": na(_pct_diff(px, prev)),
        "gap_vs_prev_close_pct": na(_pct_diff(_num(last.Open), prev)),
    }
    lastd = daily.index[-1]
    win = daily[daily.index > lastd - pd.Timedelta(days=365)]
    if len(win) and win.index[0] <= lastd - pd.Timedelta(days=350):
        h52, l52 = _num(win.High.max()), _num(win.Low.min())
        snap["high_52w"], snap["low_52w"] = na(h52), na(l52)
        snap["pct_from_52w_high"] = na(_pct_diff(px, h52))
        snap["pct_from_52w_low"] = na(_pct_diff(px, l52))
        snap["position_in_52w_range_pct"] = na(None if h52 is None or l52 is None or h52 == l52
                                               else (px - l52) / (h52 - l52) * 100)
        if px is not None and h52 is not None and l52 is not None and not (l52 - 1e-9 <= px <= h52 + 1e-9):
            snap["position_in_52w_range_pct"] = NA
    else:
        snap["high_52w"] = snap["low_52w"] = NA
        snap["52w_note"] = "Less than ~1 year of data"
    ath = _num(daily.High.max())
    snap["highest_in_data_window"] = {"value": na(ath), "date": _date(daily.High.idxmax()),
                                      "pct_from_it": na(_pct_diff(px, ath)), "window_years": round(
                                          (daily.index[-1] - daily.index[0]).days / 365.25, 1)}
    return snap


def build_performance(close: pd.Series) -> dict:
    perf: dict[str, Any] = {"1d": na(_pct_diff(_num(close.iloc[-1]), _num(close.iloc[-2])) if len(close) > 1 else None)}
    for k, off in HORIZONS.items():
        perf[k] = na(_ret_from(close, off))
    prev_year = close[close.index.year < close.index[-1].year]
    perf["ytd"] = na(_pct_diff(_num(close.iloc[-1]), _num(prev_year.iloc[-1])) if len(prev_year) else None)
    return {"returns_pct": perf}


def build_relative_strength(stock: pd.Series, bench: pd.Series, bench_name: str) -> dict:
    out: dict[str, Any] = {"benchmark": bench_name, "periods": {}}
    for k in ("1m", "3m", "6m", "1y"):
        s, b = _ret_from(stock, HORIZONS[k]), _ret_from(bench, HORIZONS[k])
        out["periods"][k] = {"stock_pct": na(s), "benchmark_pct": na(b),
                             "outperformance_pp": na(None if s is None or b is None else s - b)}
    both = pd.concat([stock, bench], axis=1, join="inner").dropna()
    if len(both) >= 60:
        ratio = both.iloc[:, 0] / both.iloc[:, 1]
        r_last, r_ma = _last(ratio), _last(sma(ratio, 50))
        out["rs_line"] = {"trend_vs_50bar_avg": NA if r_last is None or r_ma is None
                          else "improving (above 50-bar avg)" if r_last > r_ma else "deteriorating (below 50-bar avg)",
                          "change_63_bars_pct": na(_pct_diff(r_last, _num(ratio.iloc[-64])) if len(ratio) > 64 else None)}
    else:
        out["rs_line"] = NA
    return out


def _benchmark_for(symbol: str) -> tuple[str, str]:
    if symbol.endswith(".NS"):
        return "^NSEI", "NIFTY 50"
    if symbol.endswith(".BO"):
        return "^BSESN", "BSE SENSEX"
    return "^GSPC", "S&P 500"


# ==========================================================================
# The tool
# ==========================================================================
@tool
def get_technical_analysis(ticker: str, pretty: bool = False) -> str:
    """
    Full technical-analysis snapshot for a stock across daily, weekly and monthly
    timeframes: trend (moving averages, ADX, Supertrend), momentum (RSI, MACD,
    Stochastic, CCI, Williams %R, ROC), volatility (ATR, Bollinger, historical
    vol), volume (OBV, MFI, CMF, VWAP), support/resistance (pivots, swing levels,
    Fibonacci), candlestick patterns, returns, relative strength vs the index and
    a rule-based signal summary.

    Args:
        ticker: Symbol such as 'RELIANCE', 'TCS.NS' or 'AAPL'. A bare symbol is
            tried on NSE (.NS), then BSE (.BO), then as given.
        pretty: Indent the JSON for human reading (default compact, which uses
            far fewer tokens when the result is fed to an LLM).

    Returns:
        JSON string. Any value that is unavailable or fails validation is "NA".
        'data_quality' reports cleaning actions and warnings; 'errors' lists
        failed data sources.
    """
    symbol = _resolve_symbol(ticker)
    t = yf.Ticker(symbol)
    errors: dict[str, str] = {}
    warn: list[str] = []

    info = _safe(errors, "info", t.get_info) or {}
    raw = _safe(errors, "history", lambda: t.history(period="10y", interval="1d", auto_adjust=False))
    if raw is None or raw.empty:
        return json.dumps({"error": f"No price data for '{ticker}' (tried '{symbol}').", "errors": errors}, indent=2)

    tz = getattr(raw.index, "tz", None)
    today = (pd.Timestamp.now(tz=tz).tz_localize(None) if tz is not None else pd.Timestamp.now()).normalize()

    try:
        daily, issues = clean_ohlcv(raw)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"Price data failed validation: {e}", "errors": errors}, indent=2)
    if len(daily) < 2:
        return json.dumps({"error": "Fewer than 2 valid daily bars after cleaning.", "data_quality": issues}, indent=2)

    last_daily = daily.index[-1]
    age = int((today - last_daily).days)
    if age > 5:
        warn.append(f"daily: last bar is {age} days old, data may be stale or the stock suspended")
    if issues["jumps_over_40pct"]:
        warn.append(f"daily: >40% single-day moves on {issues['jumps_over_40pct']} (check for unadjusted splits/bad ticks)")
    for k in ("dropped_nan_ohlc", "dropped_non_positive_price", "repaired_ohlc_inconsistency"):
        if issues[k]:
            warn.append(f"daily: {k}={issues[k]}")

    has_volume = bool((daily.Volume > 0).mean() > 0.5)
    market_open = str(info.get("marketState", "")).upper() == "REGULAR"

    frames = {"daily": daily, "weekly": resample_ohlcv(daily, "W-FRI"), "monthly": resample_ohlcv(daily, "ME")}

    dq: dict[str, Any] = {"daily_cleaning": issues}
    timeframes: dict[str, Any] = {}
    for tf, df in frames.items():
        partial = bool(len(df)) and _is_partial(tf, df.index[-1], last_daily, today, market_open)
        dq[tf] = {"bars": int(len(df)), "first_bar": _date(df.index[0]), "last_bar": _date(df.index[-1]),
                  "last_bar_is_partial": partial}
        try:
            timeframes[tf] = analyze_timeframe(df, tf, has_volume, partial, warn)
        except Exception as e:  # noqa: BLE001
            errors[f"analysis_{tf}"] = f"{type(e).__name__}: {e}"
            timeframes[tf] = NA

    # Multi-timeframe alignment
    biases = {tf: (v["signals"]["summary"]["bias"] if isinstance(v, dict) and "signals" in v else NA)
              for tf, v in timeframes.items()}
    known = [b for b in biases.values() if b != NA]
    if len(known) == 3 and len(set(known)) == 1:
        align = f"aligned_{known[0]}"
    elif known:
        align = "mixed"
    else:
        align = NA
    mtf = {"bias_by_timeframe": biases, "alignment": align,
           "note": "Rule-based tally of indicator readings; not a prediction or investment advice."}

    # Relative strength vs benchmark (isolated failure)
    bench_sym, bench_name = _benchmark_for(symbol)
    rs: Any = NA
    braw = _safe(errors, "benchmark_history",
                 lambda: yf.Ticker(bench_sym).history(period="5y", interval="1d", auto_adjust=False))
    if braw is not None and not braw.empty:
        try:
            bclean, _ = clean_ohlcv(braw)
            rs = build_relative_strength(daily.Close, bclean.Close, f"{bench_name} ({bench_sym})")
        except Exception as e:  # noqa: BLE001
            errors["benchmark_processing"] = f"{type(e).__name__}: {e}"

    candles = [{"date": _date(i), **{k: na(r[k], 2 if k != "Volume" else 0) for k in OHLCV}}
               for i, r in daily.tail(10).iterrows()]

    result = {
        "meta": {
            "stock_name": info.get("longName") or info.get("shortName") or NA,
            "stock_symbol": symbol,
            "exchange": info.get("fullExchangeName") or info.get("exchange") or NA,
            "currency": info.get("currency") or NA,
            "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_daily_bar": _date(last_daily),
            "market_state": info.get("marketState") or NA,
            "data_source": "Yahoo Finance via yfinance",
            "price_basis": "split-adjusted, not dividend-adjusted (matches most charting platforms)",
        },
        "data_quality": {
            "status": "DEGRADED" if errors else "OK_WITH_WARNINGS" if warn else "OK",
            "has_usable_volume": has_volume, **dq, "warnings": warn,
        },
        "price_snapshot": build_snapshot(daily, timeframes["daily"].get("last_bar_is_partial", False)
                                         if isinstance(timeframes["daily"], dict) else False),
        "performance": build_performance(daily.Close),
        "relative_strength": rs,
        "recent_daily_candles": candles,
        "timeframes": timeframes,
        "multi_timeframe_summary": mtf,
        "errors": errors,
        "notes": [
            "Weekly bars end Friday; monthly bars end on the calendar month-end. Both are built from the same validated daily bars.",
            "If last_bar_is_partial is true, that bar's indicators are provisional; pivots then use the previous completed bar.",
            "RSI/ATR/ADX use Wilder smoothing; EMAs/MACD are SMA-seeded.",
            "Indicators needing more history than available (e.g. monthly SMA24 on a young stock) are reported as NA.",
        ],
    }
    if pretty:
        return json.dumps(_sanitize(result), indent=2)
    return json.dumps(_sanitize(result), separators=(",", ":"))


@tool
def get_stock_history(ticker: str, days: int = 30) -> str:
    """
    Fetches historical daily OHLCV prices for a given stock ticker and 
    calculates key technical indicators (20-day SMA, 14-day RSI).
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        days: The number of recent trading days of data to return to the analyst (default is 30).
        
    Returns:
        A string representation of the stock's historical data and indicators, or an error message.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Fetch extra buffer days to calculate indicators accurately.
        # We need at least 20 extra days for the SMA, and 14 for the RSI.
        buffer_days = 30 
        total_days_to_fetch = days + buffer_days
        
        # yfinance period string formatting (e.g., '60d')
        hist = stock.history(period=f"{total_days_to_fetch}d")
        
        if hist.empty:
            return f"Error: No data found for ticker {ticker}. It might be invalid or delisted."
        
        # 2. Keep Open, High, Low, Close, Volume (OHLCV)
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # 3. Calculate 20-day Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # 4. Calculate 14-day Relative Strength Index (RSI)
        # Calculate daily price changes
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate the exponential moving average of gains and losses (Wilder's Smoothing)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 5. Clean up the dataframe
        # Drop the NaN values from our buffer period
        df = df.dropna()
        
        # Round prices to 2 decimals to save LLM tokens and improve readability
        df = df.round(2)
        
        # 6. Slice the dataframe to return ONLY the requested number of days
        final_data = df.tail(days)
        
        # Convert to string format for the LLM
        data_str = final_data.to_string()
        
        return f"Technical Data for {ticker} over the last {days} days (includes OHLC, 20-day SMA, and 14-day RSI):\n{data_str}"
        
    except Exception as e:
        return f"An error occurred while fetching data for {ticker}: {str(e)}"