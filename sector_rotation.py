#!/usr/bin/env python3
"""
Sector Rotation Dashboard
Mirrors the TradingView Pine Script — ranks 26 sector ETFs by 3-month
performance vs SPY and flags relative-strength new highs.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

# ── ETF universe ──────────────────────────────────────────────────────────
ETFS = [
    ("SMH",  "Semis"),
    ("XSD",  "Semis EW"),
    ("SOXX", "Semis Broad"),
    ("XLK",  "Technology"),
    ("IGV",  "Software"),
    ("XBI",  "Biotech"),
    ("XLE",  "Energy"),
    ("XOP",  "Oil & Gas E&P"),
    ("XLI",  "Industrials"),
    ("XAR",  "Aero & Defense"),
    ("XLF",  "Financials"),
    ("XLV",  "Healthcare"),
    ("XLY",  "Cons Disc"),
    ("XLP",  "Cons Staples"),
    ("XLU",  "Utilities"),
    ("XLC",  "Comm Svcs"),
    ("XLB",  "Materials"),
    ("XLRE", "Real Estate"),
    ("URA",  "Uranium/Nuclear"),
    ("GDX",  "Gold Miners"),
    ("IWM",  "Small Cap"),
    ("TAN",  "Solar"),
    ("AIQ",  "AI Theme"),
    ("WCLD", "Cloud"),
    ("BUG",  "Cybersecurity"),
    ("ITB",  "Homebuilders"),
]

BENCHMARK = "SPY"


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted close prices for all tickers in one batch."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices


def compute_performance(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """Percentage change over `lookback` trading days for each column."""
    if len(prices) < lookback + 1:
        return pd.Series(dtype=float)
    current = prices.iloc[-1]
    past = prices.iloc[-(lookback + 1)]
    perf = (current - past) / past * 100
    return perf


def compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI (14-day) using Wilder's smoothing (EMA with alpha=1/period)."""
    results = {}
    for col in prices.columns:
        delta = prices[col].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        # Wilder's smoothing: EMA with alpha = 1/period
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        results[col] = round(val, 1) if not pd.isna(val) else float("nan")
    return pd.Series(results)


def compute_atr_pct(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as % of price (using close-to-close as proxy for true range)."""
    results = {}
    for col in prices.columns:
        series = prices[col].dropna()
        if len(series) < period + 1:
            results[col] = float("nan")
            continue
        tr = series.diff().abs()
        atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
        current = series.iloc[-1]
        results[col] = round((atr / current) * 100, 2) if current > 0 else float("nan")
    return pd.Series(results)


def compute_iv_rank(tickers):
    """
    IV Rank: where current IV sits in its 1-year range.
    Uses yfinance option chain implied volatility as a proxy.
    Falls back to historical volatility rank if options data unavailable.
    """
    results = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1y")
            if hist.empty or len(hist) < 30:
                results[ticker] = None
                continue
            # Historical volatility: 20-day rolling std of log returns, annualized
            log_ret = (hist["Close"] / hist["Close"].shift(1)).apply(
                lambda x: x if pd.isna(x) else __import__("math").log(x)
            )
            hv = log_ret.rolling(20).std() * (252 ** 0.5) * 100
            hv = hv.dropna()
            if len(hv) < 30:
                results[ticker] = None
                continue
            current = hv.iloc[-1]
            hi = hv.max()
            lo = hv.min()
            if hi == lo:
                results[ticker] = 50.0
            else:
                results[ticker] = round(((current - lo) / (hi - lo)) * 100, 1)
        except Exception:
            results[ticker] = None
    return results


def compute_rs_new_high(
    prices: pd.DataFrame,
    benchmark_col: str,
    rs_base_len: int,
    rs_new_high_len: int,
) -> dict[str, bool]:
    """
    For each ETF column, compute the RS line (ETF / SPY), then check
    whether today's RS value is within 0.5% of the 52-bar high.
    This ensures only genuinely fresh breakouts are flagged.
    """
    spy = prices[benchmark_col]
    results = {}
    for col in prices.columns:
        if col == benchmark_col:
            continue
        rs_line = prices[col] / spy
        rs_line = rs_line.dropna()
        if len(rs_line) < rs_base_len:
            results[col] = False
            continue
        rs_base_high = rs_line.iloc[-rs_base_len:].max()
        rs_today = rs_line.iloc[-1]
        # Flag if today's RS is within 0.5% of the 52-bar high
        results[col] = rs_today >= rs_base_high * 0.995
    return results


LOOKBACKS = {
    "1D": 1,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


def build_dashboard(
    perf_lookback: int = 63,
    rs_base_len: int = 52,
    rs_new_high_len: int = 3,
) -> tuple:
    """Fetch data, compute metrics for all lookback periods."""
    tickers = [BENCHMARK] + [t for t, _ in ETFS]
    prices = fetch_prices(tickers)

    # Compute perf for all lookback windows
    all_perfs = {}
    for label, days in LOOKBACKS.items():
        all_perfs[label] = compute_performance(prices, days)

    rs_flags = compute_rs_new_high(prices, BENCHMARK, rs_base_len, rs_new_high_len)
    rsi_values = compute_rsi(prices)
    atr_values = compute_atr_pct(prices)

    print("  Computing IV Rank (this takes a moment)...")
    iv_ranks = compute_iv_rank([t for t, _ in ETFS])

    rows = []
    for ticker, sector in ETFS:
        row = {
            "ETF": ticker,
            "Sector": sector,
            "RS New High": rs_flags.get(ticker, False),
            "RSI": rsi_values.get(ticker, float("nan")),
            "ATR %": atr_values.get(ticker, float("nan")),
            "IV Rank": iv_ranks.get(ticker),
        }
        for label in LOOKBACKS:
            row[f"{label} Perf %"] = round(all_perfs[label].get(ticker, float("nan")), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by the requested primary lookback
    sort_label = next((l for l, d in LOOKBACKS.items() if d == perf_lookback), "3M")
    df = df.sort_values(f"{sort_label} Perf %", ascending=False, na_position="last").reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    spy_perfs = {}
    for label in LOOKBACKS:
        spy_perfs[label] = round(all_perfs[label].get(BENCHMARK, float("nan")), 2)

    return df, spy_perfs, sort_label


# ── Terminal display with colour ──────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
WHITE = "\033[97m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
BG_GREEN = "\033[42m"
BG_BLUE = "\033[44m"


def colour_perf(val, spy_perf, rank):
    """Colour a performance value based on rank / vs SPY / positive-negative."""
    if pd.isna(val):
        return f"{DIM}  N/A  {RESET}"
    s = f"{val:+.2f}%"
    if rank <= 3:
        return f"{GREEN}{BOLD}{s:>8}{RESET}"
    if val > spy_perf:
        return f"{GREEN}{s:>8}{RESET}"
    if val > 0:
        return f"{YELLOW}{s:>8}{RESET}"
    return f"{RED}{s:>8}{RESET}"


def _colour_rsi(val):
    if pd.isna(val):
        return f"{DIM}{'N/A':>5}{RESET}"
    if val <= 30:
        return f"{GREEN}{BOLD}{val:>5.1f}{RESET}"
    if val >= 70:
        return f"{RED}{BOLD}{val:>5.1f}{RESET}"
    return f"{WHITE}{val:>5.1f}{RESET}"


def _colour_iv_rank(val):
    if val is None or pd.isna(val):
        return f"{DIM}{'N/A':>5}{RESET}"
    if val >= 60:
        return f"{GREEN}{BOLD}{val:>5.1f}{RESET}"
    if val <= 20:
        return f"{RED}{val:>5.1f}{RESET}"
    return f"{WHITE}{val:>5.1f}{RESET}"


def print_table(df: pd.DataFrame, spy_perfs: dict, sort_label: str):
    """Pretty-print the dashboard to the terminal."""
    perf_col = f"{sort_label} Perf %"
    spy_perf = spy_perfs[sort_label]

    header = (
        f"{BOLD}{WHITE}{'Rank':>4}  {'ETF':<6} {'Sector':<16} {sort_label + ' Perf':>8}"
        f"  {'RSI':>5}  {'IVR':>5}  {'ATR%':>5}  {'RS High':>9}{RESET}"
    )
    sep = f"{DIM}{'─' * 76}{RESET}"

    print()
    print(f"  {BOLD}{WHITE}Sector Rotation Dashboard{RESET}")
    print(f"  {DIM}{datetime.now().strftime('%Y-%m-%d %H:%M')}{RESET}")
    print()
    print(f"  {header}")
    print(f"  {sep}")

    # SPY benchmark row
    spy_str = f"{spy_perf:+.2f}%" if not pd.isna(spy_perf) else "N/A"
    print(
        f"  {CYAN}{BOLD}{'REF':>4}  {'SPY':<6} {'Benchmark':<16} {spy_str:>8}"
        f"  {'---':>5}  {'---':>5}  {'---':>5}  {'---':>9}{RESET}"
    )
    print(f"  {sep}")

    for rank, row in df.iterrows():
        perf_str = colour_perf(row[perf_col], spy_perf, rank)
        rsi_str = _colour_rsi(row.get("RSI"))
        ivr_str = _colour_iv_rank(row.get("IV Rank"))
        atr_val = row.get("ATR %")
        atr_str = f"{DIM}{'N/A':>5}{RESET}" if pd.isna(atr_val) else f"{WHITE}{atr_val:>5.2f}{RESET}"
        rs_str = f"{GREEN}{BOLD}{'NEW HIGH':>9}{RESET}" if row["RS New High"] else f"{DIM}{'---':>9}{RESET}"
        print(
            f"  {WHITE}{rank:>4}{RESET}  {WHITE}{row['ETF']:<6}{RESET} {DIM}{row['Sector']:<16}{RESET}"
            f" {perf_str}  {rsi_str}  {ivr_str}  {atr_str}  {rs_str}"
        )

    print()


def _build_payload(df: pd.DataFrame, spy_perfs: dict) -> dict:
    records = []
    for rank, row in df.iterrows():
        rsi_val = row.get("RSI")
        atr_val = row.get("ATR %")
        ivr_val = row.get("IV Rank")
        entry = {
            "rank": rank,
            "etf": row["ETF"],
            "sector": row["Sector"],
            "rs_new_high": bool(row["RS New High"]),
            "rsi": None if pd.isna(rsi_val) else rsi_val,
            "atr_pct": None if pd.isna(atr_val) else atr_val,
            "iv_rank": None if (ivr_val is None or pd.isna(ivr_val)) else ivr_val,
        }
        for label in LOOKBACKS:
            val = row.get(f"{label} Perf %")
            entry[f"perf_{label.lower()}"] = None if pd.isna(val) else val
        records.append(entry)

    benchmark = {"etf": "SPY"}
    for label in LOOKBACKS:
        val = spy_perfs.get(label)
        benchmark[f"perf_{label.lower()}"] = None if pd.isna(val) else val

    return {
        "generated": datetime.now().isoformat(),
        "lookbacks": list(LOOKBACKS.keys()),
        "benchmark": benchmark,
        "sectors": records,
    }


def export_json(df: pd.DataFrame, spy_perfs: dict, path: str):
    """Export dashboard data as JSON."""
    payload = _build_payload(df, spy_perfs)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Exported to {path}")


def export_js(df: pd.DataFrame, spy_perfs: dict, path: str):
    """Export as a JS file that index.html can load via <script src>."""
    payload = _build_payload(df, spy_perfs)
    with open(path, "w") as f:
        f.write("// Auto-generated by sector_rotation.py — do not edit\n")
        f.write("const SR_DATA = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")
    print(f"  Exported to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sector Rotation Dashboard")
    parser.add_argument("--lookback", type=int, default=63, help="Performance lookback in trading days (default 63 ≈ 3 months)")
    parser.add_argument("--rs-base", type=int, default=52, help="RS line lookback in trading days (default 52)")
    parser.add_argument("--rs-window", type=int, default=3, help="RS new-high window in trading days (default 3)")
    parser.add_argument("--json", type=str, metavar="PATH", help="Export results as JSON to PATH")
    parser.add_argument("--no-js", action="store_true", help="Skip auto JS export for index.html")
    args = parser.parse_args()

    df, spy_perfs, sort_label = build_dashboard(
        perf_lookback=args.lookback,
        rs_base_len=args.rs_base,
        rs_new_high_len=args.rs_window,
    )

    print_table(df, spy_perfs, sort_label)

    if args.json:
        export_json(df, spy_perfs, args.json)

    if not args.no_js:
        export_js(df, spy_perfs, "sector_rotation_data.js")


if __name__ == "__main__":
    main()
