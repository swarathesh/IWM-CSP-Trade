#!/usr/bin/env python3
"""
Sector Rotation Dashboard
Mirrors the TradingView Pine Script — ranks 28 sector ETFs by 3-month
performance vs SPY and flags relative-strength new highs.
Last triggered: 2026-05-06
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

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
    ("UFO",  "Space Theme"),
    ("MARS", "Space Theme"),
    ("WCLD", "Cloud"),
    ("BUG",  "Cybersecurity"),
    ("ITB",  "Homebuilders"),
]

BENCHMARK = "SPY"
MODEL_STATE_PATH = "sector_rotation_model_state.json"
MODEL_VERSION = "markov-v1"
ETF_TO_SECTOR = {ticker: sector for ticker, sector in ETFS}


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

ML_LOOKBACK = LOOKBACKS["3M"]


def _build_leader_history(
    prices: pd.DataFrame,
    benchmark_col: str,
    perf_lookback: int,
) -> pd.DataFrame:
    """Build daily leader series based on relative performance vs benchmark."""
    etf_tickers = [t for t, _ in ETFS]
    needed = [benchmark_col] + etf_tickers
    available = [c for c in needed if c in prices.columns]
    if benchmark_col not in available:
        return pd.DataFrame(columns=["leader_etf", "leader_sector"])

    frame = prices[available].dropna(how="all")
    if len(frame) < perf_lookback + 2:
        return pd.DataFrame(columns=["leader_etf", "leader_sector"])

    perf = frame.pct_change(perf_lookback) * 100
    spy_perf = perf[benchmark_col]
    rel = pd.DataFrame(index=perf.index)
    for ticker in etf_tickers:
        if ticker in perf.columns:
            rel[ticker] = perf[ticker] - spy_perf

    rel = rel.dropna(how="all")
    if rel.empty:
        return pd.DataFrame(columns=["leader_etf", "leader_sector"])

    leaders = rel.idxmax(axis=1).dropna()
    if leaders.empty:
        return pd.DataFrame(columns=["leader_etf", "leader_sector"])

    history = pd.DataFrame({
        "leader_etf": leaders,
        "leader_sector": leaders.map(lambda t: ETF_TO_SECTOR.get(t, t)),
    })
    return history


def _load_ml_state(path: str) -> dict:
    """Load persisted transition model state."""
    try:
        with open(path) as f:
            state = json.load(f)
    except FileNotFoundError:
        return {"version": MODEL_VERSION, "transition_counts": {}, "last_observed_date": None}
    except Exception:
        return {"version": MODEL_VERSION, "transition_counts": {}, "last_observed_date": None}

    if not isinstance(state, dict):
        return {"version": MODEL_VERSION, "transition_counts": {}, "last_observed_date": None}
    state.setdefault("version", MODEL_VERSION)
    state.setdefault("transition_counts", {})
    state.setdefault("last_observed_date", None)
    return state


def _save_ml_state(path: str, state: dict) -> None:
    """Persist transition model state."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _estimate_next_sector(leader_history: pd.DataFrame, state_path: str = MODEL_STATE_PATH) -> dict:
    """
    Learn transition probabilities between top sectors and predict next leader.
    Uses an online Markov transition model that is updated each script run.
    """
    if leader_history.empty or len(leader_history) < 2:
        return {
            "model": MODEL_VERSION,
            "status": "insufficient_data",
            "message": "Need more historical data for prediction.",
        }

    state = _load_ml_state(state_path)
    counts = state.get("transition_counts", {})
    last_observed_date = state.get("last_observed_date")
    new_samples = 0

    leaders = leader_history["leader_etf"].tolist()
    dates = [idx.strftime("%Y-%m-%d") for idx in leader_history.index]

    for i in range(1, len(leaders)):
        current_date = dates[i]
        if last_observed_date and current_date <= last_observed_date:
            continue
        src = leaders[i - 1]
        dst = leaders[i]
        if src not in counts:
            counts[src] = {}
        counts[src][dst] = counts[src].get(dst, 0) + 1
        new_samples += 1

    state["transition_counts"] = counts
    state["last_observed_date"] = dates[-1]
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _save_ml_state(state_path, state)

    current_leader = leaders[-1]
    current_sector = ETF_TO_SECTOR.get(current_leader, current_leader)
    transition_row = counts.get(current_leader, {})

    if not transition_row:
        global_counts = {}
        for row in counts.values():
            for dst, value in row.items():
                global_counts[dst] = global_counts.get(dst, 0) + value
        transition_row = global_counts

    total = sum(transition_row.values())
    if total <= 0:
        return {
            "model": MODEL_VERSION,
            "status": "insufficient_data",
            "message": "Model has no learned transitions yet.",
            "current_leader": {"etf": current_leader, "sector": current_sector},
            "new_samples": new_samples,
            "total_samples": 0,
        }

    sorted_probs = sorted(transition_row.items(), key=lambda x: x[1], reverse=True)
    top_predictions = [
        {
            "etf": ticker,
            "sector": ETF_TO_SECTOR.get(ticker, ticker),
            "probability": round((count / total) * 100, 2),
            "transition_count": int(count),
        }
        for ticker, count in sorted_probs[:3]
    ]

    total_samples = sum(
        count for row in counts.values() for count in row.values()
    )

    return {
        "model": MODEL_VERSION,
        "status": "ok",
        "lookback_days": ML_LOOKBACK,
        "current_leader": {"etf": current_leader, "sector": current_sector},
        "predicted_next": top_predictions[0],
        "top_predictions": top_predictions,
        "new_samples": new_samples,
        "total_samples": int(total_samples),
        "state_file": state_path,
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
    leader_history = _build_leader_history(prices, BENCHMARK, perf_lookback=ML_LOOKBACK)
    ml_prediction = _estimate_next_sector(leader_history, MODEL_STATE_PATH)

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

    return df, spy_perfs, sort_label, ml_prediction


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


def _build_payload(df: pd.DataFrame, spy_perfs: dict, ml_prediction: dict) -> dict:
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
        "generated": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
        "lookbacks": list(LOOKBACKS.keys()),
        "benchmark": benchmark,
        "sectors": records,
        "ml_prediction": ml_prediction,
    }


def export_json(df: pd.DataFrame, spy_perfs: dict, ml_prediction: dict, path: str):
    """Export dashboard data as JSON."""
    payload = _build_payload(df, spy_perfs, ml_prediction)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Exported to {path}")


def export_js(df: pd.DataFrame, spy_perfs: dict, ml_prediction: dict, path: str):
    """Export as a JS file that index.html can load via <script src>."""
    payload = _build_payload(df, spy_perfs, ml_prediction)
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

    df, spy_perfs, sort_label, ml_prediction = build_dashboard(
        perf_lookback=args.lookback,
        rs_base_len=args.rs_base,
        rs_new_high_len=args.rs_window,
    )

    print_table(df, spy_perfs, sort_label)
    if ml_prediction.get("status") == "ok":
        next_pick = ml_prediction.get("predicted_next", {})
        current = ml_prediction.get("current_leader", {})
        print(
            "  ML Prediction:",
            f"{current.get('etf', 'N/A')} → {next_pick.get('etf', 'N/A')}",
            f"({next_pick.get('probability', 0):.2f}% confidence,",
            f"{ml_prediction.get('total_samples', 0)} learned transitions)",
        )
    else:
        print(f"  ML Prediction: {ml_prediction.get('message', 'Not available')}")

    if args.json:
        export_json(df, spy_perfs, ml_prediction, args.json)

    if not args.no_js:
        export_js(df, spy_perfs, ml_prediction, "sector_rotation_data.js")


if __name__ == "__main__":
    main()
