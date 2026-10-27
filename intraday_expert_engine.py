"""
Institutional Intraday Trading Expert Engine (Indian NSE & International Markets)
=================================================================================
Engineered for `Intraday-only` to replace the flawed daily-quota & retroactive-SL logic
in `smart-etf-tracker`.

Core Capabilities:
1. Multi-Market & Sectoral Shift Radar (`detect_global_and_indian_regime`):
   Tracks Nifty 50, India VIX, 7 NSE Sectoral Indices, and International Cues
   (S&P 500, Nasdaq 100, Gold, Brent Crude, USD/INR).
2. Top 5 Institutional Sniper Logics (`scan_live_sniper_opportunities`):
   - L1: Sectoral Shift + NR7 Volatility Compression ORB Sniper
   - L2: Structural Support S1 + 50DMA Institutional Spring & VWAP Reclaim Sniper
   - L3: Fundamental Catalyst / Delivery Volume Shock Day-2 VWAP Pullback Sniper
   - L4: Bear / High-VIX Defensive Relative Strength Fortress Sniper
   - L5: International & Overnight Global Macro Transmission Sniper
   *Sniper Frequency*: Zero forced daily trades. Fires only when multi-factor confluence
   and downside risk protection align (~1–2 high-conviction opportunities per week).
3. Downside Risk Protection & Bug-Free Chronological Trailing Stop (`evaluate_chronological_exits`):
   - Anchors initial stops below Structural Support (`S1` Pivot / `10D Swing Low` / `ORB Low` - ATR buffer).
   - Tracks `High_Water_Mark` and `Last_Evaluated_Bar_TS` so afternoon SL upgrades are NEVER
     retroactively applied to morning bars.
4. 3:10 PM Intraday-to-Swing Conversion Predictor (`predict_swing_conversion_success`):
   - Implements the 3-Year empirically validated 5-Gate Swing Conversion model (84.90% win rate).
"""

import os
import sys
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest_engine_3y import (
    UNIVERSE_META,
    BENCHMARKS_AND_SECTORS,
    normalize_index,
    compute_rsi,
    compute_atr,
    compute_structural_support_s1,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SNIPER_TRADES_CSV = os.path.join(DATA_DIR, "intraday_sniper_trades.csv")
SNIPER_TELEMETRY_CSV = os.path.join(DATA_DIR, "sniper_telemetry_trades.csv")

# Strict Capital & Daily Slot Constraints (User Rule: ₹20K max/ticker, Max 3 entries/day, CNC Only)
MAX_BUDGET_PER_TICKER_INR = 20000.0
MAX_DAILY_ENTRIES = 3
MAX_DAILY_TOTAL_BUDGET_INR = MAX_BUDGET_PER_TICKER_INR * MAX_DAILY_ENTRIES


TOP_5_LOGICS_METADATA = {
    "L1_SECTOR_ORB_SNIPER": {
        "name": "L1: Sectoral Shift + NR7 Compression ORB Sniper",
        "regime_fit": ["BULL_EXPANSION", "SIDEWAYS_ROTATION"],
        "shift_type": "Sectoral Rotation & Volatility Compression",
        "cadence": "~1 trade every 7–10 trading days",
        "stop_architecture": "Max(Open - 0.45*ATR, Structural S1 - 0.15*ATR) + 4-Stage Chronological Trail",
    },
    "L2_SUPPORT_SPRING_SNIPER": {
        "name": "L2: Structural S1 + 50DMA Institutional Spring Sniper",
        "regime_fit": ["SIDEWAYS_ROTATION", "BULL_EXPANSION"],
        "shift_type": "Technical Structural Support Spring & VWAP Reclaim",
        "cadence": "~1 trade every 5–8 trading days",
        "stop_architecture": "Morning Spring Low - 0.18*ATR (Below S1 Floor) + 4-Stage Chronological Trail",
    },
    "L3_CATALYST_VWAP_SNIPER": {
        "name": "L3: Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper",
        "regime_fit": ["BULL_EXPANSION", "SIDEWAYS_ROTATION", "BEAR_OR_CORRECTION"],
        "shift_type": "Fundamental Earnings / Institutional Delivery Shock",
        "cadence": "~1 trade every 8–12 trading days",
        "stop_architecture": "Day-1 Catalyst VWAP Floor - 0.15*ATR + 4-Stage Chronological Trail",
    },
    "L4_DEFENSIVE_RS_SNIPER": {
        "name": "L4: Bear/High-VIX Defensive Relative Strength Fortress Sniper",
        "regime_fit": ["BEAR_OR_CORRECTION", "HIGH_VOL_MACRO_SHOCK"],
        "shift_type": "Defensive Sectoral Sanctuary (Pharma/FMCG/Gold/IT RS Divergence)",
        "cadence": "~1 trade every 6–10 trading days during market corrections",
        "stop_architecture": "Tight S1/Open - 0.38*ATR Floor + Fast Profit-Lock Ratchet",
    },
    "L5_GLOBAL_MACRO_SNIPER": {
        "name": "L5: International & Overnight Global Macro Transmission Sniper",
        "regime_fit": ["BULL_EXPANSION", "SIDEWAYS_ROTATION", "HIGH_VOL_MACRO_SHOCK"],
        "shift_type": "International US Tech (Nasdaq/S&P500) & Gold Macro Breakout",
        "cadence": "~1 trade every 5–8 trading days",
        "stop_architecture": "Structural S1 + Non-Exhausted Opening Gap Floor + 4-Stage Trail",
    },
    "L6_5GATE_10AM_SNIPER": {
        "name": "L6: 5-Gate Pre-Qualified 10:00 AM Intraday Sniper (Trail SL + Target)",
        "regime_fit": ["BULL_EXPANSION", "SIDEWAYS_ROTATION", "HIGH_VOL_MACRO_SHOCK"],
        "shift_type": "5-Gate Structural-Support Pre-Qualified 10:00 AM ORB + Dual Intraday/Swing Payoff",
        "cadence": "~1 trade every 4–6 trading days",
        "stop_architecture": "9:15–10:00 AM ORB Low / Structural S1 - 0.15*ATR + 4-Stage Chronological Trail & Target",
    },
}


def compute_downside_protection_ladder(entry_price: float, initial_sl: float, atr: float, s1_floor: float) -> dict:
    """
    Returns the complete institutional Downside Risk Protection & 4-Stage Trailing Stop Ladder
    plus CNC Broker GTT (Good Till Triggered) OCO Trigger & Limit prices.
    """
    atr = max(atr, entry_price * 0.01)
    # Ensure initial_sl is strictly below entry_price (at least 0.35*ATR or 0.5% below entry)
    safe_sl = min(initial_sl, entry_price - 0.35 * atr, entry_price * 0.994)
    safe_sl = max(safe_sl, entry_price * 0.955)  # Cap max single-trade CNC stop distance at 4.5%
    risk_pct = max(0.25, (entry_price - safe_sl) / entry_price * 100.0)
    intraday_tgt = round(entry_price + 2.00 * atr, 2)
    swing_tgt = round(entry_price + 3.20 * atr, 2)
    return {
        "entry_price": round(entry_price, 2),
        "structural_s1_support": round(s1_floor, 2),
        "initial_stop_loss": round(safe_sl, 2),
        "gtt_sl_limit_price": round(safe_sl * 0.997, 2),
        "initial_downside_risk_pct": round(risk_pct, 2),
        "stage1_trigger_price": round(entry_price + 0.80 * atr, 2),
        "stage1_trail_sl": round(entry_price + 0.18 * atr, 2),
        "stage2_trigger_price": round(entry_price + 1.25 * atr, 2),
        "stage2_trail_sl": round(entry_price + 0.65 * atr, 2),
        "stage3_trigger_price": round(entry_price + 1.75 * atr, 2),
        "stage3_trail_sl": round(entry_price + 1.15 * atr, 2),
        "stage4_trigger_price": round(entry_price + 2.20 * atr, 2),
        "stage4_chandelier_offset": round(0.50 * atr, 2),
        "intraday_target": intraday_tgt,
        "gtt_target_limit_price": round(intraday_tgt * 0.999, 2),
        "swing_conversion_target": swing_tgt,
    }


def predict_swing_conversion_success(
    entry_price: float,
    current_price: float,
    day_high: float,
    day_low: float,
    s1_support: float,
    sma20: float,
    sma50: float,
    sma200: float,
    vol_ratio: float,
    market_regime: str,
    sl_breached: bool = False,
) -> dict:
    """
    Evaluates an intraday position against our 3-Year empirically validated
    Intraday-to-Swing Conversion Model (4,955 historical trades analyzed):
      - Loser Bag-Hold (SL breached): 35.38% win rate (-1.53% avg return) -> NEVER CONVERT
      - Unconditional Stalled Conversion: 60.70% win rate (+0.38% avg return)
      - 5-Gate Smart Swing Conversion: 84.90% win rate (+1.96% avg return)
    """
    day_range = max(day_high - day_low, 1e-6)
    close_pos = (current_price - day_low) / day_range

    gate_1_no_sl_breach = (not sl_breached) and (current_price > max(s1_support, sma20 * 0.995))
    gate_2_upper_range_close = close_pos >= 0.55
    gate_3_htf_trend = (sma20 >= sma50 * 0.995) and (current_price > sma200)
    gate_4_volume_support = vol_ratio >= 1.15
    gate_5_regime_safe = market_regime != "BEAR_OR_CORRECTION"

    gates_passed = sum([
        gate_1_no_sl_breach,
        gate_2_upper_range_close,
        gate_3_htf_trend,
        gate_4_volume_support,
        gate_5_regime_safe,
    ])

    if sl_breached or current_price < s1_support:
        prob = 35.4
        verdict = "🔴 DO NOT CONVERT (Hard Intraday Exit — Structural Support Broken)"
        expected_ret = -1.53
    elif gates_passed == 5:
        prob = 84.9
        verdict = "🟢 CONVERT TO SWING TRADE (All 5 Institutional Gates Passed — 84.9% 3Y Win Rate)"
        expected_ret = +1.96
    elif gates_passed == 4 and gate_1_no_sl_breach:
        prob = 71.5
        verdict = "🟡 CONDITIONAL SWING HOLD (Half Size Only — Trail Stop at S1/Day Low)"
        expected_ret = +0.95
    else:
        prob = 47.9
        verdict = "🟠 EXIT BY 3:15 PM IST (Insufficient Swing Follow-Through Edge)"
        expected_ret = +0.08

    return {
        "predicted_swing_win_rate_pct": prob,
        "expected_swing_return_pct": expected_ret,
        "gates_passed": f"{gates_passed}/5",
        "verdict": verdict,
        "gate_details": {
            "Gate 1: Above S1 Support & 20DMA": bool(gate_1_no_sl_breach),
            "Gate 2: Upper Range Close (>=55%)": bool(gate_2_upper_range_close),
            "Gate 3: HTF Trend (20DMA >= 50DMA > 200DMA)": bool(gate_3_htf_trend),
            "Gate 4: Institutional Volume (>=1.15x)": bool(gate_4_volume_support),
            "Gate 5: Non-Bear Market Regime": bool(gate_5_regime_safe),
        },
    }


def detect_global_and_indian_regime(raw_data: pd.DataFrame = None) -> dict:
    """
    Analyzes live/latest Nifty 50, India VIX, 7 NSE Sectoral Indices, and Global Cues
    (S&P 500, Nasdaq 100, Gold, Brent Crude, USD/INR) to classify the current regime
    and identify sectoral rotation leadership.
    """
    if raw_data is None:
        raw_data = yf.download(BENCHMARKS_AND_SECTORS, period="1y", interval="1d", group_by="ticker", progress=False, threads=True)

    metrics = {}
    for sym in BENCHMARKS_AND_SECTORS:
        try:
            if sym in raw_data.columns.get_level_values(0):
                df = normalize_index(raw_data[sym].dropna(subset=["Close"]))
                if len(df) >= 20:
                    c = float(df["Close"].iloc[-1])
                    ret1 = float(df["Close"].pct_change().iloc[-1] * 100.0)
                    ret5 = float(df["Close"].pct_change(5).iloc[-1] * 100.0)
                    sma20 = float(df["Close"].rolling(20, min_periods=10).mean().iloc[-1])
                    sma50 = float(df["Close"].rolling(50, min_periods=20).mean().iloc[-1])
                    sma200 = float(df["Close"].rolling(200, min_periods=50).mean().iloc[-1])
                    metrics[sym] = {
                        "close": round(c, 2),
                        "ret_1d": round(ret1, 2),
                        "ret_5d": round(ret5, 2),
                        "sma20": round(sma20, 2),
                        "sma50": round(sma50, 2),
                        "sma200": round(sma200, 2),
                    }
        except Exception:
            pass

    n_info = metrics.get("^NSEI", {"close": 24000, "ret_1d": 0.0, "ret_5d": 0.0, "sma50": 24000, "sma200": 23500})
    vix_val = metrics.get("^INDIAVIX", {}).get("close", 14.0)

    nc = n_info["close"]
    n50 = n_info["sma50"]
    n200 = n_info["sma200"]
    n_ret1 = n_info["ret_1d"]
    n_ret5 = n_info["ret_5d"]

    if vix_val >= 16.5 or abs(n_ret1) >= 1.25:
        regime = "HIGH_VOL_MACRO_SHOCK"
        regime_badge = "🟠 High-Volatility / Macro Shock"
    elif nc < n200 or (nc < n50 and n_ret5 < -1.0):
        regime = "BEAR_OR_CORRECTION"
        regime_badge = "🔴 Bear Market / Correction"
    elif nc > n50 and n50 >= n200 and n_ret5 >= 0.25:
        regime = "BULL_EXPANSION"
        regime_badge = "🟢 Bull Market Expansion"
    else:
        regime = "SIDEWAYS_ROTATION"
        regime_badge = "🟡 Sideways / Sectoral Rotation"

    # Rank NSE Sectors by 5D Relative Strength vs Nifty
    sector_names = {
        "^CNXPHARMA": "Nifty Pharma",
        "^CNXIT": "Nifty IT",
        "^NSEBANK": "Nifty Bank",
        "^CNXAUTO": "Nifty Auto",
        "^CNXMETAL": "Nifty Metal",
        "^CNXENERGY": "Nifty Energy",
        "^CNXFMCG": "Nifty FMCG",
    }
    sector_rankings = []
    for s_sym, s_label in sector_names.items():
        s_m = metrics.get(s_sym)
        if s_m:
            rs_5d = round(s_m["ret_5d"] - n_ret5, 2)
            sector_rankings.append({
                "symbol": s_sym,
                "sector": s_label,
                "ret_1d_pct": s_m["ret_1d"],
                "ret_5d_pct": s_m["ret_5d"],
                "rs_vs_nifty_5d_pct": rs_5d,
                "above_20dma": bool(s_m["close"] >= s_m["sma20"]),
            })
    sector_rankings.sort(key=lambda x: x["rs_vs_nifty_5d_pct"], reverse=True)

    # Determine active Sniper Logics for this regime
    active_logics = [
        lid for lid, lmeta in TOP_5_LOGICS_METADATA.items()
        if regime in lmeta["regime_fit"]
    ]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "regime_code": regime,
        "regime_badge": regime_badge,
        "nifty_close": nc,
        "nifty_ret_1d_pct": n_ret1,
        "nifty_ret_5d_pct": n_ret5,
        "india_vix": vix_val,
        "global_cues": {
            "S&P 500 (^GSPC)": metrics.get("^GSPC", {}),
            "Nasdaq 100 (^IXIC)": metrics.get("^IXIC", {}),
            "Gold Futures (GC=F)": metrics.get("GC=F", {}),
            "Brent Crude (BZ=F)": metrics.get("BZ=F", {}),
            "USD/INR (INR=X)": metrics.get("INR=X", {}),
        },
        "sector_leadership": sector_rankings,
        "authorized_sniper_logics": active_logics,
    }


def scan_live_sniper_opportunities() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Scans the entire 54-ticker Indian NSE & International universe using live market data.
    Returns:
      - `regime_report`: Full Indian + International Regime & Sectoral Radar dict
      - `sniper_triggers_df`: High-conviction Sniper setups firing right now (may be empty if no A+ setup!)
      - `watchlist_df`: Complete universe radar ranked by Readiness / Distance to Structural S1 Support & Swing Conversion Score
    """
    all_tickers = sorted(list(set(list(UNIVERSE_META.keys()) + BENCHMARKS_AND_SECTORS)))
    raw = yf.download(all_tickers, period="1y", interval="1d", group_by="ticker", progress=False, threads=True)

    regime_report = detect_global_and_indian_regime(raw)
    mkt_regime = regime_report["regime_code"]
    n_ret1 = regime_report["nifty_ret_1d_pct"]
    n_ret5 = regime_report["nifty_ret_5d_pct"]

    sec_lookup = {item["symbol"]: item for item in regime_report["sector_leadership"]}
    sp_ret = regime_report["global_cues"].get("S&P 500 (^GSPC)", {}).get("ret_1d", 0.0)
    ndx_ret = regime_report["global_cues"].get("Nasdaq 100 (^IXIC)", {}).get("ret_1d", 0.0)
    gold_ret = regime_report["global_cues"].get("Gold Futures (GC=F)", {}).get("ret_1d", 0.0)

    sniper_rows = []
    watchlist_rows = []

    for sym, meta in UNIVERSE_META.items():
        try:
            if sym not in raw.columns.get_level_values(0):
                continue
            df = normalize_index(raw[sym].dropna(subset=["Open", "High", "Low", "Close"]))
            if len(df) < 100:
                continue
        except Exception:
            continue

        df["Ret_1D"] = df["Close"].pct_change() * 100.0
        df["Ret_5D"] = df["Close"].pct_change(5) * 100.0
        df["Range"] = df["High"] - df["Low"]
        df["Min_Range_4D"] = df["Range"].rolling(4).min()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200, min_periods=50).mean()
        df["RSI14"] = compute_rsi(df["Close"], 14)
        df["ATR14"] = compute_atr(df, 14)
        df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
        df["S1_Support"] = compute_structural_support_s1(df, 20)

        row = df.iloc[-1]
        prev = df.iloc[-2]

        o_curr = float(row["Open"])
        h_curr = float(row["High"])
        l_curr = float(row["Low"])
        c_curr = float(row["Close"])
        v_curr = float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0

        c_prev = float(prev["Close"])
        h_prev = float(prev["High"])
        l_prev = float(prev["Low"])
        r_prev = float(prev["Range"])
        atr = float(row["ATR14"]) if pd.notna(row["ATR14"]) and row["ATR14"] > 0 else c_curr * 0.018
        rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else 50.0
        rsi_prev = float(prev["RSI14"]) if pd.notna(prev["RSI14"]) else 50.0
        sma20 = float(row["SMA20"]) if pd.notna(row["SMA20"]) else c_curr
        sma50 = float(row["SMA50"]) if pd.notna(row["SMA50"]) else c_curr
        sma200 = float(row["SMA200"]) if pd.notna(row["SMA200"]) else c_curr
        s1_floor = float(prev["S1_Support"]) if pd.notna(prev["S1_Support"]) else c_prev * 0.96
        vol_sma20 = float(prev["Vol_SMA20"]) if pd.notna(prev["Vol_SMA20"]) and prev["Vol_SMA20"] > 0 else 1.0
        vol_ratio = v_curr / vol_sma20
        prev_vol_ratio = (float(prev["Volume"]) / vol_sma20) if pd.notna(prev["Volume"]) else 1.0

        gap_pct = (o_curr - c_prev) / c_prev * 100.0
        day_ret_pct = (c_curr - c_prev) / c_prev * 100.0
        stock_ret_5d = float(row["Ret_5D"]) if pd.notna(row["Ret_5D"]) else 0.0
        stock_rs_5d = stock_ret_5d - n_ret5

        sec_info = sec_lookup.get(meta["sector"], {"ret_1d_pct": n_ret1, "rs_vs_nifty_5d_pct": 0.0})
        sec_ret_1d = sec_info["ret_1d_pct"]
        sector_rs_5d = sec_info["rs_vs_nifty_5d_pct"]

        # Evaluate all 5 Sniper Logics
        matched_logic_id = None
        matched_logic_name = None
        conviction = 0.0
        entry_trigger = round(o_curr + 0.26 * atr, 2)
        initial_sl = round(max(o_curr - 0.42 * atr, s1_floor - 0.15 * atr, entry_trigger * 0.990), 2)

        # L1: Sectoral Shift + NR7 Compression ORB
        is_nr_compressed = (r_prev <= float(prev["Min_Range_4D"]) * 1.15) or (r_prev < 0.95 * atr)
        near_sma20_or_s1 = ((c_prev - sma20) / sma20 * 100.0 <= 3.5) or ((c_prev - s1_floor) / c_prev * 100.0 <= 4.5)
        if (
            mkt_regime in ("BULL_EXPANSION", "SIDEWAYS_ROTATION")
            and c_prev > sma20 >= sma50 * 0.995
            and sma50 > sma200
            and near_sma20_or_s1
            and is_nr_compressed
            and sector_rs_5d >= 0.45
            and sec_ret_1d >= 0.35
            and -0.1 <= gap_pct <= 1.15
            and l_curr >= o_curr - 0.35 * atr
            and vol_ratio >= 1.35
            and 48.0 <= rsi_prev <= 67.0
            and c_curr >= o_curr + 0.25 * atr
        ):
            matched_logic_id = "L1_SECTOR_ORB_SNIPER"
            matched_logic_name = TOP_5_LOGICS_METADATA[matched_logic_id]["name"]
            entry_trigger = round(o_curr + 0.28 * atr, 2)
            initial_sl = round(max(o_curr - 0.45 * atr, s1_floor - 0.15 * atr, entry_trigger * 0.990), 2)
            conviction = min(99.0, 84.0 + sector_rs_5d * 2.5 + (vol_ratio - 1.3) * 5.0)

        # L2: Structural S1 + 50DMA Institutional Spring
        tested_support_zone = (l_curr <= max(s1_floor * 1.015, sma20 * 1.005)) and (l_curr >= min(s1_floor, sma50) - 0.40 * atr)
        if (
            matched_logic_id is None
            and mkt_regime in ("SIDEWAYS_ROTATION", "BULL_EXPANSION")
            and c_prev > sma200
            and sma50 >= sma200 * 0.99
            and 36.0 <= rsi_prev <= 51.0
            and tested_support_zone
            and (o_curr - l_curr) <= 0.45 * atr
            and c_curr >= o_curr + 0.30 * atr
            and vol_ratio >= 1.25
            and sec_ret_1d >= 0.15
        ):
            matched_logic_id = "L2_SUPPORT_SPRING_SNIPER"
            matched_logic_name = TOP_5_LOGICS_METADATA[matched_logic_id]["name"]
            entry_trigger = round(o_curr + 0.25 * atr, 2)
            initial_sl = round(l_curr - 0.18 * atr, 2)
            conviction = min(99.0, 85.0 + (52.0 - rsi_prev) * 0.5 + (vol_ratio - 1.2) * 5.0)

        # L3: Fundamental Catalyst / Volume Shock Day-2 VWAP
        prev_day_gain = float(prev["Ret_1D"]) if pd.notna(prev["Ret_1D"]) else 0.0
        prev_close_pos = (c_prev - l_prev) / max(h_prev - l_prev, 1e-6)
        had_catalyst_day1 = (prev_vol_ratio >= 1.75) and (prev_day_gain >= 1.85) and (prev_close_pos >= 0.68)
        prev_vwap_proxy = (h_prev + l_prev + c_prev) / 3.0
        if (
            matched_logic_id is None
            and had_catalyst_day1
            and c_prev > sma50 >= sma200 * 0.99
            and -0.35 <= gap_pct <= 1.25
            and l_curr >= max(prev_vwap_proxy, o_curr - 0.35 * atr)
            and c_curr >= o_curr + 0.26 * atr
            and vol_ratio >= 1.25
            and sec_ret_1d >= 0.05
        ):
            matched_logic_id = "L3_CATALYST_VWAP_SNIPER"
            matched_logic_name = TOP_5_LOGICS_METADATA[matched_logic_id]["name"]
            entry_trigger = round(o_curr + 0.26 * atr, 2)
            initial_sl = round(max(prev_vwap_proxy - 0.15 * atr, o_curr - 0.42 * atr, entry_trigger * 0.990), 2)
            conviction = min(99.0, 86.0 + prev_vol_ratio * 2.5)

        # L4: Bear / High-VIX Defensive Relative Strength Fortress
        is_defensive_or_rs = meta["group"] in ("PHARMA_DEFENSIVE", "FMCG_DEFENSIVE", "INTL_GOLD_MACRO", "IT_GLOBAL", "CAP_GOODS")
        if (
            matched_logic_id is None
            and mkt_regime in ("BEAR_OR_CORRECTION", "HIGH_VOL_MACRO_SHOCK")
            and is_defensive_or_rs
            and stock_rs_5d >= 1.10
            and c_prev > sma20 >= sma50 * 0.99
            and -0.15 <= gap_pct <= 1.05
            and l_curr >= o_curr - 0.32 * atr
            and c_curr >= o_curr + 0.26 * atr
            and vol_ratio >= 1.25
            and sec_ret_1d >= 0.15
        ):
            matched_logic_id = "L4_DEFENSIVE_RS_SNIPER"
            matched_logic_name = TOP_5_LOGICS_METADATA[matched_logic_id]["name"]
            entry_trigger = round(o_curr + 0.26 * atr, 2)
            initial_sl = round(max(o_curr - 0.38 * atr, s1_floor - 0.15 * atr, entry_trigger * 0.991), 2)
            conviction = min(99.0, 84.0 + stock_rs_5d * 2.5)

        # L5: International & Overnight Global Macro Transmission
        is_global_linked = meta["market"] == "INTERNATIONAL" or meta["group"] in ("IT_GLOBAL", "METALS_COMMODITY")
        global_tailwind = (
            (meta["group"] in ("INTL_US_TECH", "IT_GLOBAL") and ndx_ret >= 0.85)
            or (meta["group"] == "INTL_US_BROAD" and sp_ret >= 0.70)
            or (meta["group"] in ("INTL_GOLD_MACRO", "METALS_COMMODITY") and gold_ret >= 0.75)
        )
        if (
            matched_logic_id is None
            and is_global_linked
            and global_tailwind
            and c_prev > sma20
            and c_prev >= s1_floor
            and -0.15 <= gap_pct <= 1.15
            and l_curr >= o_curr - 0.30 * atr
            and c_curr >= o_curr + 0.24 * atr
            and vol_ratio >= 1.25
        ):
            matched_logic_id = "L5_GLOBAL_MACRO_SNIPER"
            matched_logic_name = TOP_5_LOGICS_METADATA[matched_logic_id]["name"]
            entry_trigger = round(o_curr + 0.24 * atr, 2)
            initial_sl = round(max(o_curr - 0.38 * atr, s1_floor - 0.15 * atr, entry_trigger * 0.991), 2)
            conviction = min(99.0, 85.0 + max(ndx_ret, sp_ret, gold_ret) * 4.5)

        # L6: 5-Gate Pre-Qualified 10:00 AM Intraday Sniper (Triggered ONLY when 5-Gate Swing Criteria are met at 10 AM)
        meets_5gate_10am = (
            c_prev > max(s1_floor, sma20)
            and l_curr >= max(s1_floor, sma20 * 0.998)
            and prev_close_pos >= 0.58
            and (sma20 >= sma50 * 0.995 and sma50 > sma200)
            and (prev_vol_ratio >= 1.15 and vol_ratio >= 1.25)
            and mkt_regime != "BEAR_OR_CORRECTION"
            and -0.10 <= gap_pct <= 1.10
            and (o_curr - l_curr) <= 0.32 * atr
            and c_curr >= o_curr + 0.26 * atr
            and sec_ret_1d >= 0.25
            and sector_rs_5d >= 0.20
            and 50.0 <= rsi_prev <= 66.0
        )
        if matched_logic_id is None and meets_5gate_10am:
            matched_logic_id = "L6_5GATE_10AM_SNIPER"
            matched_logic_name = TOP_5_LOGICS_METADATA[matched_logic_id]["name"]
            entry_trigger = round(o_curr + 0.26 * atr, 2)
            initial_sl = round(l_curr - 0.15 * atr, 2)
            conviction = min(99.0, 88.0 + sector_rs_5d * 2.0 + (vol_ratio - 1.2) * 4.5)

        # Determine candidate Best Sniper Logic even for Standby rows so user sees which logic applies
        if matched_logic_id is not None:
            best_logic_id = matched_logic_id
            best_logic_name = matched_logic_name
        elif meets_5gate_10am or (c_curr > max(s1_floor, sma20) and sma20 >= sma50 * 0.995 and sma50 > sma200):
            best_logic_id = "L6_5GATE_10AM_SNIPER"
            best_logic_name = TOP_5_LOGICS_METADATA[best_logic_id]["name"]
        elif mkt_regime in ("BEAR_OR_CORRECTION", "HIGH_VOL_MACRO_SHOCK") and is_defensive_or_rs:
            best_logic_id = "L4_DEFENSIVE_RS_SNIPER"
            best_logic_name = TOP_5_LOGICS_METADATA[best_logic_id]["name"]
        elif is_global_linked and global_tailwind:
            best_logic_id = "L5_GLOBAL_MACRO_SNIPER"
            best_logic_name = TOP_5_LOGICS_METADATA[best_logic_id]["name"]
        elif had_catalyst_day1:
            best_logic_id = "L3_CATALYST_VWAP_SNIPER"
            best_logic_name = TOP_5_LOGICS_METADATA[best_logic_id]["name"]
        elif tested_support_zone:
            best_logic_id = "L2_SUPPORT_SPRING_SNIPER"
            best_logic_name = TOP_5_LOGICS_METADATA[best_logic_id]["name"]
        else:
            best_logic_id = "L1_SECTOR_ORB_SNIPER"
            best_logic_name = TOP_5_LOGICS_METADATA[best_logic_id]["name"]

        # Actionable CNC Buy Limit Price: if already triggered, buy at Live_Price (or up to entry_trigger + 0.35*ATR);
        # if on standby, order trigger is the 10:00 AM ORB / breakout price (max(entry_trigger, c_curr))
        cnc_buy_limit = round(c_curr, 2) if matched_logic_id is not None else round(max(entry_trigger, c_curr), 2)
        ladder = compute_downside_protection_ladder(cnc_buy_limit, initial_sl, atr, s1_floor)
        swing_pred = predict_swing_conversion_success(
            entry_price=cnc_buy_limit,
            current_price=c_curr,
            day_high=h_curr,
            day_low=l_curr,
            s1_support=s1_floor,
            sma20=sma20,
            sma50=sma50,
            sma200=sma200,
            vol_ratio=vol_ratio,
            market_regime=mkt_regime,
            sl_breached=(c_curr <= ladder["initial_stop_loss"]) or (c_curr < s1_floor),
        )

        # Readiness score even if not triggered today
        readiness = 50.0
        if c_curr > sma200:
            readiness += 12.0
        if c_curr > sma50:
            readiness += 10.0
        if stock_rs_5d > 0.5:
            readiness += 10.0
        if c_curr >= s1_floor and (c_curr - s1_floor) / c_curr * 100.0 <= 3.0:
            readiness += 10.0
        if vol_ratio >= 1.2:
            readiness += 8.0

        final_score = round(max(readiness, conviction), 1)

        # CNC Order Action & Priority Rank (Initial rows = actionable CNC orders first!)
        if matched_logic_id is not None:
            order_priority = 1
            cnc_order_action = f"🟢 BUY CNC NOW ({matched_logic_id.split('_')[0]})"
            cnc_order_type = "CNC LIMIT BUY (Immediate / 10:00 AM)"
        elif meets_5gate_10am:
            order_priority = 2
            cnc_order_action = "🟢 BUY CNC @ 10AM ORB (L6 5-Gate)"
            cnc_order_type = "CNC SL-M / LIMIT TRIGGER @ 10:00 AM"
        elif final_score >= 80.0 and c_curr > s1_floor and stock_rs_5d > 0:
            order_priority = 3
            cnc_order_action = f"🟡 WATCH CNC TRIGGER ({best_logic_id.split('_')[0]})"
            cnc_order_type = "CNC Conditional Limit Above Trigger"
        else:
            order_priority = 4
            cnc_order_action = "⚪ STANDBY (NO CNC ORDER)"
            cnc_order_type = "No Order — Wait for Alignment"

        # Default ₹20,000 MAX CNC cash allocation sizing per ticker (1x Delivery, zero MIS leverage, Max 3 entries/day)
        default_cnc_capital = 20000.0
        suggested_cnc_qty = max(1, int(default_cnc_capital // max(cnc_buy_limit, 1.0)))
        if suggested_cnc_qty * cnc_buy_limit > default_cnc_capital and suggested_cnc_qty > 1:
            suggested_cnc_qty -= 1
        cnc_order_value = round(suggested_cnc_qty * cnc_buy_limit, 2)
        per_share_risk = max(0.05, cnc_buy_limit - ladder["initial_stop_loss"])
        per_share_reward = max(0.05, ladder["intraday_target"] - cnc_buy_limit)
        rr_ratio = round(per_share_reward / per_share_risk, 2)
        max_stop_loss_inr = round(suggested_cnc_qty * per_share_risk, 2)

        # Evaluate Live Intraday GTT Exit / Trailing SL Stage if entered at 10:00 AM ORB (entry_trigger)
        orb_ladder = compute_downside_protection_ladder(entry_trigger, initial_sl, atr, s1_floor)
        mfe_from_trigger = max(0.0, h_curr - entry_trigger) / max(atr, 1e-6) if c_curr >= entry_trigger else 0.0
        active_gtt_sl = ladder["initial_stop_loss"]
        active_trail_stage = "Stage 0: Initial S1/ORB Protected SL"
        if matched_logic_id is not None and c_curr >= entry_trigger:
            if mfe_from_trigger >= 2.20:
                cand_sl = round(h_curr - 0.50 * atr, 2)
                if cand_sl < c_curr:
                    active_gtt_sl = max(active_gtt_sl, cand_sl)
                    active_trail_stage = "Stage 4: Chandelier Trail (HWM - 0.50 ATR)"
            elif mfe_from_trigger >= 1.75 and orb_ladder["stage3_trail_sl"] < c_curr:
                active_gtt_sl = max(active_gtt_sl, orb_ladder["stage3_trail_sl"])
                active_trail_stage = "Stage 3: Profit-Lock (+1.15 ATR / 66% Gain)"
            elif mfe_from_trigger >= 1.25 and orb_ladder["stage2_trail_sl"] < c_curr:
                active_gtt_sl = max(active_gtt_sl, orb_ladder["stage2_trail_sl"])
                active_trail_stage = "Stage 2: Profit-Lock (+0.65 ATR / 50% Gain)"
            elif mfe_from_trigger >= 0.80 and orb_ladder["stage1_trail_sl"] < c_curr:
                active_gtt_sl = max(active_gtt_sl, orb_ladder["stage1_trail_sl"])
                active_trail_stage = "Stage 1: Breakeven+ (+0.18 ATR Above 10AM ORB)"

        gtt_sl_limit = round(active_gtt_sl * 0.997, 2)
        gtt_target_trigger = ladder["intraday_target"]
        gtt_target_limit = ladder["gtt_target_limit_price"]

        if matched_logic_id is not None:
            if c_curr >= gtt_target_trigger:
                exit_readiness = f"🎯 READY FOR EXIT: TARGET HIT (Sell CNC @ ₹{c_curr:.2f})"
            elif c_curr <= active_gtt_sl:
                exit_readiness = f"🛑 READY FOR EXIT: SL HIT (Sell CNC @ ₹{c_curr:.2f})"
            elif active_gtt_sl > ladder["initial_stop_loss"]:
                exit_readiness = f"🔼 MODIFY GTT OCO SL -> ₹{active_gtt_sl:.2f} ({active_trail_stage.split(':')[0]})"
            else:
                exit_readiness = f"🛡️ PLACE GTT OCO SELL: SL ₹{active_gtt_sl:.2f} | TGT ₹{gtt_target_trigger:.2f}"
        elif order_priority <= 3:
            exit_readiness = f"📋 PRE-SET GTT OCO: SL ₹{active_gtt_sl:.2f} | TGT ₹{gtt_target_trigger:.2f}"
        else:
            exit_readiness = "⚪ NO ACTIVE POSITION / GTT STANDBY"

        if swing_pred["predicted_swing_win_rate_pct"] >= 80.0:
            cnc_310pm_action = f"🟢 HOLD CNC OVERNIGHT (5/5 Gates | Raise GTT TGT -> ₹{ladder['swing_conversion_target']:.2f})"
        elif swing_pred["predicted_swing_win_rate_pct"] >= 70.0:
            cnc_310pm_action = f"🟡 HOLD 50% CNC OVERNIGHT (4/5 Gates | Trail GTT SL @ ₹{active_gtt_sl:.2f})"
        else:
            cnc_310pm_action = "🟠 EXIT CNC BY 3:20 PM IST (Cancel GTT & Sell CNC Intraday)"

        rec = {
            # Frozen First 3 Columns (Identifier + CNC Order Action + CNC Order Limit Price)
            "Ticker": sym.replace(".NS", ""),
            "CNC_Order_Action": cnc_order_action,
            "CNC_Buy_Limit_Price": cnc_buy_limit,
            # Immediate Order-Related Columns (Strictly CNC, Zero MIS)
            "Product_Type": "CNC (Delivery - 0% MIS)",
            "CNC_Order_Type": cnc_order_type,
            "Suggested_CNC_Qty": suggested_cnc_qty,
            "CNC_Order_Value_INR": cnc_order_value,
            "Max_Stop_Risk_INR": max_stop_loss_inr,
            "Risk_Reward_Ratio": f"1:{rr_ratio:.2f}",
            "Exit_Readiness_Status": exit_readiness,
            "GTT_Order_Mode": "GTT OCO (2-Leg Sell CNC)",
            "GTT_SL_Trigger": active_gtt_sl,
            "GTT_SL_Limit": gtt_sl_limit,
            "GTT_Target_Trigger": gtt_target_trigger,
            "GTT_Target_Limit": gtt_target_limit,
            "GTT_Swing_Target": ladder["swing_conversion_target"],
            "Active_Trail_Stage": active_trail_stage,
            "CNC_310PM_Decision": cnc_310pm_action,
            # Logic & Technical Context Columns
            "Order_Priority_Rank": order_priority,
            "Best_Logic_ID": best_logic_id,
            "Best_Logic_Name": best_logic_name,
            "Matched_Sniper_Logic_ID": matched_logic_id or "STANDBY",
            "Matched_Sniper_Logic_Name": matched_logic_name or f"Candidate: {best_logic_name}",
            "Sniper_Readiness_Score": final_score,
            "Meets_5Gate_10AM_Entry": bool(meets_5gate_10am),
            "Live_Price": round(c_curr, 2),
            "ORB_10AM_Trigger": entry_trigger,
            "Day_Change_Pct": round(day_ret_pct, 2),
            "5D_Rel_Strength_Pct": round(stock_rs_5d, 2),
            "Sector_5D_RS_Pct": round(sector_rs_5d, 2),
            "RSI_14": round(rsi, 1),
            "Vol_Ratio_20D": round(vol_ratio, 2),
            "Structural_S1_Floor": round(s1_floor, 2),
            "Dist_Above_S1_Pct": round((c_curr - s1_floor) / c_curr * 100.0, 2),
            "Initial_Protected_SL": ladder["initial_stop_loss"],
            "Downside_Risk_Pct": ladder["initial_downside_risk_pct"],
            "Stage1_Breakeven_SL": ladder["stage1_trail_sl"],
            "Stage2_ProfitLock_SL": ladder["stage2_trail_sl"],
            "Stage3_Chandelier_SL": ladder["stage3_trail_sl"],
            "Intraday_Target": ladder["intraday_target"],
            "ATR_14": round(atr, 2),
            "Swing_Conversion_Prob_Pct": swing_pred["predicted_swing_win_rate_pct"],
            "Swing_Conversion_Gates": swing_pred["gates_passed"],
            "Swing_Conversion_Verdict": swing_pred["verdict"],
            "Market": meta["market"],
            "Sector_Group": meta["group"],
        }
        watchlist_rows.append(rec)
        if matched_logic_id is not None:
            sniper_rows.append(rec)

    watchlist_df = pd.DataFrame(watchlist_rows).sort_values(
        ["Order_Priority_Rank", "Sniper_Readiness_Score", "5D_Rel_Strength_Pct"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    # Tag Top 3 Daily Entries (Max 3 entries/day @ <= ₹20,000/ticker = Max ₹60,000/day)
    slot_labels = []
    for i in range(len(watchlist_df)):
        if i < MAX_DAILY_ENTRIES:
            slot_labels.append(f"🏆 Top #{i+1}/3 (<=₹20K)")
        else:
            slot_labels.append("⚪ Standby (>Top 3 Cap)")
    watchlist_df.insert(3, "Daily_Top3_Slot", slot_labels)

    if sniper_rows:
        sniper_df = pd.DataFrame(sniper_rows).sort_values(
            ["Order_Priority_Rank", "Sniper_Readiness_Score", "5D_Rel_Strength_Pct"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        s_slots = [
            f"🏆 Top #{i+1}/3 (<=₹20K)" if i < MAX_DAILY_ENTRIES else "⚪ Standby (>Top 3 Cap)"
            for i in range(len(sniper_df))
        ]
        sniper_df.insert(3, "Daily_Top3_Slot", s_slots)
    else:
        sniper_df = pd.DataFrame(columns=watchlist_df.columns)

    watchlist_df.to_csv(os.path.join(DATA_DIR, "live_watchlist_radar.csv"), index=False)
    sniper_df.to_csv(os.path.join(DATA_DIR, "live_sniper_signals.csv"), index=False)
    with open(os.path.join(DATA_DIR, "live_regime_report.json"), "w", encoding="utf-8") as f:
        json.dump(regime_report, f, indent=2)

    return regime_report, sniper_df, watchlist_df


def _format_performance_verdict(pnl_inr: float, pnl_pct: float, is_closed: bool) -> str:
    tag = "REALIZED" if is_closed else "LIVE"
    if pnl_inr > 0.01:
        return f"🟢 {tag} PROFIT +₹{pnl_inr:,.2f} (+{pnl_pct:.2f}%)"
    elif pnl_inr < -0.01:
        return f"🔴 {tag} LOSS -₹{abs(pnl_inr):,.2f} ({pnl_pct:.2f}%)"
    return f"⚪ {tag} FLAT ₹0.00 (0.00%)"


def evaluate_chronological_exits(
    trades_csv_path: str = SNIPER_TELEMETRY_CSV,
    force_full_replay: bool = False,
    eod_315pm_auto_exit: bool = True,
    sync_purchase_candle_price: bool = True,
    click_timestamp_str: str | None = None,
) -> pd.DataFrame:
    """
    Bug-Free Chronological 5-Minute CNC Auto-Exit & Late-Click Historical Candle Replay Engine:
    - Enforces ₹20,000 max budget per ticker (`MAX_BUDGET_PER_TICKER_INR = 20000.0`) and Max 3 entries/day.
    - Tracks 5-minute candles starting strictly 5 minutes AFTER purchase time (`Purchase_Time + 5m`,
      e.g. `10:05 IST` for a `10:00 AM` entry) up to `3:15 PM IST` (`15:15:00`) or the time the button
      was clicked (`click_timestamp_str`).
    - If a manual click is done late (afternoon, after market close, or on a subsequent day), it loads
      historical 5-minute candles from the purchase date & time up to click time / 3:15 PM to check if
      any exit condition was met (Stop Loss, 4-Stage Trailing Stop, Profit Target, or 3:15 PM EOD close)
      and records the exact exit candle, exit price, and Profit/Loss (`PnL_INR` and `PnL_Pct`).
    """
    if not os.path.exists(trades_csv_path):
        return pd.DataFrame()

    df = pd.read_csv(trades_csv_path)
    if df.empty:
        return df

    for sc in ("Status", "Exit_Timestamp", "Exit_Reason", "GTT_Order_Status", "Performance_Verdict"):
        if sc in df.columns:
            df[sc] = df[sc].fillna("").astype(object)

    # Ensure all required CNC, GTT, Top-3 Slot, Replay & Performance columns exist with proper dtypes
    str_cols = [
        "Daily_Slot", "Ticker", "Product_Type", "Purchase_Date", "Purchase_Time",
        "GTT_Order_Status", "Active_Trail_Stage", "CNC_310PM_Decision",
        "Performance_Verdict", "Replay_Audit_Trail", "Status",
        "Entry_Timestamp", "Last_Evaluated_Bar_TS", "Exit_Timestamp", "Exit_Reason",
    ]
    for col, default_val in [
        ("Daily_Slot", "Slot #1/3"),
        ("Product_Type", "CNC (Delivery - 0% MIS)"),
        ("CNC_Qty", 1),
        ("Capital_Used_INR", 0.0),
        ("Max_Budget_INR", MAX_BUDGET_PER_TICKER_INR),
        ("Purchase_Date", ""),
        ("Purchase_Time", "10:00:00"),
        ("GTT_Order_Status", "🛡️ GTT OCO ARMED (HOLD CNC)"),
        ("GTT_SL_Trigger", 0.0),
        ("GTT_SL_Limit", 0.0),
        ("GTT_Target_Trigger", 0.0),
        ("GTT_Target_Limit", 0.0),
        ("Active_Trail_Stage", "Stage 0: Initial S1/ORB Floor"),
        ("CNC_310PM_Decision", "🟠 EXIT CNC BY 3:15 PM IST"),
        ("PnL_INR", 0.0),
        ("PnL_Pct", 0.0),
        ("Performance_Verdict", "⚪ FLAT ₹0.00 (0.00%)"),
        ("Bars_Tracked_5m", 0),
        ("Window_Low_5m", 0.0),
        ("Window_High_5m", 0.0),
        ("Replay_Audit_Trail", ""),
        ("Exit_Timestamp", ""),
        ("Exit_Reason", ""),
    ]:
        if col not in df.columns:
            df[col] = default_val
    for sc in str_cols:
        if sc in df.columns:
            df[sc] = df[sc].fillna("").astype(object)

    if force_full_replay:
        eval_mask = pd.Series(True, index=df.index)
    else:
        eval_mask = df["Status"].isin(["ACTIVE_INTRADAY", "SWING_CONVERTED"])

    if not eval_mask.any():
        return df

    click_ts = pd.to_datetime(click_timestamp_str) if click_timestamp_str else pd.Timestamp.now()

    # Determine if any trade requires >5d of 5m history (up to 60d supported by Yahoo Finance for 5m)
    min_entry_ts = click_ts
    for idx in df[eval_mask].index:
        raw_ts = str(df.at[idx, "Entry_Timestamp"]) if "Entry_Timestamp" in df.columns else ""
        try:
            ets = pd.to_datetime(raw_ts)
            if ets < min_entry_ts:
                min_entry_ts = ets
        except Exception:
            pass
    days_back = max(1, (click_ts - min_entry_ts).days + 2)
    yf_period = "60d" if days_back > 5 else "5d"

    tickers = sorted(list({f"{str(t).strip().replace('.NS', '')}.NS" for t in df.loc[eval_mask, "Ticker"].unique()}))
    data_5m = yf.download(tickers, period=yf_period, interval="5m", group_by="ticker", progress=False, threads=True)

    for idx in df[eval_mask].index:
        ticker = str(df.at[idx, "Ticker"]).strip().replace(".NS", "")
        sym = f"{ticker}.NS"
        entry_p = float(df.at[idx, "Entry_Price"])
        s1_floor = float(df.at[idx, "Structural_S1_Floor"]) if "Structural_S1_Floor" in df.columns and pd.notna(df.at[idx, "Structural_S1_Floor"]) else round(entry_p * 0.98, 2)
        atr = float(df.at[idx, "ATR_At_Entry"]) if "ATR_At_Entry" in df.columns and pd.notna(df.at[idx, "ATR_At_Entry"]) else round(entry_p * 0.015, 2)
        init_sl = float(df.at[idx, "Initial_SL"]) if "Initial_SL" in df.columns and pd.notna(df.at[idx, "Initial_SL"]) else round(entry_p - 0.95 * atr, 2)
        target_p = float(df.at[idx, "Target_Price"]) if "Target_Price" in df.columns and pd.notna(df.at[idx, "Target_Price"]) else round(entry_p + 1.80 * atr, 2)

        # Parse Entry_Timestamp; if outside market hours (e.g. evening test click), normalize purchase time to 10:00:00 on the session date
        raw_entry_ts = str(df.at[idx, "Entry_Timestamp"]) if "Entry_Timestamp" in df.columns and pd.notna(df.at[idx, "Entry_Timestamp"]) else click_ts.strftime("%Y-%m-%d 10:00:00")
        try:
            entry_ts = pd.to_datetime(raw_entry_ts)
        except Exception:
            entry_ts = pd.to_datetime(click_ts.strftime("%Y-%m-%d 10:00:00"))

        bars = pd.DataFrame()
        if len(tickers) == 1 and "Close" in data_5m.columns and not isinstance(data_5m.columns, pd.MultiIndex):
            bars = data_5m.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        elif sym in data_5m.columns.get_level_values(0):
            bars = data_5m[sym].dropna(subset=["Open", "High", "Low", "Close"]).copy()

        if not bars.empty:
            if bars.index.tz is not None:
                bars.index = bars.index.tz_convert("Asia/Kolkata").tz_localize(None)

            # If entry_ts date has no bars (e.g. weekend/holiday) or time is after 15:10, map to latest available session date @ 10:00 AM
            avail_dates = sorted(list(set(bars.index.date)))
            if entry_ts.date() not in avail_dates and avail_dates:
                # Pick latest session date <= entry_ts.date(), or latest overall
                prior_dates = [d for d in avail_dates if d <= entry_ts.date()]
                session_date = prior_dates[-1] if prior_dates else avail_dates[-1]
                p_time_str = str(df.at[idx, "Purchase_Time"]) if pd.notna(df.at[idx, "Purchase_Time"]) and str(df.at[idx, "Purchase_Time"]).strip() not in ("", "nan") else "10:00:00"
                entry_ts = pd.to_datetime(f"{session_date} {p_time_str}")

            if entry_ts.time() > pd.Timestamp("15:10:00").time() or entry_ts.time() < pd.Timestamp("09:15:00").time():
                entry_ts = pd.to_datetime(f"{entry_ts.date()} 10:00:00")
                df.at[idx, "Entry_Timestamp"] = entry_ts.strftime("%Y-%m-%d %H:%M:%S")

            df.at[idx, "Purchase_Date"] = str(entry_ts.date())
            df.at[idx, "Purchase_Time"] = entry_ts.strftime("%H:%M:%S")

            # Optional: anchor Entry_Price to the actual 5m candle close at Purchase_Time
            if sync_purchase_candle_price:
                same_day_bars = bars[(bars.index.date == entry_ts.date()) & (bars.index <= entry_ts)]
                if not same_day_bars.empty:
                    entry_p = round(float(same_day_bars["Close"].iloc[-1]), 2)
                    df.at[idx, "Entry_Price"] = entry_p
                    ladder_recalc = compute_downside_protection_ladder(entry_p, entry_p - 0.95 * atr, atr, s1_floor)
                    init_sl = ladder_recalc["initial_stop_loss"]
                    target_p = ladder_recalc["intraday_target"]
                    df.at[idx, "Initial_SL"] = init_sl
                    df.at[idx, "Target_Price"] = target_p

        # Enforce <= ₹20,000 max budget per ticker
        budget_cap = min(MAX_BUDGET_PER_TICKER_INR, float(df.at[idx, "Max_Budget_INR"]) if "Max_Budget_INR" in df.columns and pd.notna(df.at[idx, "Max_Budget_INR"]) and float(df.at[idx, "Max_Budget_INR"]) > 0 else MAX_BUDGET_PER_TICKER_INR)
        cnc_qty = max(1, int(budget_cap // max(entry_p, 1.0)))
        if cnc_qty * entry_p > budget_cap and cnc_qty > 1:
            cnc_qty -= 1
        capital_used = round(cnc_qty * entry_p, 2)

        df.at[idx, "Product_Type"] = "CNC (Delivery - 0% MIS)"
        df.at[idx, "Max_Budget_INR"] = budget_cap
        df.at[idx, "CNC_Qty"] = cnc_qty
        df.at[idx, "Capital_Used_INR"] = capital_used

        # Tracking starts strictly 5 minutes after purchase time (e.g. 10:05:00 for 10:00:00 entry)
        track_start_ts = entry_ts + pd.Timedelta(minutes=5)
        eod_315_ts = pd.to_datetime(f"{entry_ts.date()} 15:15:00")

        if force_full_replay:
            curr_sl = init_sl
            hwm = entry_p
            cutoff_ts = track_start_ts - pd.Timedelta(seconds=1)
            df.at[idx, "Status"] = "ACTIVE_INTRADAY"
            df.at[idx, "Exit_Price"] = np.nan
            df.at[idx, "Exit_Timestamp"] = ""
            df.at[idx, "Exit_Reason"] = ""
        else:
            curr_sl = float(df.at[idx, "Current_Trailing_SL"]) if pd.notna(df.at[idx, "Current_Trailing_SL"]) else init_sl
            hwm = float(df.at[idx, "High_Water_Mark"]) if "High_Water_Mark" in df.columns and pd.notna(df.at[idx, "High_Water_Mark"]) else entry_p
            last_eval_str = str(df.at[idx, "Last_Evaluated_Bar_TS"]) if "Last_Evaluated_Bar_TS" in df.columns and pd.notna(df.at[idx, "Last_Evaluated_Bar_TS"]) else ""
            try:
                parsed_last = pd.to_datetime(last_eval_str) if last_eval_str and last_eval_str != "nan" else (track_start_ts - pd.Timedelta(seconds=1))
                cutoff_ts = max(parsed_last, track_start_ts - pd.Timedelta(seconds=1))
            except Exception:
                cutoff_ts = track_start_ts - pd.Timedelta(seconds=1)

        exited = False
        bars_tracked = int(df.at[idx, "Bars_Tracked_5m"]) if not force_full_replay and pd.notna(df.at[idx, "Bars_Tracked_5m"]) else 0
        win_low = entry_p
        win_high = entry_p
        stage_events = []

        if not bars.empty:
            # Filter 5m candles on purchase date from Purchase_Time + 5m up to min(click_ts, 15:15 IST)
            # (or multi-day if SWING_CONVERTED and not eod_315pm_auto_exit)
            if eod_315pm_auto_exit:
                upper_bound_ts = min(click_ts, eod_315_ts)
                window_bars = bars[(bars.index > cutoff_ts) & (bars.index >= track_start_ts) & (bars.index <= upper_bound_ts) & (bars.index.date == entry_ts.date())]
            else:
                window_bars = bars[(bars.index > cutoff_ts) & (bars.index >= track_start_ts) & (bars.index <= click_ts)]

            if not window_bars.empty:
                win_low = float(window_bars["Low"].min())
                win_high = float(window_bars["High"].max())

            for b_ts, b_row in window_bars.iterrows():
                # Skip any post-3:15 PM bars on purchase date
                if b_ts.date() == entry_ts.date() and b_ts.time() > pd.Timestamp("15:15:00").time():
                    continue

                bars_tracked += 1
                bl = float(b_row["Low"])
                bh = float(b_row["High"])
                bc = float(b_row["Close"])

                # 1) Check active stop loss / trailing stop BEFORE upgrading HWM/SL from current bar's high
                if bl <= curr_sl:
                    exit_p = round(curr_sl, 2)
                    pnl_pct = round((exit_p - entry_p) / entry_p * 100.0, 2)
                    pnl_inr = round((exit_p - entry_p) * cnc_qty, 2)
                    df.at[idx, "Status"] = "STOP_OR_TRAIL_EXIT"
                    df.at[idx, "Exit_Price"] = exit_p
                    df.at[idx, "Exit_Timestamp"] = b_ts.strftime("%Y-%m-%d %H:%M:%S")
                    df.at[idx, "Exit_Reason"] = f"🛑 5m Candle Low (₹{bl:.2f}) hit SL/Trail ₹{exit_p:.2f} at {b_ts.strftime('%H:%M')}"
                    df.at[idx, "GTT_Order_Status"] = f"🛑 AUTO-EXITED @ ₹{exit_p:.2f} ({b_ts.strftime('%H:%M')} SL/Trail Hit)"
                    df.at[idx, "Live_CMP"] = exit_p
                    df.at[idx, "PnL_Pct"] = pnl_pct
                    df.at[idx, "PnL_INR"] = pnl_inr
                    df.at[idx, "Performance_Verdict"] = _format_performance_verdict(pnl_inr, pnl_pct, True)
                    df.at[idx, "Last_Evaluated_Bar_TS"] = b_ts.strftime("%Y-%m-%d %H:%M:%S")
                    stage_events.append(f"{b_ts.strftime('%H:%M')} SL/Trail Exit @ ₹{exit_p:.2f}")
                    exited = True
                    break

                # 2) Check intraday profit target achievement
                if bh >= target_p:
                    exit_p = round(target_p, 2)
                    pnl_pct = round((exit_p - entry_p) / entry_p * 100.0, 2)
                    pnl_inr = round((exit_p - entry_p) * cnc_qty, 2)
                    df.at[idx, "Status"] = "TARGET_ACHIEVED"
                    df.at[idx, "Exit_Price"] = exit_p
                    df.at[idx, "Exit_Timestamp"] = b_ts.strftime("%Y-%m-%d %H:%M:%S")
                    df.at[idx, "Exit_Reason"] = f"🎯 5m Candle High (₹{bh:.2f}) hit Profit Target ₹{exit_p:.2f} at {b_ts.strftime('%H:%M')}"
                    df.at[idx, "GTT_Order_Status"] = f"🎯 AUTO-EXITED @ ₹{exit_p:.2f} ({b_ts.strftime('%H:%M')} Target Hit)"
                    df.at[idx, "Live_CMP"] = exit_p
                    df.at[idx, "PnL_Pct"] = pnl_pct
                    df.at[idx, "PnL_INR"] = pnl_inr
                    df.at[idx, "Performance_Verdict"] = _format_performance_verdict(pnl_inr, pnl_pct, True)
                    df.at[idx, "Last_Evaluated_Bar_TS"] = b_ts.strftime("%Y-%m-%d %H:%M:%S")
                    stage_events.append(f"{b_ts.strftime('%H:%M')} Target Hit @ ₹{exit_p:.2f}")
                    exited = True
                    break

                # 3) Upgrade High-Water-Mark and 4-Stage Trailing Stop for SUBSEQUENT 5m bars
                prev_sl = curr_sl
                hwm = max(hwm, bh)
                mfe_atr = (hwm - entry_p) / max(atr, 1e-6)
                if mfe_atr >= 2.20:
                    curr_sl = max(curr_sl, round(hwm - 0.50 * atr, 2))
                elif mfe_atr >= 1.75:
                    curr_sl = max(curr_sl, round(entry_p + 1.15 * atr, 2))
                elif mfe_atr >= 1.25:
                    curr_sl = max(curr_sl, round(entry_p + 0.65 * atr, 2))
                elif mfe_atr >= 0.80:
                    curr_sl = max(curr_sl, round(entry_p + 0.18 * atr, 2))

                if curr_sl > prev_sl:
                    stage_events.append(f"{b_ts.strftime('%H:%M')} Trail SL->{curr_sl:.2f}")

                df.at[idx, "High_Water_Mark"] = round(hwm, 2)
                df.at[idx, "Current_Trailing_SL"] = round(curr_sl, 2)
                df.at[idx, "Live_CMP"] = round(bc, 2)
                pnl_pct = round((bc - entry_p) / entry_p * 100.0, 2)
                pnl_inr = round((bc - entry_p) * cnc_qty, 2)
                df.at[idx, "PnL_Pct"] = pnl_pct
                df.at[idx, "PnL_INR"] = pnl_inr
                df.at[idx, "Performance_Verdict"] = _format_performance_verdict(pnl_inr, pnl_pct, False)
                df.at[idx, "Last_Evaluated_Bar_TS"] = b_ts.strftime("%Y-%m-%d %H:%M:%S")

                # 4) Check 3:15 PM IST (15:15:00) Auto-Exit Rule on Purchase Date
                if eod_315pm_auto_exit and b_ts.date() == entry_ts.date() and b_ts.time() >= pd.Timestamp("15:15:00").time():
                    exit_p = round(bc, 2)
                    pnl_pct = round((exit_p - entry_p) / entry_p * 100.0, 2)
                    pnl_inr = round((exit_p - entry_p) * cnc_qty, 2)
                    df.at[idx, "Status"] = "AUTO_EXIT_315PM_CLOSE"
                    df.at[idx, "Exit_Price"] = exit_p
                    df.at[idx, "Exit_Timestamp"] = b_ts.strftime("%Y-%m-%d %H:%M:%S")
                    df.at[idx, "Exit_Reason"] = f"⏰ 3:15 PM IST Auto-Exit @ Candle Close ₹{exit_p:.2f} (Peak ₹{hwm:.2f}, Trail SL ₹{curr_sl:.2f})"
                    df.at[idx, "GTT_Order_Status"] = f"⏰ AUTO-EXITED @ ₹{exit_p:.2f} (3:15 PM IST Close)"
                    df.at[idx, "Live_CMP"] = exit_p
                    df.at[idx, "PnL_Pct"] = pnl_pct
                    df.at[idx, "PnL_INR"] = pnl_inr
                    df.at[idx, "Performance_Verdict"] = _format_performance_verdict(pnl_inr, pnl_pct, True)
                    stage_events.append(f"15:15 3:15PM Close Exit @ ₹{exit_p:.2f}")
                    exited = True
                    break

        df.at[idx, "Bars_Tracked_5m"] = bars_tracked
        df.at[idx, "Window_Low_5m"] = round(win_low, 2)
        df.at[idx, "Window_High_5m"] = round(win_high, 2)

        if not exited:
            df.at[idx, "High_Water_Mark"] = round(hwm, 2)
            df.at[idx, "Current_Trailing_SL"] = round(curr_sl, 2)
            live_cmp = float(df.at[idx, "Live_CMP"]) if "Live_CMP" in df.columns and pd.notna(df.at[idx, "Live_CMP"]) else entry_p
            pnl_pct = round((live_cmp - entry_p) / entry_p * 100.0, 2)
            pnl_inr = round((live_cmp - entry_p) * cnc_qty, 2)
            df.at[idx, "PnL_Pct"] = pnl_pct
            df.at[idx, "PnL_INR"] = pnl_inr
            df.at[idx, "Performance_Verdict"] = _format_performance_verdict(pnl_inr, pnl_pct, False)

            mfe_atr = (hwm - entry_p) / max(atr, 1e-6)
            if mfe_atr >= 2.20:
                stage_lbl = "Stage 4: Chandelier Trail (HWM - 0.50 ATR)"
                gtt_status = f"🔼 MODIFY GTT SL -> ₹{curr_sl:.2f} (Stage 4 Chandelier)"
            elif mfe_atr >= 1.75:
                stage_lbl = "Stage 3: Profit-Lock (+1.15 ATR)"
                gtt_status = f"🔼 MODIFY GTT SL -> ₹{curr_sl:.2f} (Stage 3 Lock)"
            elif mfe_atr >= 1.25:
                stage_lbl = "Stage 2: Profit-Lock (+0.65 ATR)"
                gtt_status = f"🔼 MODIFY GTT SL -> ₹{curr_sl:.2f} (Stage 2 Lock)"
            elif mfe_atr >= 0.80:
                stage_lbl = "Stage 1: Breakeven+ (+0.18 ATR)"
                gtt_status = f"🔼 MODIFY GTT SL -> ₹{curr_sl:.2f} (Stage 1 Breakeven+)"
            else:
                stage_lbl = "Stage 0: Initial S1/ORB Floor"
                gtt_status = f"🛡️ GTT OCO ARMED: SL ₹{curr_sl:.2f} | TGT ₹{target_p:.2f}"

            df.at[idx, "Active_Trail_Stage"] = stage_lbl
            df.at[idx, "GTT_Order_Status"] = gtt_status

        audit_head = f"Tracked {bars_tracked} bars ({track_start_ts.strftime('%H:%M')}->15:15) | Range ₹{win_low:.2f}-₹{win_high:.2f}"
        if stage_events:
            audit_head += " | " + " -> ".join(stage_events[-3:])
        df.at[idx, "Replay_Audit_Trail"] = audit_head

        df.at[idx, "GTT_SL_Trigger"] = round(curr_sl, 2)
        df.at[idx, "GTT_SL_Limit"] = round(curr_sl * 0.997, 2)
        df.at[idx, "GTT_Target_Trigger"] = round(target_p, 2)
        df.at[idx, "GTT_Target_Limit"] = round(target_p * 0.999, 2)

        prob = float(df.at[idx, "Swing_Conversion_Prob_Pct"]) if "Swing_Conversion_Prob_Pct" in df.columns and pd.notna(df.at[idx, "Swing_Conversion_Prob_Pct"]) else 50.0
        if prob >= 80.0:
            df.at[idx, "CNC_310PM_Decision"] = "🟢 3:15 PM Auto-Exit OR Hold CNC (Passed 5/5 Swing Gates)"
        elif prob >= 70.0:
            df.at[idx, "CNC_310PM_Decision"] = "🟡 3:15 PM Auto-Exit OR Hold 50% CNC (4/5 Gates)"
        else:
            df.at[idx, "CNC_310PM_Decision"] = "🟠 AUTO-EXIT CNC AT 3:15 PM IST"

    df.to_csv(trades_csv_path, index=False)
    if trades_csv_path != SNIPER_TRADES_CSV:
        df.to_csv(SNIPER_TRADES_CSV, index=False)
    return df


def stage_ticker_into_top3_cnc(
    ticker: str,
    watchlist_df: pd.DataFrame,
    regime_badge: str = "TRENDING_OR_ROTATION",
    entry_price: float | None = None,
    budget_inr: float = MAX_BUDGET_PER_TICKER_INR,
    purchase_date_str: str | None = None,
    purchase_time_str: str = "10:00:00",
    eod_315pm_auto_exit: bool = True,
    sync_purchase_candle_price: bool = False,
    replace_slot3_if_full: bool = True,
    trades_csv_path: str = SNIPER_TELEMETRY_CSV,
) -> tuple[bool, str, pd.DataFrame]:
    """
    Stages a selected ticker into the Top 3 daily CNC entries (`Slot #1/3`, `Slot #2/3`, `Slot #3/3`)
    with a strict maximum budget of ₹20,000 per ticker (`MAX_BUDGET_PER_TICKER_INR = 20000.0`)
    and max 3 entries per day (`MAX_DAILY_ENTRIES = 3`), then immediately runs the 5-minute
    candle replay from `Purchase_Time + 5m` through `3:15 PM IST` / click time.
    """
    clean_ticker = str(ticker).strip().replace(".NS", "")
    budget_cap = min(MAX_BUDGET_PER_TICKER_INR, max(1000.0, float(budget_inr)))

    row_match = watchlist_df[watchlist_df["Ticker"] == clean_ticker]
    if row_match.empty:
        return False, f"Ticker {clean_ticker} not found in live watchlist radar.", pd.DataFrame()
    r = row_match.iloc[0]

    # Resolve purchase date (if not provided, use latest session date from existing ledger or today)
    now_ist = pd.Timestamp.now()
    if not purchase_date_str:
        purchase_date_str = now_ist.strftime("%Y-%m-%d")
    if len(purchase_time_str.strip()) == 5:
        purchase_time_str = purchase_time_str.strip() + ":00"
    entry_ts_str = f"{purchase_date_str} {purchase_time_str}"

    eff_entry_p = round(float(entry_price if entry_price and entry_price > 0 else r["CNC_Buy_Limit_Price"]), 2)
    s1_floor = round(float(r["Structural_S1_Floor"]), 2)
    atr_val = round(float(r["ATR_14"]) if "ATR_14" in r and pd.notna(r["ATR_14"]) else max(eff_entry_p * 0.015, ( float(r["Intraday_Target"]) - eff_entry_p ) / 1.80), 2)
    ladder = compute_downside_protection_ladder(eff_entry_p, eff_entry_p - 0.95 * atr_val, atr_val, s1_floor)

    cnc_qty = max(1, int(budget_cap // max(eff_entry_p, 1.0)))
    if cnc_qty * eff_entry_p > budget_cap and cnc_qty > 1:
        cnc_qty -= 1
    cap_used = round(cnc_qty * eff_entry_p, 2)

    if os.path.exists(trades_csv_path):
        df = pd.read_csv(trades_csv_path)
    else:
        df = pd.DataFrame()

    if not df.empty and "Purchase_Date" not in df.columns and "Entry_Timestamp" in df.columns:
        df["Purchase_Date"] = df["Entry_Timestamp"].astype(str).str.slice(0, 10)

    # Check existing entries for the same session date (or overall Top 3 active deck)
    existing_idx = None
    if not df.empty and "Ticker" in df.columns:
        same_t = df[df["Ticker"].astype(str) == clean_ticker]
        if not same_t.empty:
            existing_idx = same_t.index[-1]

    if existing_idx is None and not df.empty:
        if len(df) >= MAX_DAILY_ENTRIES:
            if replace_slot3_if_full:
                # Replace Slot #3/3 so Top 3 cap (max 3 entries/day) is strictly preserved
                existing_idx = df.index[MAX_DAILY_ENTRIES - 1]
                df = df.iloc[:MAX_DAILY_ENTRIES].copy()
            else:
                return (
                    False,
                    f"⚠️ Max {MAX_DAILY_ENTRIES} entries/day cap reached (₹60,000 max daily budget). Enable 'Replace Slot #3' or reset slots.",
                    df,
                )

    slot_num = (df.index.get_loc(existing_idx) + 1) if existing_idx is not None else (len(df) + 1)
    slot_label = f"Slot #{min(slot_num, MAX_DAILY_ENTRIES)}/3"

    new_rec = {
        "Daily_Slot": slot_label,
        "Ticker": clean_ticker,
        "Product_Type": "CNC (Delivery - 0% MIS)",
        "CNC_Qty": cnc_qty,
        "Capital_Used_INR": cap_used,
        "Max_Budget_INR": budget_cap,
        "Purchase_Date": purchase_date_str,
        "Purchase_Time": purchase_time_str,
        "Logic_ID": r.get("Matched_Sniper_Logic_ID", r.get("Best_Logic_ID", "L6_5GATE_10AM_ORB")),
        "Logic_Name": r.get("Matched_Sniper_Logic_Name", r.get("Best_Logic_Name", "10:00 AM 5-Gate ORB")),
        "Market_Regime": regime_badge,
        "Entry_Timestamp": entry_ts_str,
        "Entry_Price": eff_entry_p,
        "Live_CMP": round(float(r["Live_Price"]), 2),
        "Structural_S1_Floor": s1_floor,
        "ATR_At_Entry": atr_val,
        "Initial_SL": ladder["initial_stop_loss"],
        "Current_Trailing_SL": ladder["initial_stop_loss"],
        "High_Water_Mark": eff_entry_p,
        "Stage1_Breakeven_SL": ladder["stage1_trail_sl"],
        "Stage2_ProfitLock_SL": ladder["stage2_trail_sl"],
        "Stage3_Chandelier_SL": ladder["stage3_trail_sl"],
        "Target_Price": ladder["intraday_target"],
        "GTT_Order_Status": f"🛡️ GTT OCO ARMED: SL ₹{ladder['initial_stop_loss']:.2f} | TGT ₹{ladder['intraday_target']:.2f}",
        "GTT_SL_Trigger": ladder["initial_stop_loss"],
        "GTT_SL_Limit": ladder["gtt_sl_limit_price"],
        "GTT_Target_Trigger": ladder["intraday_target"],
        "GTT_Target_Limit": ladder["gtt_target_limit_price"],
        "Active_Trail_Stage": "Stage 0: Initial S1/ORB Floor",
        "CNC_310PM_Decision": r.get("CNC_310PM_Decision", "🟠 AUTO-EXIT CNC AT 3:15 PM IST"),
        "Swing_Conversion_Prob_Pct": r.get("Swing_Conversion_Prob_Pct", 84.9),
        "Swing_Conversion_Verdict": r.get("Swing_Conversion_Verdict", ""),
        "Status": "ACTIVE_INTRADAY",
        "PnL_INR": 0.0,
        "PnL_Pct": 0.0,
        "Performance_Verdict": "⚪ FLAT ₹0.00 (0.00%)",
        "Bars_Tracked_5m": 0,
        "Window_Low_5m": eff_entry_p,
        "Window_High_5m": eff_entry_p,
        "Replay_Audit_Trail": "",
        "Last_Evaluated_Bar_TS": entry_ts_str,
        "Exit_Timestamp": "",
        "Exit_Price": np.nan,
        "Exit_Reason": "",
    }

    if existing_idx is not None:
        for k, v in new_rec.items():
            df.at[existing_idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_rec])], ignore_index=True)

    # Strictly cap ledger to Top 3 entries for the day and re-number slots #1/3, #2/3, #3/3
    df = df.head(MAX_DAILY_ENTRIES).reset_index(drop=True)
    df["Daily_Slot"] = [f"Slot #{i+1}/3" for i in range(len(df))]
    df.to_csv(trades_csv_path, index=False)

    # Immediately run the 5-minute candle replay from Purchase_Time + 5m to min(click_time, 3:15 PM)
    df = evaluate_chronological_exits(
        trades_csv_path=trades_csv_path,
        force_full_replay=True,
        eod_315pm_auto_exit=eod_315pm_auto_exit,
        sync_purchase_candle_price=sync_purchase_candle_price,
    )
    row_out = df[df["Ticker"] == clean_ticker].iloc[-1]
    msg = (
        f"✅ Staged **{clean_ticker}** in **{row_out['Daily_Slot']}** | "
        f"Qty: **{int(row_out['CNC_Qty'])} CNC** (`₹{float(row_out['Capital_Used_INR']):,.2f} <= ₹20,000`) | "
        f"Entry: `₹{float(row_out['Entry_Price']):,.2f}` @ `{row_out['Purchase_Time']}` | "
        f"Status: **{row_out['Status']}** | **{row_out['Performance_Verdict']}**"
    )
    return True, msg, df


def stage_top3_daily_cnc_entries(
    watchlist_df: pd.DataFrame,
    regime_badge: str = "TRENDING_OR_ROTATION",
    purchase_time_str: str = "10:00:00",
    eod_315pm_auto_exit: bool = True,
    sync_purchase_candle_price: bool = True,
    trades_csv_path: str = SNIPER_TELEMETRY_CSV,
) -> pd.DataFrame:
    """
    Populates the Top 3 daily CNC slots (`Slot #1/3`, `Slot #2/3`, `Slot #3/3`) using the top 3
    ranked candidates from `watchlist_df` with max ₹20,000 budget per ticker (<= ₹60,000 total)
    and replays 5-minute candles from `Purchase_Time + 5m` through `3:15 PM IST`.
    """
    if watchlist_df.empty:
        return pd.DataFrame()

    # Clear existing file so Top 3 slots are cleanly built
    if os.path.exists(trades_csv_path):
        os.remove(trades_csv_path)

    top3 = watchlist_df.head(MAX_DAILY_ENTRIES)
    df_out = pd.DataFrame()
    for _, r in top3.iterrows():
        _, _, df_out = stage_ticker_into_top3_cnc(
            ticker=str(r["Ticker"]),
            watchlist_df=watchlist_df,
            regime_badge=regime_badge,
            entry_price=float(r["CNC_Buy_Limit_Price"]),
            budget_inr=MAX_BUDGET_PER_TICKER_INR,
            purchase_time_str=purchase_time_str,
            eod_315pm_auto_exit=eod_315pm_auto_exit,
            sync_purchase_candle_price=sync_purchase_candle_price,
            trades_csv_path=trades_csv_path,
        )
    return df_out


def force_manual_exit_trade(
    ticker: str = "ALL",
    trades_csv_path: str = SNIPER_TELEMETRY_CSV,
) -> pd.DataFrame:
    """
    Immediately exits the specified ticker (or ALL active trades) at the latest 5-minute / live CMP
    and calculates realized Profit or Loss (`PnL_INR` and `PnL_Pct`).
    """
    if not os.path.exists(trades_csv_path):
        return pd.DataFrame()
    df = pd.read_csv(trades_csv_path)
    if df.empty:
        return df

    for sc in ("Status", "Exit_Timestamp", "Exit_Reason", "GTT_Order_Status", "Performance_Verdict"):
        if sc in df.columns:
            df[sc] = df[sc].fillna("").astype(object)

    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx in df.index:
        t_sym = str(df.at[idx, "Ticker"]).strip()
        if ticker != "ALL" and t_sym != ticker:
            continue
        entry_p = float(df.at[idx, "Entry_Price"])
        cmp_p = float(df.at[idx, "Live_CMP"]) if "Live_CMP" in df.columns and pd.notna(df.at[idx, "Live_CMP"]) else entry_p
        qty = int(df.at[idx, "CNC_Qty"]) if "CNC_Qty" in df.columns and pd.notna(df.at[idx, "CNC_Qty"]) else max(1, int(MAX_BUDGET_PER_TICKER_INR // max(entry_p, 1.0)))
        pnl_pct = round((cmp_p - entry_p) / entry_p * 100.0, 2)
        pnl_inr = round((cmp_p - entry_p) * qty, 2)
        df.at[idx, "Status"] = "MANUAL_EXIT_EXECUTED"
        df.at[idx, "Exit_Price"] = round(cmp_p, 2)
        df.at[idx, "Exit_Timestamp"] = now_str
        df.at[idx, "Exit_Reason"] = f"⚡ Manual Exit Button Clicked @ CMP ₹{cmp_p:.2f}"
        df.at[idx, "GTT_Order_Status"] = f"⚡ MANUAL EXIT @ ₹{cmp_p:.2f} (CNC Sold)"
        df.at[idx, "PnL_Pct"] = pnl_pct
        df.at[idx, "PnL_INR"] = pnl_inr
        df.at[idx, "Performance_Verdict"] = _format_performance_verdict(pnl_inr, pnl_pct, True)

    df.to_csv(trades_csv_path, index=False)
    if trades_csv_path != SNIPER_TRADES_CSV:
        df.to_csv(SNIPER_TRADES_CSV, index=False)
    return df


if __name__ == "__main__":
    report, snipers, watchlist = scan_live_sniper_opportunities()
    print(f"Live Regime: {report['regime_badge']} | Nifty: {report['nifty_close']} ({report['nifty_ret_1d_pct']:+.2f}%) | VIX: {report['india_vix']}")
    print(f"Authorized Sniper Logics Today: {report['authorized_sniper_logics']}")
    print(f"High-Conviction Sniper Signals Triggered Today: {len(snipers)}")
    if not snipers.empty:
        print(snipers[["Ticker", "Daily_Top3_Slot", "Matched_Sniper_Logic_ID", "Live_Price", "Suggested_CNC_Qty", "CNC_Order_Value_INR", "Initial_Protected_SL", "Intraday_Target"]].to_string(index=False))
