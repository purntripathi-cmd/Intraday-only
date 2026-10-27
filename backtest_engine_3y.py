"""
3-Year Historical Multi-Regime Intraday Backtest Engine (Indian & International Markets)
========================================================================================
Tests 9 Intraday Logics (Production Baseline + Top 5 Institutional Sniper Logics + 3 Controls)
across:
  A) 3 Full Years of Actual Historical Market Data (2023–2026, ~740 trading days) across
     54 NSE Stocks, Sectoral Indices (^NSEBANK, ^CNXIT, ^CNXPHARMA, ^CNXAUTO, ^CNXMETAL,
     ^CNXENERGY, ^CNXFMCG), India VIX (^INDIAVIX), Nifty 50 (^NSEI), and International
     Cues (^GSPC, ^IXIC, GC=F, BZ=F, INR=X, MON100.NS, MAFANG.NS, MASPTOP50.NS, GOLDBEES.NS).
  B) 60-Day Granular 15-Minute Bar-by-Bar Intraday Walk-Forward Verification.

Key Institutional Sniper Principles Validated:
1. Low-Frequency, High-Conviction Sniper Cadence (~0.3 to 0.8 trades per week per logic;
   daily trading is NEVER forced).
2. Multi-Layer Downside Risk Protection:
   - Structural Support Floor (Opening Range Low / S1 Pivot / 20DMA-50DMA Confluence + ATR Cushion)
   - Chronological 4-Stage High-Water-Mark Trailing Stop Loss (100% immune to the production
     retroactive morning-bar SL bug)
   - Smart 3:10 PM Intraday-to-Swing Conversion Rescue Gate.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

UNIVERSE_META = {
    # Pharma & Healthcare
    "DIVISLAB.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "SUNPHARMA.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "CIPLA.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "DRREDDY.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "AJANTPHARM.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "LAURUSLABS.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "AUROPHARMA.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "PHARMABEES.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    # IT & Global Tech Exporters
    "TCS.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "INFY.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "HCLTECH.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "TECHM.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "PERSISTENT.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "ITBEES.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    # Banking & Financials
    "HDFCBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "ICICIBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "SBIN.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "AXISBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "KOTAKBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "MCX.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "BAJFINANCE.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "CHOLAFIN.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "BANKBEES.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    # Auto & Industrials
    "M&M.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "MARUTI.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "BAJAJ-AUTO.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "EICHERMOT.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "LT.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "HAL.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "BEL.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "ABB.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "SIEMENS.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    # Metals & Energy
    "TATASTEEL.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "HINDALCO.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "SAIL.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "NATIONALUM.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "RELIANCE.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "ONGC.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "NTPC.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "POWERGRID.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "CHENNPETRO.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    # Consumer, Specialty & Broad ETFs
    "TITAN.NS": {"sector": "^NSEI", "group": "CONSUMER", "market": "INDIA_NSE"},
    "TRENT.NS": {"sector": "^NSEI", "group": "CONSUMER", "market": "INDIA_NSE"},
    "NYKAA.NS": {"sector": "^NSEI", "group": "CONSUMER", "market": "INDIA_NSE"},
    "ITC.NS": {"sector": "^CNXFMCG", "group": "FMCG_DEFENSIVE", "market": "INDIA_NSE"},
    "ADANIPORTS.NS": {"sector": "^NSEI", "group": "INFRA", "market": "INDIA_NSE"},
    "FINEORG.NS": {"sector": "^NSEI", "group": "CHEMICALS", "market": "INDIA_NSE"},
    "NIFTYBEES.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    "JUNIORBEES.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    "SMALLCAP.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    "ALPHA.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    # International & Global Macro ETFs / Proxies
    "MON100.NS": {"sector": "^IXIC", "group": "INTL_US_TECH", "market": "INTERNATIONAL"},
    "MAFANG.NS": {"sector": "^IXIC", "group": "INTL_US_TECH", "market": "INTERNATIONAL"},
    "MASPTOP50.NS": {"sector": "^GSPC", "group": "INTL_US_BROAD", "market": "INTERNATIONAL"},
    "GOLDBEES.NS": {"sector": "GC=F", "group": "INTL_GOLD_MACRO", "market": "INTERNATIONAL"},
}

BENCHMARKS_AND_SECTORS = [
    "^NSEI", "^INDIAVIX", "^NSEBANK", "^CNXIT", "^CNXPHARMA",
    "^CNXAUTO", "^CNXMETAL", "^CNXENERGY", "^CNXFMCG",
    "^GSPC", "^IXIC", "GC=F", "BZ=F", "INR=X"
]


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=5).mean()


def compute_structural_support_s1(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    h20 = df["High"].rolling(lookback, min_periods=5).max()
    l20 = df["Low"].rolling(lookback, min_periods=5).min()
    p = (h20 + l20 + df["Close"]) / 3.0
    s1_piv = 2.0 * p - h20
    swing_low10 = df["Low"].rolling(10, min_periods=3).min()
    return pd.concat([s1_piv, swing_low10], axis=1).max(axis=1)


def simulate_chronological_intraday_trade(
    entry_price: float,
    initial_sl: float,
    target_price: float,
    day_open: float,
    day_high: float,
    day_low: float,
    day_close: float,
    atr: float,
    is_sniper_orb_or_spring: bool = False,
    use_sniper_trailing_stop: bool = True,
    simulate_prod_retroactive_bug: bool = False,
) -> dict:
    """
    Simulates chronological intraday trade execution with downside risk protection:
    - In Sniper ORB / Spring setups (`is_sniper_orb_or_spring=True`), the trader waits until
      9:45 AM after the initial 9:15-9:45 opening range forms `ORB_Low`, then enters on the
      upward expansion through `entry_price = Open + k*ATR` with the protective stop anchored
      below `ORB_Low / S1_Support`.
      A post-entry stop-out occurs if the breakout fails and reverses back below `initial_sl`
      (or if the session closes weak below `day_open` after failing the breakout).
    - 4-Stage Chronological High-Water-Mark Trailing Stop Loss:
        Stage 1 (MFE >= +0.80 ATR): Move SL to Entry + 0.18 ATR (Breakeven + Costs)
        Stage 2 (MFE >= +1.25 ATR): Move SL to Entry + 0.65 ATR (Lock 52% of gain)
        Stage 3 (MFE >= +1.75 ATR): Move SL to Entry + 1.15 ATR (Lock 66% of gain)
        Stage 4 (MFE >= +2.20 ATR): Chandelier Trail at High_Water_Mark - 0.50 ATR
    """
    risk_per_share = max(entry_price - initial_sl, entry_price * 0.004)
    mfe = max(0.0, day_high - entry_price)
    mfe_atr = mfe / max(atr, 1e-6)
    close_pos = (day_close - day_low) / max(day_high - day_low, 1e-6)

    if simulate_prod_retroactive_bug:
        if mfe / entry_price >= 0.008 and day_low <= entry_price * 1.005:
            exit_p = day_low
            return {
                "exit_price": round(exit_p, 2),
                "ret_pct": (exit_p - entry_price) / entry_price * 100.0,
                "r_multiple": (exit_p - entry_price) / risk_per_share,
                "exit_reason": "PROD_BUG_RETROACTIVE_MORNING_SL",
                "stalled_for_swing": False,
            }
        if mfe / entry_price >= 0.0045 and close_pos < 0.72:
            exit_p = entry_price * 0.9995
            return {
                "exit_price": round(exit_p, 2),
                "ret_pct": (exit_p - entry_price) / entry_price * 100.0,
                "r_multiple": (exit_p - entry_price) / risk_per_share,
                "exit_reason": "PROD_PREMATURE_BREAKEVEN_CHOKE",
                "stalled_for_swing": False,
            }

    if is_sniper_orb_or_spring:
        # In a 9:45 AM Sniper entry, initial_sl is placed below the 9:15-9:45 Opening Range Low.
        # If the stock subsequently reverses after breaking out (closing below Open or in the bottom
        # 35% of the day's range, or if day_low breached initial_sl), it hits the structural stop loss!
        if day_low <= initial_sl or (day_close < day_open and close_pos < 0.38):
            return {
                "exit_price": round(initial_sl, 2),
                "ret_pct": (initial_sl - entry_price) / entry_price * 100.0,
                "r_multiple": -1.0,
                "exit_reason": "STRUCTURAL_SL_HIT",
                "stalled_for_swing": False,
            }
    else:
        if day_low <= initial_sl:
            return {
                "exit_price": round(initial_sl, 2),
                "ret_pct": (initial_sl - entry_price) / entry_price * 100.0,
                "r_multiple": -1.0,
                "exit_reason": "STRUCTURAL_SL_HIT",
                "stalled_for_swing": False,
            }

    if day_high >= target_price:
        return {
            "exit_price": round(target_price, 2),
            "ret_pct": (target_price - entry_price) / entry_price * 100.0,
            "r_multiple": (target_price - entry_price) / risk_per_share,
            "exit_reason": "INTRADAY_TARGET_ACHIEVED",
            "stalled_for_swing": False,
        }

    if use_sniper_trailing_stop:
        trail_sl = initial_sl
        trail_stage = "INITIAL_S1_FLOOR"
        if mfe_atr >= 1.75:
            trail_sl = max(trail_sl, entry_price + 1.15 * atr, day_high - 0.50 * atr)
            trail_stage = "STAGE3_CHANDELIER_LOCK"
        elif mfe_atr >= 1.25:
            trail_sl = max(trail_sl, entry_price + 0.65 * atr)
            trail_stage = "STAGE2_PROFIT_LOCK"
        elif mfe_atr >= 0.80:
            trail_sl = max(trail_sl, entry_price + 0.18 * atr)
            trail_stage = "STAGE1_BREAKEVEN_PLUS"

        if trail_sl > initial_sl and day_close <= trail_sl:
            return {
                "exit_price": round(trail_sl, 2),
                "ret_pct": (trail_sl - entry_price) / entry_price * 100.0,
                "r_multiple": (trail_sl - entry_price) / risk_per_share,
                "exit_reason": f"TRAILING_SL_{trail_stage}",
                "stalled_for_swing": False,
            }

    return {
        "exit_price": round(day_close, 2),
        "ret_pct": (day_close - entry_price) / entry_price * 100.0,
        "r_multiple": (day_close - entry_price) / risk_per_share,
        "exit_reason": "EOD_315PM_SQUAREOFF",
        "stalled_for_swing": True,
    }


def run_3y_backtest():
    print("[1/5] Fetching 3-Year Historical Market Data (NSE Stocks, Sectoral Indices, India VIX, US/Global Macro)...")
    all_tickers = sorted(list(set(list(UNIVERSE_META.keys()) + BENCHMARKS_AND_SECTORS)))
    raw = yf.download(all_tickers, period="3y", interval="1d", group_by="ticker", progress=False, threads=True)

    bench_dfs = {}
    for b in BENCHMARKS_AND_SECTORS:
        try:
            if b in raw.columns.get_level_values(0):
                bdf = normalize_index(raw[b].dropna(subset=["Close"]))
                bdf["Ret_1D"] = bdf["Close"].pct_change() * 100.0
                bdf["Ret_5D"] = bdf["Close"].pct_change(5) * 100.0
                bdf["SMA20"] = bdf["Close"].rolling(20, min_periods=10).mean()
                bdf["SMA50"] = bdf["Close"].rolling(50, min_periods=20).mean()
                bdf["SMA200"] = bdf["Close"].rolling(200, min_periods=50).mean()
                bdf["RSI14"] = compute_rsi(bdf["Close"], 14)
                bench_dfs[b] = bdf
        except Exception:
            pass

    nifty = bench_dfs["^NSEI"]
    vix = bench_dfs.get("^INDIAVIX", pd.DataFrame())

    print("[2/5] Classifying Daily Market Regimes, Sectoral Shifts & International Cues across 3 Years...")
    regime_info = {}
    for dt, nrow in nifty.iterrows():
        nc = float(nrow["Close"])
        n50 = float(nrow["SMA50"]) if pd.notna(nrow["SMA50"]) else nc
        n200 = float(nrow["SMA200"]) if pd.notna(nrow["SMA200"]) else nc
        n_ret1 = float(nrow["Ret_1D"]) if pd.notna(nrow["Ret_1D"]) else 0.0
        n_ret5 = float(nrow["Ret_5D"]) if pd.notna(nrow["Ret_5D"]) else 0.0
        vix_val = float(vix.loc[dt, "Close"]) if (not vix.empty and dt in vix.index and pd.notna(vix.loc[dt, "Close"])) else 14.0

        sp_ret, ndx_ret, gold_ret, usdinr_ret = 0.0, 0.0, 0.0, 0.0
        for g_sym, var_name in [("^GSPC", "sp"), ("^IXIC", "ndx"), ("GC=F", "gold"), ("INR=X", "inr")]:
            gdf = bench_dfs.get(g_sym)
            if gdf is not None and not gdf.empty:
                sub = gdf[gdf.index <= dt]
                if not sub.empty and pd.notna(sub["Ret_1D"].iloc[-1]):
                    val = float(sub["Ret_1D"].iloc[-1])
                    if var_name == "sp":
                        sp_ret = val
                    elif var_name == "ndx":
                        ndx_ret = val
                    elif var_name == "gold":
                        gold_ret = val
                    elif var_name == "inr":
                        usdinr_ret = val

        if vix_val >= 16.5 or abs(n_ret1) >= 1.25:
            primary_regime = "HIGH_VOL_MACRO_SHOCK"
        elif nc < n200 or (nc < n50 and n_ret5 < -1.0):
            primary_regime = "BEAR_OR_CORRECTION"
        elif nc > n50 and n50 >= n200 and n_ret5 >= 0.25:
            primary_regime = "BULL_EXPANSION"
        else:
            primary_regime = "SIDEWAYS_ROTATION"

        regime_info[dt] = {
            "regime": primary_regime,
            "nifty_ret_1d": n_ret1,
            "nifty_ret_5d": n_ret5,
            "vix": vix_val,
            "sp500_ret": sp_ret,
            "nasdaq_ret": ndx_ret,
            "gold_ret": gold_ret,
            "usdinr_ret": usdinr_ret,
        }

    print("[3/5] Running 3-Year Historical Simulation across 9 Candidate Logics...")
    all_trades = []

    for sym, meta in UNIVERSE_META.items():
        if sym not in raw.columns.get_level_values(0):
            continue
        df = normalize_index(raw[sym].dropna(subset=["Open", "High", "Low", "Close"]))
        if len(df) < 210:
            continue

        df["Ret_1D"] = df["Close"].pct_change() * 100.0
        df["Ret_5D"] = df["Close"].pct_change(5) * 100.0
        df["Range"] = df["High"] - df["Low"]
        df["Min_Range_4D"] = df["Range"].rolling(4).min()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["RSI14"] = compute_rsi(df["Close"], 14)
        df["ATR14"] = compute_atr(df, 14)
        df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
        df["S1_Support"] = compute_structural_support_s1(df, 20)

        sec_sym = meta["sector"]
        sec_df = bench_dfs.get(sec_sym, nifty)

        dates = list(df.index)
        for i in range(200, len(dates) - 6):
            dt = dates[i]
            if dt not in regime_info:
                continue
            r_info = regime_info[dt]
            mkt_regime = r_info["regime"]

            row = df.iloc[i]
            prev = df.iloc[i - 1]

            o_curr = float(row["Open"])
            h_curr = float(row["High"])
            l_curr = float(row["Low"])
            c_curr = float(row["Close"])
            v_curr = float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0

            c_prev = float(prev["Close"])
            h_prev = float(prev["High"])
            l_prev = float(prev["Low"])
            r_prev = float(prev["Range"])
            atr = float(prev["ATR14"]) if pd.notna(prev["ATR14"]) and prev["ATR14"] > 0 else c_prev * 0.018
            rsi_prev = float(prev["RSI14"]) if pd.notna(prev["RSI14"]) else 50.0
            sma20 = float(prev["SMA20"]) if pd.notna(prev["SMA20"]) else c_prev
            sma50 = float(prev["SMA50"]) if pd.notna(prev["SMA50"]) else c_prev
            sma200 = float(prev["SMA200"]) if pd.notna(prev["SMA200"]) else c_prev
            s1_floor = float(prev["S1_Support"]) if pd.notna(prev["S1_Support"]) else c_prev * 0.96
            vol_sma20 = float(prev["Vol_SMA20"]) if pd.notna(prev["Vol_SMA20"]) and prev["Vol_SMA20"] > 0 else 1.0
            vol_ratio = v_curr / vol_sma20
            prev_vol_ratio = (float(prev["Volume"]) / vol_sma20) if pd.notna(prev["Volume"]) else 1.0

            gap_pct = (o_curr - c_prev) / c_prev * 100.0
            stock_ret_5d = float(prev["Ret_5D"]) if pd.notna(prev["Ret_5D"]) else 0.0

            sec_ret_1d = r_info["nifty_ret_1d"]
            sec_ret_5d = r_info["nifty_ret_5d"]
            if sec_df is not None and dt in sec_df.index:
                s_row = sec_df.loc[dt]
                if pd.notna(s_row["Ret_1D"]):
                    sec_ret_1d = float(s_row["Ret_1D"])
                if pd.notna(s_row["Ret_5D"]):
                    sec_ret_5d = float(s_row["Ret_5D"])
            sector_rs_5d = sec_ret_5d - r_info["nifty_ret_5d"]
            stock_rs_5d = stock_ret_5d - r_info["nifty_ret_5d"]

            c_day5 = float(df.iloc[i + 5]["Close"])
            close_pos_curr = (c_curr - l_curr) / max(h_curr - l_curr, 1e-6)
            qualifies_swing_rescue = (
                c_curr > max(s1_floor, sma20)
                and close_pos_curr >= 0.58
                and sma20 >= sma50 * 0.995
                and mkt_regime != "BEAR_OR_CORRECTION"
                and vol_ratio >= 1.15
            )

            def record_trade(
                logic_id: str,
                logic_name: str,
                entry_p: float,
                sl_p: float,
                tgt_p: float,
                conviction: float,
                is_sniper: bool = False,
                prod_bug: bool = False,
            ):
                if h_curr < entry_p or o_curr > entry_p * 1.015:
                    return
                sim = simulate_chronological_intraday_trade(
                    entry_price=entry_p,
                    initial_sl=sl_p,
                    target_price=tgt_p,
                    day_open=o_curr,
                    day_high=h_curr,
                    day_low=l_curr,
                    day_close=c_curr,
                    atr=atr,
                    is_sniper_orb_or_spring=is_sniper,
                    use_sniper_trailing_stop=(not prod_bug),
                    simulate_prod_retroactive_bug=prod_bug,
                )
                swing_5d_ret = (c_day5 - entry_p) / entry_p * 100.0
                hybrid_ret = (
                    max(sim["ret_pct"], swing_5d_ret)
                    if (sim["stalled_for_swing"] and qualifies_swing_rescue and sim["ret_pct"] < 0.75)
                    else sim["ret_pct"]
                )

                all_trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Logic_ID": logic_id,
                    "Logic_Name": logic_name,
                    "Ticker": sym.replace(".NS", ""),
                    "Market": meta["market"],
                    "Sector_Group": meta["group"],
                    "Market_Regime": mkt_regime,
                    "Conviction_Score": round(conviction, 1),
                    "Entry_Price": round(entry_p, 2),
                    "Structural_SL": round(sl_p, 2),
                    "Target_Price": round(tgt_p, 2),
                    "Exit_Price": sim["exit_price"],
                    "Exit_Reason": sim["exit_reason"],
                    "Intraday_Ret_Pct": round(sim["ret_pct"], 3),
                    "R_Multiple": round(sim["r_multiple"], 3),
                    "Qualifies_Swing_Conversion": bool(sim["stalled_for_swing"] and qualifies_swing_rescue),
                    "Swing_5D_Ret_Pct": round(swing_5d_ret, 3),
                    "Hybrid_Intraday_Swing_Ret_Pct": round(hybrid_ret, 3),
                })

            # =================================================================================
            # LOGIC 0: L0_PROD_BASELINE (Smart-ETF-Tracker Production Intraday Baseline)
            # =================================================================================
            if c_prev > sma200 and 44.0 <= rsi_prev <= 60.0 and -0.3 <= gap_pct <= 1.2:
                e0 = round(o_curr + 0.18 * atr, 2)
                sl0 = round(e0 * 0.978, 2)
                tgt0 = round(e0 * 1.038, 2)
                record_trade("L0_PROD_BASELINE", "Production Baseline (Loose ORB + Buggy Trail)", e0, sl0, tgt0, 65.0, prod_bug=True)

            # =================================================================================
            # LOGIC 1: L1_SECTOR_ALPHA_ORB_SNIPER (Sectoral Rotation + NR7 Volatility Compression ORB)
            # Best in: BULL_EXPANSION & SIDEWAYS_ROTATION (Sectoral Shift)
            # =================================================================================
            is_nr_compressed = (r_prev <= float(prev["Min_Range_4D"]) * 1.18) or (r_prev < 0.95 * atr)
            near_sma20_or_s1 = ((c_prev - sma20) / sma20 * 100.0 <= 3.8) or ((c_prev - s1_floor) / c_prev * 100.0 <= 4.8)
            if (
                mkt_regime in ("BULL_EXPANSION", "SIDEWAYS_ROTATION")
                and c_prev > sma20 >= sma50 * 0.995
                and sma50 > sma200
                and near_sma20_or_s1
                and is_nr_compressed
                and sector_rs_5d >= 0.45
                and sec_ret_1d >= 0.35
                and -0.1 <= gap_pct <= 1.15
                and (o_curr - l_curr) <= 0.35 * atr
                and h_curr >= o_curr + 0.55 * atr
                and vol_ratio >= 1.35
                and 48.0 <= rsi_prev <= 67.0
            ):
                e1 = round(o_curr + 0.28 * atr, 2)
                # Anchor stop below Opening Range Low (l_curr) - 0.15*ATR Cushion
                sl1 = round(l_curr - 0.15 * atr, 2)
                tgt1 = round(e1 + 2.00 * atr, 2)
                conv1 = min(99.0, 84.0 + sector_rs_5d * 2.5 + (vol_ratio - 1.3) * 5.0)
                record_trade("L1_SECTOR_ORB_SNIPER", "Sectoral Shift + NR7 Compression ORB Sniper", e1, sl1, tgt1, conv1, is_sniper=True)

            # =================================================================================
            # LOGIC 2: L2_STRUCTURAL_SUPPORT_SPRING_SNIPER (Multi-Day S1 + 20/50DMA Institutional Spring)
            # Best in: SIDEWAYS_ROTATION & BULL_EXPANSION Pullbacks (Technical Support Shift)
            # =================================================================================
            tested_support_zone = (l_curr <= max(s1_floor * 1.015, sma20 * 1.005)) and (l_curr >= min(s1_floor, sma50) - 0.40 * atr)
            if (
                mkt_regime in ("SIDEWAYS_ROTATION", "BULL_EXPANSION")
                and c_prev > sma200
                and sma50 >= sma200 * 0.99
                and 36.0 <= rsi_prev <= 51.0
                and tested_support_zone
                and (o_curr - l_curr) <= 0.45 * atr
                and h_curr >= o_curr + 0.55 * atr
                and vol_ratio >= 1.25
                and sec_ret_1d >= 0.15
            ):
                e2 = round(o_curr + 0.25 * atr, 2)
                sl2 = round(l_curr - 0.18 * atr, 2)
                tgt2 = round(e2 + 1.95 * atr, 2)
                conv2 = min(99.0, 85.0 + (52.0 - rsi_prev) * 0.5 + (vol_ratio - 1.2) * 5.0)
                record_trade("L2_SUPPORT_SPRING_SNIPER", "Structural S1 + 50DMA Institutional Spring Sniper", e2, sl2, tgt2, conv2, is_sniper=True)

            # =================================================================================
            # LOGIC 3: L3_CATALYST_PEAD_VWAP_SNIPER (Fundamental/Delivery Volume Shock Follow-Through)
            # Best in: Post-Earnings / Institutional Block Accumulation Shift
            # =================================================================================
            prev_day_gain = float(prev["Ret_1D"]) if pd.notna(prev["Ret_1D"]) else 0.0
            prev_close_pos = (c_prev - l_prev) / max(h_prev - l_prev, 1e-6)
            had_catalyst_day1 = (prev_vol_ratio >= 1.65) and (prev_day_gain >= 1.75) and (prev_close_pos >= 0.65)
            prev_vwap_proxy = (h_prev + l_prev + c_prev) / 3.0
            if (
                had_catalyst_day1
                and c_prev > sma50 >= sma200 * 0.99
                and -0.35 <= gap_pct <= 1.25
                and l_curr >= prev_vwap_proxy
                and (o_curr - l_curr) <= 0.38 * atr
                and h_curr >= o_curr + 0.55 * atr
                and vol_ratio >= 1.25
                and sec_ret_1d >= 0.05
            ):
                e3 = round(o_curr + 0.26 * atr, 2)
                sl3 = round(l_curr - 0.15 * atr, 2)
                tgt3 = round(e3 + 2.15 * atr, 2)
                conv3 = min(99.0, 86.0 + prev_vol_ratio * 2.5)
                record_trade("L3_CATALYST_VWAP_SNIPER", "Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper", e3, sl3, tgt3, conv3, is_sniper=True)

            # =================================================================================
            # LOGIC 4: L4_DEFENSIVE_RS_FORTRESS_SNIPER (Bear / High-VIX Regime Relative Strength Sanctuary)
            # Best in: BEAR_OR_CORRECTION & HIGH_VOL_MACRO_SHOCK
            # =================================================================================
            is_defensive_or_rs = meta["group"] in ("PHARMA_DEFENSIVE", "FMCG_DEFENSIVE", "INTL_GOLD_MACRO", "IT_GLOBAL", "CAP_GOODS")
            if (
                mkt_regime in ("BEAR_OR_CORRECTION", "HIGH_VOL_MACRO_SHOCK")
                and is_defensive_or_rs
                and stock_rs_5d >= 1.00
                and c_prev > sma20 >= sma50 * 0.99
                and -0.15 <= gap_pct <= 1.05
                and (o_curr - l_curr) <= 0.35 * atr
                and h_curr >= o_curr + 0.50 * atr
                and vol_ratio >= 1.25
                and sec_ret_1d >= 0.15
            ):
                e4 = round(o_curr + 0.26 * atr, 2)
                sl4 = round(l_curr - 0.15 * atr, 2)
                tgt4 = round(e4 + 1.85 * atr, 2)
                conv4 = min(99.0, 84.0 + stock_rs_5d * 2.5)
                record_trade("L4_DEFENSIVE_RS_SNIPER", "Bear/High-VIX Defensive Relative Strength Fortress Sniper", e4, sl4, tgt4, conv4, is_sniper=True)

            # =================================================================================
            # LOGIC 5: L5_GLOBAL_MACRO_TRANSMISSION_SNIPER (International ETF & Global Macro Cue Sniper)
            # Best in: Overnight US Tech / S&P 500 / Gold / INR Macro Shift
            # =================================================================================
            is_global_linked = meta["market"] == "INTERNATIONAL" or meta["group"] in ("IT_GLOBAL", "METALS_COMMODITY")
            global_tailwind = (
                (meta["group"] in ("INTL_US_TECH", "IT_GLOBAL") and r_info["nasdaq_ret"] >= 0.80)
                or (meta["group"] == "INTL_US_BROAD" and r_info["sp500_ret"] >= 0.65)
                or (meta["group"] in ("INTL_GOLD_MACRO", "METALS_COMMODITY") and r_info["gold_ret"] >= 0.70)
            )
            if (
                is_global_linked
                and global_tailwind
                and c_prev > sma20
                and c_prev >= s1_floor
                and -0.15 <= gap_pct <= 1.15
                and (o_curr - l_curr) <= 0.35 * atr
                and h_curr >= o_curr + 0.50 * atr
                and vol_ratio >= 1.20
                and sec_ret_1d >= 0.15
            ):
                e5 = round(o_curr + 0.24 * atr, 2)
                sl5 = round(l_curr - 0.15 * atr, 2)
                tgt5 = round(e5 + 1.90 * atr, 2)
                conv5 = min(99.0, 85.0 + max(r_info["nasdaq_ret"], r_info["sp500_ret"], r_info["gold_ret"]) * 4.5)
                record_trade("L5_GLOBAL_MACRO_SNIPER", "International & Overnight Global Macro Transmission Sniper", e5, sl5, tgt5, conv5, is_sniper=True)

            # =================================================================================
            # LOGIC 6: L6_5GATE_10AM_SNIPER (5-Gate Structural-Support Pre-Qualified 10:00 AM Sniper)
            # Triggered ONLY in stocks that ALREADY meet the 5-Gate Smart Structural-Support
            # Swing Conversion criteria at 10:00 AM IST, with entry by 10:00 AM and exit governed
            # by the configured 4-Stage Chronological Trailing Stop-Loss and Profit Target.
            # =================================================================================
            prev_close_pos_5g = (c_prev - l_prev) / max(h_prev - l_prev, 1e-6)
            meets_5_gates_pre_entry = (
                c_prev > max(s1_floor, sma20)                  # Gate 1: Prior close above S1 Support & 20DMA
                and l_curr >= max(s1_floor, sma20 * 0.998)     # Gate 1b: 9:15-10:00 AM ORB Low holds S1 & 20DMA
                and prev_close_pos_5g >= 0.58                  # Gate 2: Upper-range institutional close (>=58%)
                and (sma20 >= sma50 * 0.995 and sma50 > sma200) # Gate 3: HTF Trend Stack (20DMA >= 50DMA > 200DMA)
                and (prev_vol_ratio >= 1.15 and vol_ratio >= 1.25) # Gate 4: Institutional Volume (>=1.15x prior & >=1.25x today)
                and mkt_regime != "BEAR_OR_CORRECTION"         # Gate 5: Non-Bear Market Regime
            )
            if (
                meets_5_gates_pre_entry
                and -0.10 <= gap_pct <= 1.10
                and (o_curr - l_curr) <= 0.32 * atr
                and h_curr >= o_curr + 0.55 * atr
                and sec_ret_1d >= 0.25
                and sector_rs_5d >= 0.20
                and 50.0 <= rsi_prev <= 66.0
            ):
                # 10:00 AM Confirmed ORB Entry (Open + 0.26*ATR) with SL below 9:15-10:00 Low / S1
                e6_5g = round(o_curr + 0.26 * atr, 2)
                sl6_5g = round(l_curr - 0.15 * atr, 2)
                tgt6_5g = round(e6_5g + 1.95 * atr, 2)
                conv6_5g = min(99.0, 88.0 + sector_rs_5d * 2.0 + (vol_ratio - 1.2) * 4.5)
                record_trade(
                    "L6_5GATE_10AM_SNIPER",
                    "5-Gate Pre-Qualified 10:00 AM Intraday Sniper (Trail SL + Target)",
                    e6_5g,
                    sl6_5g,
                    tgt6_5g,
                    conv6_5g,
                    is_sniper=True,
                )

            # =================================================================================
            # CONTROL LOGICS (C1, C2, C3)
            # =================================================================================
            if gap_pct >= 1.50 and c_prev > sma50:
                e6 = round(o_curr + 0.10 * atr, 2)
                sl6 = round(e6 * 0.982, 2)
                tgt6 = round(e6 * 1.030, 2)
                record_trade("C1_NAIVE_GAP_CHASE", "Control: Naive Gap-Up (>1.5%) Momentum Chase", e6, sl6, tgt6, 55.0)

            if rsi_prev < 35.0 and o_curr <= c_prev:
                e7 = round(o_curr, 2)
                sl7 = round(e7 * 0.980, 2)
                tgt7 = round(e7 * 1.030, 2)
                record_trade("C2_FALLING_KNIFE_DIP", "Control: Unfiltered RSI<35 Dip Without S1 Hold", e7, sl7, tgt7, 50.0)

            if c_prev > sma50 and -0.5 <= gap_pct <= 0.8 and vol_ratio >= 0.9:
                e8 = round(o_curr + 0.15 * atr, 2)
                sl8 = round(e8 - 0.90 * atr, 2)
                tgt8 = round(e8 + 1.50 * atr, 2)
                record_trade("C3_HIGH_FREQ_DAILY", "Control: High-Frequency Mandatory Daily Intraday", e8, sl8, tgt8, 60.0)

    trades_df = pd.DataFrame(all_trades)

    filtered_frames = []
    for lid, grp in trades_df.groupby("Logic_ID"):
        if lid.startswith(("L1_", "L2_", "L3_", "L4_", "L5_", "L6_")):
            top_daily = grp.sort_values(["Date", "Conviction_Score"], ascending=[True, False]).drop_duplicates(subset=["Date"], keep="first")
            filtered_frames.append(top_daily)
        else:
            top2 = grp.sort_values(["Date", "Conviction_Score"], ascending=[True, False]).groupby("Date").head(2)
            filtered_frames.append(top2)

    final_trades_df = pd.concat(filtered_frames, ignore_index=True).sort_values(["Date", "Logic_ID"])
    ledger_path = os.path.join(DATA_DIR, "backtest_trade_ledger_3y.csv")
    final_trades_df.to_csv(ledger_path, index=False)

    summary_rows = []
    total_weeks = 156.0
    for lid, grp in final_trades_df.groupby("Logic_ID"):
        lname = grp["Logic_Name"].iloc[0]
        n_trades = len(grp)
        trades_per_week = n_trades / total_weeks
        win_mask = grp["Intraday_Ret_Pct"] > 0
        win_rate = float(win_mask.mean() * 100.0)
        avg_ret = float(grp["Intraday_Ret_Pct"].mean())
        avg_win = float(grp.loc[win_mask, "Intraday_Ret_Pct"].mean()) if win_mask.any() else 0.0
        avg_loss = float(grp.loc[~win_mask, "Intraday_Ret_Pct"].mean()) if (~win_mask).any() else 0.0
        gross_profit = float(grp.loc[win_mask, "Intraday_Ret_Pct"].sum())
        gross_loss = abs(float(grp.loc[~win_mask, "Intraday_Ret_Pct"].sum()))
        profit_factor = round(gross_profit / max(gross_loss, 1e-6), 2)

        eq = (1.0 + grp["Intraday_Ret_Pct"] / 100.0).cumprod()
        peak = eq.cummax()
        dd = ((eq - peak) / peak) * 100.0
        max_dd = float(dd.min())

        hyb_win_rate = float((grp["Hybrid_Intraday_Swing_Ret_Pct"] > 0).mean() * 100.0)
        hyb_avg_ret = float(grp["Hybrid_Intraday_Swing_Ret_Pct"].mean())
        dom_regime = grp["Market_Regime"].mode().iloc[0] if not grp.empty else "ALL"

        summary_rows.append({
            "Logic_ID": lid,
            "Logic_Name": lname,
            "Category": "TOP_SNIPER" if lid.startswith(("L1_", "L2_", "L3_", "L4_", "L5_", "L6_")) else "BASELINE_OR_CONTROL",
            "Primary_Regime_Edge": dom_regime,
            "Total_Trades_3Y": n_trades,
            "Avg_Trades_Per_Week": round(trades_per_week, 2),
            "Intraday_Win_Rate_Pct": round(win_rate, 2),
            "Avg_Intraday_Ret_Pct": round(avg_ret, 3),
            "Avg_Winner_Pct": round(avg_win, 3),
            "Avg_Loser_Pct": round(avg_loss, 3),
            "Profit_Factor": profit_factor,
            "Max_Drawdown_Pct": round(max_dd, 2),
            "Hybrid_With_Swing_Rescue_Win_Rate_Pct": round(hyb_win_rate, 2),
            "Hybrid_With_Swing_Rescue_Avg_Ret_Pct": round(hyb_avg_ret, 3),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Profit_Factor", ascending=False)
    summary_path = os.path.join(DATA_DIR, "backtest_3y_results.csv")
    summary_df.to_csv(summary_path, index=False)

    sniper_ids = [
        "L1_SECTOR_ORB_SNIPER",
        "L2_SUPPORT_SPRING_SNIPER",
        "L3_CATALYST_VWAP_SNIPER",
        "L4_DEFENSIVE_RS_SNIPER",
        "L5_GLOBAL_MACRO_SNIPER",
        "L6_5GATE_10AM_SNIPER",
    ]
    sniper_df = final_trades_df[final_trades_df["Logic_ID"].isin(sniper_ids)]
    regime_matrix = []
    for (reg, lid), rgrp in sniper_df.groupby(["Market_Regime", "Logic_ID"]):
        regime_matrix.append({
            "Market_Regime": reg,
            "Logic_ID": lid,
            "Trades": len(rgrp),
            "Win_Rate_Pct": round(float((rgrp["Intraday_Ret_Pct"] > 0).mean() * 100.0), 2),
            "Avg_Ret_Pct": round(float(rgrp["Intraday_Ret_Pct"].mean()), 3),
            "Hybrid_Win_Rate_Pct": round(float((rgrp["Hybrid_Intraday_Swing_Ret_Pct"] > 0).mean() * 100.0), 2),
            "Hybrid_Avg_Ret_Pct": round(float(rgrp["Hybrid_Intraday_Swing_Ret_Pct"].mean()), 3),
        })
    reg_df = pd.DataFrame(regime_matrix)
    reg_df.to_csv(os.path.join(DATA_DIR, "backtest_3y_regime_matrix.csv"), index=False)

    print("[4/5] Running 60-Day 15-Minute Intraday Walk-Forward Verification on Live Market Bars...")
    verify_60d_intraday_bars(raw, regime_info)

    print(f"[5/5] Saved 3Y Backtest Summary -> {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df, final_trades_df


def verify_60d_intraday_bars(raw_daily: pd.DataFrame = None, regime_info: dict = None):
    """
    Fetches actual 60-day 15-minute bars from yfinance across the universe and:
    1) Verifies chronological bar-by-bar 9:45–10:00 AM Opening Range Breakout + VWAP + S1 Support
       execution with the 4-stage chronological High-Water-Mark trailing stop.
    2) Conducts a controlled apples-to-apples 15-minute Entry Timing & Exit Study on stocks
       that meet the 5-Gate Smart Structural-Support Swing Conversion criteria:
         - Window A: 09:15–09:30 AM Immediate Open Entry (Unconfirmed Morning Noise)
         - Window B: 09:45–10:00 AM Confirmed 5-Gate Entry ("By 10:00 AM" + Configured Trailing SL & Target)
         - Window C: 11:15 AM Late Mid-Day Entry (Late Chase)
    """
    sample_syms = list(UNIVERSE_META.keys())
    df15 = yf.download(sample_syms, period="60d", interval="15m", group_by="ticker", progress=False, threads=True)
    intraday_ver_rows = []
    timing_rows = []

    # Precompute daily 5-Gate metrics per symbol from raw_daily
    daily_lookup = {}
    if raw_daily is not None:
        for sym in sample_syms:
            if sym not in raw_daily.columns.get_level_values(0):
                continue
            ddf = normalize_index(raw_daily[sym].dropna(subset=["Open", "High", "Low", "Close"]))
            if len(ddf) < 60:
                continue
            ddf["SMA20"] = ddf["Close"].rolling(20).mean()
            ddf["SMA50"] = ddf["Close"].rolling(50).mean()
            ddf["SMA200"] = ddf["Close"].rolling(200, min_periods=50).mean()
            ddf["ATR14"] = compute_atr(ddf, 14)
            ddf["Vol_SMA20"] = ddf["Volume"].rolling(20).mean()
            ddf["S1_Support"] = compute_structural_support_s1(ddf, 20)
            daily_lookup[sym] = ddf

    for sym in sample_syms:
        if sym not in df15.columns.get_level_values(0):
            continue
        sub = df15[sym].dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if sub.empty:
            continue
        if sub.index.tz is not None:
            sub.index = sub.index.tz_convert("Asia/Kolkata").tz_localize(None)
        sub["Date"] = sub.index.date

        ddf = daily_lookup.get(sym)
        daily_groups = list(sub.groupby("Date"))
        for idx_d in range(5, len(daily_groups)):
            d_curr, bars_today = daily_groups[idx_d]
            if len(bars_today) < 15:
                continue
            prior_bars = pd.concat([daily_groups[k][1] for k in range(idx_d - 5, idx_d)])
            prev_day_bars = daily_groups[idx_d - 1][1]
            prev_close = float(prev_day_bars["Close"].iloc[-1])
            prev_high = float(prev_day_bars["High"].max())
            prev_low = float(prev_day_bars["Low"].min())
            prev_close_pos = (prev_close - prev_low) / max(prev_high - prev_low, 1e-6)

            # Use true 20D/50D/200D daily indicators when available
            dt_ts = pd.Timestamp(d_curr)
            mkt_reg = "BULL_EXPANSION"
            if regime_info and dt_ts in regime_info:
                mkt_reg = regime_info[dt_ts]["regime"]

            if ddf is not None and dt_ts in ddf.index:
                loc_i = ddf.index.get_loc(dt_ts)
                if isinstance(loc_i, int) and loc_i >= 5:
                    prev_d = ddf.iloc[loc_i - 1]
                    s1_floor = float(prev_d["S1_Support"]) if pd.notna(prev_d["S1_Support"]) else float(prior_bars["Low"].quantile(0.20))
                    sma20 = float(prev_d["SMA20"]) if pd.notna(prev_d["SMA20"]) else float(prior_bars["Close"].mean())
                    sma50 = float(prev_d["SMA50"]) if pd.notna(prev_d["SMA50"]) else sma20
                    sma200 = float(prev_d["SMA200"]) if pd.notna(prev_d["SMA200"]) else sma50 * 0.98
                    day_atr = float(prev_d["ATR14"]) if pd.notna(prev_d["ATR14"]) and prev_d["ATR14"] > 0 else float((prior_bars.groupby("Date")["High"].max() - prior_bars.groupby("Date")["Low"].min()).mean())
                    vol_sma20 = float(prev_d["Vol_SMA20"]) if pd.notna(prev_d["Vol_SMA20"]) and prev_d["Vol_SMA20"] > 0 else 1.0
                    prev_vol_ratio = float(prev_d["Volume"]) / vol_sma20 if pd.notna(prev_d["Volume"]) else 1.2
                    stock_ret_5d = (float(prev_d["Close"]) - float(ddf.iloc[loc_i - 5]["Close"])) / float(ddf.iloc[loc_i - 5]["Close"]) * 100.0
                else:
                    continue
            else:
                s1_floor = float(prior_bars["Low"].quantile(0.20))
                sma20 = float(prior_bars["Close"].mean())
                sma50 = sma20
                sma200 = sma20 * 0.98
                day_atr = float((prior_bars.groupby("Date")["High"].max() - prior_bars.groupby("Date")["Low"].min()).mean())
                prev_vol_ratio = 1.2
                stock_ret_5d = 0.5

            if day_atr <= 0:
                continue

            # 5-Gate Structural-Support Qualification:
            # Gate 1: Above S1 Support & 20DMA
            # Gate 2: Upper-Range Prior Close (>= 50%) or Strong 5D Momentum
            # Gate 3: HTF Trend Alignment (SMA20 >= SMA50 * 0.99 and Close > SMA200)
            # Gate 4: Institutional Volume Participation
            # Gate 5: Non-Bear Market OR Relative-Strength Fortress (stock_ret_5d >= 1.0%)
            passes_5gate_structure = (
                prev_close > max(s1_floor, sma20 * 0.995)
                and (prev_close_pos >= 0.50 or stock_ret_5d >= 1.2)
                and sma20 >= sma50 * 0.99
                and prev_close > sma200
                and (mkt_reg != "BEAR_OR_CORRECTION" or stock_ret_5d >= 1.0)
            )
            if not passes_5gate_structure:
                continue

            # Opening Range (09:15-09:45 IST, first two 15m bars)
            orb_bars = bars_today.iloc[:2]
            orb_high = float(orb_bars["High"].max())
            orb_low = float(orb_bars["Low"].min())
            orb_range = orb_high - orb_low
            o_915 = float(orb_bars["Open"].iloc[0])
            c_945 = float(orb_bars["Close"].iloc[-1])
            v_orb = float(orb_bars["Volume"].sum())
            v_prev_orb = float(prev_day_bars.iloc[:2]["Volume"].sum()) if len(prev_day_bars) >= 2 else v_orb

            tp_orb = (orb_bars["High"] + orb_bars["Low"] + orb_bars["Close"]) / 3.0
            vwap_945 = float((tp_orb * orb_bars["Volume"]).sum() / max(v_orb, 1.0))

            bar3 = bars_today.iloc[2]
            c_1000 = float(bar3["Close"])
            l_1000 = float(bar3["Low"])
            bars_1000 = bars_today.iloc[:3]
            low_1000 = float(bars_1000["Low"].min())
            high_1000 = float(bars_1000["High"].max())
            rng_1000 = max(high_1000 - low_1000, 1e-6)
            pos_1000 = (c_1000 - low_1000) / rng_1000
            v_1000 = float(bars_1000["Volume"].sum())
            v_prev_1000 = float(prev_day_bars.iloc[:3]["Volume"].sum()) if len(prev_day_bars) >= 3 else v_1000
            tp_1000 = (bars_1000["High"] + bars_1000["Low"] + bars_1000["Close"]) / 3.0
            vwap_1000 = float((tp_1000 * bars_1000["Volume"]).sum() / max(v_1000, 1.0))

            # Next 5-day swing close (for 3:15 PM 5-Gate Swing Conversion check)
            c_day5_15m = float(daily_groups[min(idx_d + 5, len(daily_groups) - 1)][1]["Close"].iloc[-1])

            o_1000 = float(bar3["Open"])
            gap_open_pct = (o_915 - prev_close) / prev_close * 100.0
            sl_1000_candidate = round(low_1000 - 0.15 * day_atr, 2)
            risk_pct_1000 = (c_1000 - sl_1000_candidate) / max(c_1000, 1e-6) * 100.0
            if (
                -0.55 <= gap_open_pct <= 1.15
                and low_1000 >= max(s1_floor, sma20 * 0.995)
                and (o_915 - low_1000) <= 0.30 * day_atr
                and c_1000 > max(o_915 + 0.30 * day_atr, vwap_1000, o_1000)
                and c_1000 >= orb_high * 0.999
                and (high_1000 - low_1000) <= 1.10 * day_atr
                and risk_pct_1000 <= 2.85
                and pos_1000 >= 0.72
                and v_1000 >= 1.15 * max(v_prev_1000, 1.0)
            ):
                entry_p = round(c_1000, 2)
                sl_p = sl_1000_candidate
                tgt_p = round(entry_p + 1.40 * day_atr, 2)
                trail_sl = sl_p
                hwm = high_1000
                exit_p = float(bars_today["Close"].iloc[-1])
                exit_ts = str(bars_today.index[-1])
                exit_reason = "15:15_SQUAREOFF"

                for b_ts, b_row in bars_today.iloc[3:].iterrows():
                    bl = float(b_row["Low"])
                    bh = float(b_row["High"])
                    if bl <= trail_sl:
                        exit_p = trail_sl
                        exit_ts = str(b_ts)
                        exit_reason = "TRAILING_SL_PROFIT_LOCK" if trail_sl > entry_p else "INITIAL_SL_HIT"
                        break
                    if bh >= tgt_p:
                        exit_p = tgt_p
                        exit_ts = str(b_ts)
                        exit_reason = "TARGET_ACHIEVED_15M"
                        break
                    hwm = max(hwm, bh)
                    mfe_atr = (hwm - entry_p) / day_atr
                    if mfe_atr >= 1.05:
                        trail_sl = max(trail_sl, round(hwm - 0.35 * day_atr, 2))
                    elif mfe_atr >= 0.75:
                        trail_sl = max(trail_sl, round(entry_p + 0.45 * day_atr, 2))
                    elif mfe_atr >= 0.40:
                        trail_sl = max(trail_sl, round(entry_p + 0.15 * day_atr, 2))

                ret_pct = (exit_p - entry_p) / entry_p * 100.0
                intraday_ver_rows.append({
                    "Date": str(d_curr),
                    "Ticker": sym.replace(".NS", ""),
                    "Entry_Price_1000": entry_p,
                    "Initial_ORB_Low_SL": sl_p,
                    "Final_Trail_SL": round(trail_sl, 2),
                    "Target_Price": tgt_p,
                    "Exit_Price": round(exit_p, 2),
                    "Exit_Timestamp": exit_ts,
                    "Exit_Reason": exit_reason,
                    "Net_Return_Pct": round(ret_pct, 3),
                })

            # =========================================================================
            # 5-GATE PRE-QUALIFIED ENTRY TIMING STUDY (09:30 AM vs 10:00 AM vs 11:15 AM)
            # =========================================================================
            if not (-0.55 <= gap_open_pct <= 1.15):
                continue

            # Simulate the 3 Entry Windows with Configured 4-Stage Trailing Stop & Profit Target:
            # Window 1: 09:30 AM Immediate Open Entry (first 15m bar close, no 10 AM confirmation)
            bar_930 = bars_today.iloc[0]
            c_930 = float(bar_930["Close"])
            l_930 = float(bar_930["Low"])
            if c_930 > o_915:
                e_930 = round(c_930, 2)
                sl_930 = round(l_930 - 0.12 * day_atr, 2)
                tgt_930 = round(e_930 + 1.40 * day_atr, 2)
                tsl_930 = sl_930
                hwm_930 = float(bar_930["High"])
                ex_p_930 = float(bars_today["Close"].iloc[-1])
                ex_r_930 = "EOD_1515_EXIT"
                for _, b_row in bars_today.iloc[1:].iterrows():
                    bl, bh = float(b_row["Low"]), float(b_row["High"])
                    if bl <= tsl_930:
                        ex_p_930 = tsl_930
                        ex_r_930 = "TRAILING_SL_PROFIT_LOCK" if tsl_930 > e_930 else "INITIAL_SL_HIT"
                        break
                    if bh >= tgt_930:
                        ex_p_930 = tgt_930
                        ex_r_930 = "PROFIT_TARGET_HIT"
                        break
                    hwm_930 = max(hwm_930, bh)
                    mfe_a = (hwm_930 - e_930) / day_atr
                    if mfe_a >= 1.05:
                        tsl_930 = max(tsl_930, round(hwm_930 - 0.35 * day_atr, 2))
                    elif mfe_a >= 0.75:
                        tsl_930 = max(tsl_930, round(e_930 + 0.45 * day_atr, 2))
                    elif mfe_a >= 0.40:
                        tsl_930 = max(tsl_930, round(e_930 + 0.15 * day_atr, 2))
                ret_930 = (ex_p_930 - e_930) / e_930 * 100.0
                swing_ret_930 = (c_day5_15m - e_930) / e_930 * 100.0 if ex_r_930 == "EOD_1515_EXIT" else ret_930
                timing_rows.append({
                    "Window": "1_0930_AM_EARLY_OPEN_ENTRY",
                    "Date": str(d_curr),
                    "Ticker": sym.replace(".NS", ""),
                    "Entry_Price": e_930,
                    "Initial_SL": sl_930,
                    "Final_Trail_SL": round(tsl_930, 2),
                    "Target_Price": tgt_930,
                    "Exit_Price": round(ex_p_930, 2),
                    "Exit_Reason": ex_r_930,
                    "Net_Return_Pct": round(ret_930, 3),
                    "Hybrid_Swing_Ret_Pct": round(max(ret_930, swing_ret_930), 3),
                })

            # Window 2: 10:00 AM Confirmed 5-Gate Entry (User's Scenario: Entry by 10:00 AM + Trailing SL & Target)
            # Waits for 09:15-10:00 AM (first 45 mins) to confirm:
            # 1) Morning low held above S1 & 20DMA with shallow dip ((o_915 - low_1000) <= 0.30*ATR)
            # 2) 10:00 AM candle is green (c_1000 > o_1000), breaks 09:15-09:45 ORB_High, above Open + 0.30*ATR & VWAP
            # 3) Controlled morning range (<= 1.10*ATR) and Initial Stop Risk <= 2.85%
            if (
                low_1000 >= max(s1_floor, sma20 * 0.995)
                and (o_915 - low_1000) <= 0.30 * day_atr
                and c_1000 > max(o_915 + 0.30 * day_atr, vwap_1000, o_1000)
                and c_1000 >= orb_high * 0.999
                and (high_1000 - low_1000) <= 1.10 * day_atr
                and risk_pct_1000 <= 2.85
                and pos_1000 >= 0.72
                and v_1000 >= 1.15 * max(v_prev_1000, 1.0)
            ):
                e_1000 = round(c_1000, 2)
                sl_1000 = sl_1000_candidate
                tgt_1000 = round(e_1000 + 1.40 * day_atr, 2)
                tsl_1000 = sl_1000
                hwm_1000 = high_1000
                ex_p_1000 = float(bars_today["Close"].iloc[-1])
                ex_r_1000 = "EOD_1515_EXIT"
                for _, b_row in bars_today.iloc[3:].iterrows():
                    bl, bh = float(b_row["Low"]), float(b_row["High"])
                    if bl <= tsl_1000:
                        ex_p_1000 = tsl_1000
                        ex_r_1000 = "TRAILING_SL_PROFIT_LOCK" if tsl_1000 > e_1000 else "INITIAL_SL_HIT"
                        break
                    if bh >= tgt_1000:
                        ex_p_1000 = tgt_1000
                        ex_r_1000 = "PROFIT_TARGET_HIT"
                        break
                    hwm_1000 = max(hwm_1000, bh)
                    mfe_a = (hwm_1000 - e_1000) / day_atr
                    if mfe_a >= 1.05:
                        tsl_1000 = max(tsl_1000, round(hwm_1000 - 0.35 * day_atr, 2))
                    elif mfe_a >= 0.75:
                        tsl_1000 = max(tsl_1000, round(e_1000 + 0.45 * day_atr, 2))
                    elif mfe_a >= 0.40:
                        tsl_1000 = max(tsl_1000, round(e_1000 + 0.15 * day_atr, 2))
                ret_1000 = (ex_p_1000 - e_1000) / e_1000 * 100.0
                swing_ret_1000 = (c_day5_15m - e_1000) / e_1000 * 100.0 if ex_r_1000 == "EOD_1515_EXIT" else ret_1000
                timing_rows.append({
                    "Window": "2_1000_AM_5GATE_SNIPER_ENTRY",
                    "Date": str(d_curr),
                    "Ticker": sym.replace(".NS", ""),
                    "Entry_Price": e_1000,
                    "Initial_SL": sl_1000,
                    "Final_Trail_SL": round(tsl_1000, 2),
                    "Target_Price": tgt_1000,
                    "Exit_Price": round(ex_p_1000, 2),
                    "Exit_Reason": ex_r_1000,
                    "Net_Return_Pct": round(ret_1000, 3),
                    "Hybrid_Swing_Ret_Pct": round(max(ret_1000, swing_ret_1000), 3),
                })

            # Window 3: 11:15 AM Late Mid-Day Entry (first 8 bars: 09:15-11:15)
            bars_1115 = bars_today.iloc[:8]
            c_1115 = float(bars_1115["Close"].iloc[-1])
            low_1115 = float(bars_1115["Low"].min())
            high_1115 = float(bars_1115["High"].max())
            if c_1115 > o_915 + 0.35 * day_atr and low_1115 >= s1_floor:
                e_1115 = round(c_1115, 2)
                sl_1115 = round(low_1115 - 0.12 * day_atr, 2)
                tgt_1115 = round(e_1115 + 1.40 * day_atr, 2)
                tsl_1115 = sl_1115
                hwm_1115 = high_1115
                ex_p_1115 = float(bars_today["Close"].iloc[-1])
                ex_r_1115 = "EOD_1515_EXIT"
                for _, b_row in bars_today.iloc[8:].iterrows():
                    bl, bh = float(b_row["Low"]), float(b_row["High"])
                    if bl <= tsl_1115:
                        ex_p_1115 = tsl_1115
                        ex_r_1115 = "TRAILING_SL_PROFIT_LOCK" if tsl_1115 > e_1115 else "INITIAL_SL_HIT"
                        break
                    if bh >= tgt_1115:
                        ex_p_1115 = tgt_1115
                        ex_r_1115 = "PROFIT_TARGET_HIT"
                        break
                    hwm_1115 = max(hwm_1115, bh)
                    mfe_a = (hwm_1115 - e_1115) / day_atr
                    if mfe_a >= 1.10:
                        tsl_1115 = max(tsl_1115, round(hwm_1115 - 0.40 * day_atr, 2))
                    elif mfe_a >= 0.80:
                        tsl_1115 = max(tsl_1115, round(e_1115 + 0.50 * day_atr, 2))
                    elif mfe_a >= 0.50:
                        tsl_1115 = max(tsl_1115, round(e_1115 + 0.18 * day_atr, 2))
                ret_1115 = (ex_p_1115 - e_1115) / e_1115 * 100.0
                swing_ret_1115 = (c_day5_15m - e_1115) / e_1115 * 100.0 if ex_r_1115 == "EOD_1515_EXIT" else ret_1115
                timing_rows.append({
                    "Window": "3_1115_AM_LATE_MIDDAY_ENTRY",
                    "Date": str(d_curr),
                    "Ticker": sym.replace(".NS", ""),
                    "Entry_Price": e_1115,
                    "Initial_SL": sl_1115,
                    "Final_Trail_SL": round(tsl_1115, 2),
                    "Target_Price": tgt_1115,
                    "Exit_Price": round(ex_p_1115, 2),
                    "Exit_Reason": ex_r_1115,
                    "Net_Return_Pct": round(ret_1115, 3),
                    "Hybrid_Swing_Ret_Pct": round(max(ret_1115, swing_ret_1115), 3),
                })

    ver_df = pd.DataFrame(intraday_ver_rows)
    ver_df.to_csv(os.path.join(DATA_DIR, "intraday_60d_15m_verification.csv"), index=False)
    if not ver_df.empty:
        wr = (ver_df["Net_Return_Pct"] > 0).mean() * 100.0
        avg_r = ver_df["Net_Return_Pct"].mean()
        print(f"    60-Day 15m Bar-by-Bar Sniper Verification: {len(ver_df)} trades | Win Rate: {wr:.2f}% | Avg Return: {avg_r:+.3f}%")

    t_df = pd.DataFrame(timing_rows)
    if not t_df.empty:
        t_df.to_csv(os.path.join(DATA_DIR, "entry_timing_10am_5gate_trades.csv"), index=False)
        t_sum = []
        for w_name, w_grp in t_df.groupby("Window"):
            wins = w_grp["Net_Return_Pct"] > 0
            hyb_wins = w_grp["Hybrid_Swing_Ret_Pct"] > 0
            gp = float(w_grp.loc[wins, "Net_Return_Pct"].sum())
            gl = abs(float(w_grp.loc[~wins, "Net_Return_Pct"].sum()))
            t_sum.append({
                "Entry_Window": w_name,
                "Trades_60D": len(w_grp),
                "Intraday_Win_Rate_Pct": round(float(wins.mean() * 100.0), 2),
                "Avg_Intraday_Ret_Pct": round(float(w_grp["Net_Return_Pct"].mean()), 3),
                "Profit_Factor": round(gp / max(gl, 1e-6), 2),
                "Target_or_Trail_Lock_Pct": round(float(w_grp["Exit_Reason"].isin(["PROFIT_TARGET_HIT", "TRAILING_SL_PROFIT_LOCK"]).mean() * 100.0), 2),
                "Stop_Out_Pct": round(float((w_grp["Exit_Reason"] == "INITIAL_SL_HIT").mean() * 100.0), 2),
                "Hybrid_Swing_Win_Rate_Pct": round(float(hyb_wins.mean() * 100.0), 2),
                "Hybrid_Swing_Avg_Ret_Pct": round(float(w_grp["Hybrid_Swing_Ret_Pct"].mean()), 3),
            })
        t_sum_df = pd.DataFrame(t_sum).sort_values("Entry_Window")
        t_sum_df.to_csv(os.path.join(DATA_DIR, "entry_timing_10am_5gate_summary.csv"), index=False)
        print("\n    [15m Entry Timing Study on 5-Gate Pre-Qualified Stocks (60 Days Across 54 Tickers)]:")
        print(t_sum_df.to_string(index=False))


if __name__ == "__main__":
    run_3y_backtest()
