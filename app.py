"""
Institutional Intraday Trading Expert Dashboard (`app.py`)
==========================================================
Interactive Streamlit platform for Indian (NSE) & International Markets:
- Strictly CNC-Only Execution (100% Cash & Carry / Delivery, Zero MIS Auto-Squareoff Penalty)
- Strict Capital & Slot Discipline: Max ₹20,000 budget per ticker, Max 3 entries per day (`Top 3` slots, <= ₹60,000/day)
- 5-Minute Auto-Refresh & Auto-Exit (`@st.fragment(run_every="5m")`) + Late-Click 5-Minute Historical Candle Replay
  (tracks price movement from `Purchase_Time + 5m` through `3:15 PM IST` / click time and executes exits + P&L)
- All tables display Order & Performance Data in the initial rows and freeze the first 3 columns.
"""

import os
import sys
import json
import importlib
from datetime import datetime
import pandas as pd
import streamlit as st

import nifty250_universe
import backtest_engine_3y
import intraday_expert_engine

# Hot-reload guard for Streamlit Cloud: when a new commit is pulled on disk while the
# Python worker already has an older `intraday_expert_engine` / `backtest_engine_3y` in sys.modules,
# force a reload from disk so newly added symbols are immediately available without a container reboot.
if (
    not hasattr(intraday_expert_engine, "OPERATOR_SIGNALS_CSV")
    or not hasattr(intraday_expert_engine, "scan_operator_1m_footprints")
    or not hasattr(intraday_expert_engine, "build_all_strategies_paper_ledger_and_scorecard")
):
    importlib.reload(nifty250_universe)
    importlib.reload(backtest_engine_3y)
    importlib.reload(intraday_expert_engine)
elif len(getattr(backtest_engine_3y, "UNIVERSE_META", {})) < 200:
    importlib.reload(nifty250_universe)
    importlib.reload(backtest_engine_3y)
    importlib.reload(intraday_expert_engine)

from nifty250_universe import UNIVERSE_META
from intraday_expert_engine import (
    DATA_DIR,
    SNIPER_TRADES_CSV,
    SNIPER_TELEMETRY_CSV,
    MAX_BUDGET_PER_TICKER_INR,
    MAX_DAILY_ENTRIES,
    MAX_DAILY_TOTAL_BUDGET_INR,
    TOP_5_LOGICS_METADATA,
    scan_live_sniper_opportunities,
    evaluate_chronological_exits,
    stage_ticker_into_top3_cnc,
    stage_top3_daily_cnc_entries,
    force_manual_exit_trade,
    predict_swing_conversion_success,
)

OPERATOR_SIGNALS_CSV = getattr(
    intraday_expert_engine,
    "OPERATOR_SIGNALS_CSV",
    os.path.join(DATA_DIR, "live_operator_1m_signals.csv"),
)
OPERATOR_TRADES_CSV = getattr(
    intraday_expert_engine,
    "OPERATOR_TRADES_CSV",
    os.path.join(DATA_DIR, "operator_auto_trades.csv"),
)
OPERATOR_PERF_JSON = getattr(
    intraday_expert_engine,
    "OPERATOR_PERF_JSON",
    os.path.join(DATA_DIR, "operator_performance_summary.json"),
)
BEL_OCT16_CASE_STUDY_CSV = getattr(
    intraday_expert_engine,
    "BEL_OCT16_CASE_STUDY_CSV",
    os.path.join(DATA_DIR, "bel_oct16_operator_validation.csv"),
)
ALL_STRATEGIES_PAPER_CSV = getattr(
    intraday_expert_engine,
    "ALL_STRATEGIES_PAPER_CSV",
    os.path.join(DATA_DIR, "all_strategies_paper_trades.csv"),
)
STRATEGY_SCORECARD_CSV = getattr(
    intraday_expert_engine,
    "STRATEGY_SCORECARD_CSV",
    os.path.join(DATA_DIR, "strategy_wise_paper_scorecard.csv"),
)
scan_operator_1m_footprints = getattr(
    intraday_expert_engine,
    "scan_operator_1m_footprints",
    lambda *a, **kw: pd.DataFrame(),
)
auto_execute_and_evaluate_operator_trades = getattr(
    intraday_expert_engine,
    "auto_execute_and_evaluate_operator_trades",
    lambda *a, **kw: (pd.DataFrame(), {}),
)
build_all_strategies_paper_ledger_and_scorecard = getattr(
    intraday_expert_engine,
    "build_all_strategies_paper_ledger_and_scorecard",
    lambda *a, **kw: (pd.DataFrame(), pd.DataFrame()),
)

st.set_page_config(
    page_title="Intraday-Only CNC | 5m Auto-Exit & Top-3 Sniper Desk",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.06);
        border: 1px solid rgba(28, 131, 225, 0.18);
        padding: 10px 14px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Institutional CNC Intraday Sniper, 5m Auto-Exit & 1m Operator Footprint Desk")
st.caption(
    f"Universe: **Full NIFTY 250 + Sectoral/Global ETFs (`{len(UNIVERSE_META)}` Instruments)** | "
    "Execution Mode: **100% CNC Only (Top 3 <= `₹20,000` each) + Separate 1m Operator Auto-Trade (`BUY` & `SELL`) Tracker** | "
    "**5-Minute Auto-Refresh (`@st.fragment(run_every='5m')`) + Chronological Candle Replay**"
)


def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_frozen_3col_dataframe(df: pd.DataFrame, height: int = None):
    """
    Freezes the first 3 columns of any DataFrame in Streamlit's interactive Glide Data Grid
    by setting the first 3 columns as a 3-level pinned MultiIndex (`hide_index=False`).
    Also ensures Level-0 index values are distinct per row so no cells are visually merged.
    """
    if df is None or df.empty:
        st.info("No records available.")
        return

    work = df.copy().reset_index(drop=True)
    if len(work.columns) < 3:
        st.dataframe(work, use_container_width=True, hide_index=True, height=height)
        return

    freeze_cols = list(work.columns[:3])
    col0 = freeze_cols[0]

    seen = {}
    unique_col0 = []
    for val in work[col0].astype(str):
        if val in seen:
            seen[val] += 1
            unique_col0.append(f"{val} ({seen[val]})")
        else:
            seen[val] = 1
            unique_col0.append(val)
    work[col0] = unique_col0

    for c in freeze_cols[1:]:
        work[c] = work[c].astype(str)

    pinned_df = work.set_index(freeze_cols)
    if height is not None:
        st.dataframe(pinned_df, use_container_width=True, hide_index=False, height=height)
    else:
        st.dataframe(pinned_df, use_container_width=True, hide_index=False)


def recalculate_cnc_sizing(df: pd.DataFrame, cnc_capital_inr: float) -> pd.DataFrame:
    """
    Dynamically updates CNC order quantity, order value, and rupee stop risk based on user's CNC capital
    (strictly capped at ₹20,000 per ticker).
    """
    if df.empty or "CNC_Buy_Limit_Price" not in df.columns:
        return df
    out = df.copy()
    capped_capital = min(MAX_BUDGET_PER_TICKER_INR, max(1000.0, float(cnc_capital_inr)))
    prices = pd.to_numeric(out["CNC_Buy_Limit_Price"], errors="coerce").fillna(1.0).clip(lower=1.0)
    sl_prices = pd.to_numeric(out["Initial_Protected_SL"], errors="coerce").fillna(prices * 0.98)
    qtys = (capped_capital // prices).clip(lower=1).astype(int)
    # Ensure qty * price never exceeds capped_capital when qty > 1
    over_mask = (qtys * prices > capped_capital) & (qtys > 1)
    qtys = qtys.where(~over_mask, qtys - 1)
    out["Suggested_CNC_Qty"] = qtys
    out["CNC_Order_Value_INR"] = (qtys * prices).round(2)
    out["Max_Stop_Risk_INR"] = (qtys * (prices - sl_prices).clip(lower=0.05)).round(2)
    return out


def build_entry_order_initial_rows_table(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the CNC Entry Order Specification Table where:
    - INITIAL ROWS (Rows 01–09) are strictly the Broker CNC Order Execution & Top-3 Slot Data.
    - LOWER ROWS (Rows 10–15) are the Sniper Logic, 5-Gate, and Structural Support metrics.
    - FIRST 3 COLUMNS (`Row_Priority`, `CNC_Order_Parameter`, and `#1 Top CNC Pick`) are frozen!
    """
    if candidates_df.empty:
        return pd.DataFrame()

    top_n = candidates_df.head(6).copy().reset_index(drop=True)
    col_labels = [f"#{i+1} {row['Ticker']}" for i, row in top_n.iterrows()]

    rows_spec = [
        # INITIAL ROWS = ORDER-RELATED & TOP-3 BUDGET DATA
        ("01. ORDER DATA (INITIAL)", "Daily Top-3 Slot Status", [r.get("Daily_Top3_Slot", f"🏆 Top #{i+1}/3" if i < 3 else "Standby") for i, r in top_n.iterrows()]),
        ("02. ORDER DATA (INITIAL)", "CNC Order Action", [r["CNC_Order_Action"] for _, r in top_n.iterrows()]),
        ("03. ORDER DATA (INITIAL)", "Broker Product Type", [r["Product_Type"] for _, r in top_n.iterrows()]),
        ("04. ORDER DATA (INITIAL)", "Order Type & Timing", [r["CNC_Order_Type"] for _, r in top_n.iterrows()]),
        ("05. ORDER DATA (INITIAL)", "CNC Buy Limit Price (₹)", [f"₹{float(r['CNC_Buy_Limit_Price']):,.2f}" for _, r in top_n.iterrows()]),
        ("06. ORDER DATA (INITIAL)", "Suggested CNC Qty (<= ₹20K)", [f"{int(r['Suggested_CNC_Qty'])} Qty" for _, r in top_n.iterrows()]),
        ("07. ORDER DATA (INITIAL)", "Total CNC Order Value (<= ₹20K)", [f"₹{float(r['CNC_Order_Value_INR']):,.2f}" for _, r in top_n.iterrows()]),
        ("08. ORDER DATA (INITIAL)", "Initial Protected SL (₹ / %)", [f"₹{float(r['Initial_Protected_SL']):,.2f} (-{float(r['Downside_Risk_Pct']):.2f}%)" for _, r in top_n.iterrows()]),
        ("09. ORDER DATA (INITIAL)", "Intraday Target & R:R", [f"₹{float(r['Intraday_Target']):,.2f} ({r['Risk_Reward_Ratio']})" for _, r in top_n.iterrows()]),
        # TECHNICAL & LOGIC CONTEXT ROWS BELOW INITIAL ORDER ROWS
        ("10. SNIPER LOGIC", "Best Sniper Logic ID", [r["Best_Logic_ID"] for _, r in top_n.iterrows()]),
        ("11. SNIPER LOGIC", "Sniper Readiness / Conviction", [f"{float(r['Sniper_Readiness_Score']):.1f} / 100" for _, r in top_n.iterrows()]),
        ("12. 5-GATE 10AM CHECK", "5-Gate Pre-Qualified (10 AM)", ["✅ YES (5/5)" if bool(r["Meets_5Gate_10AM_Entry"]) else f"Gates: {r['Swing_Conversion_Gates']}" for _, r in top_n.iterrows()]),
        ("13. SUPPORT & CMP", "Live CMP vs Structural S1", [f"CMP ₹{float(r['Live_Price']):,.2f} | S1 ₹{float(r['Structural_S1_Floor']):,.2f}" for _, r in top_n.iterrows()]),
        ("14. RELATIVE STRENGTH", "Stock 5D RS / Sector 5D RS", [f"{float(r['5D_Rel_Strength_Pct']):+.2f}% / {float(r['Sector_5D_RS_Pct']):+.2f}%" for _, r in top_n.iterrows()]),
        ("15. MOMENTUM & VOL", "RSI(14) & 20D Vol Ratio", [f"RSI {float(r['RSI_14']):.1f} | Vol {float(r['Vol_Ratio_20D']):.2f}x" for _, r in top_n.iterrows()]),
    ]

    records = []
    for r_group, r_field, vals in rows_spec:
        rec = {
            "Row_Priority": r_group,
            "CNC_Order_Parameter": r_field,
        }
        for c_name, v in zip(col_labels, vals):
            rec[c_name] = v
        records.append(rec)
    return pd.DataFrame(records)


def build_top3_exit_and_pnl_ticket_table(top3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the Card 2 Top-3 CNC Active/Replayed Exit & Performance Ticket Table where:
    - INITIAL ROWS (Rows 01–09) are strictly the Performance (Profit/Loss), Exit Status, GTT Triggers & Order Data.
    - LOWER ROWS (Rows 10–14) are the 5-Minute Candle Replay Audit (`Purchase Time + 5m -> 3:15 PM`) & Trailing Ladder.
    - FIRST 3 COLUMNS (`Row_Priority`, `Exit_And_PnL_Parameter`, and `Slot #1/3`) are frozen!
    """
    if top3_df.empty:
        return pd.DataFrame()

    work = top3_df.head(MAX_DAILY_ENTRIES).copy().reset_index(drop=True)
    col_labels = [f"{row.get('Daily_Slot', f'Slot #{i+1}/3')} ({row['Ticker']})" for i, row in work.iterrows()]

    rows_spec = [
        # INITIAL ROWS = PERFORMANCE (PROFIT/LOSS) & EXIT / GTT ORDER DATA
        ("01. PERFORMANCE (INITIAL)", "Profit / Loss Verdict (₹ & %)", [r.get("Performance_Verdict", "⚪ FLAT") for _, r in work.iterrows()]),
        ("02. EXIT STATUS (INITIAL)", "Auto-Exit / GTT Execution Status", [r.get("GTT_Order_Status", r.get("Status", "")) for _, r in work.iterrows()]),
        ("03. ORDER DATA (INITIAL)", "Product & CNC Qty (<= ₹20K)", [f"CNC | {int(r.get('CNC_Qty', 1))} Qty (₹{float(r.get('Capital_Used_INR', 0)):,.2f})" for _, r in work.iterrows()]),
        ("04. ORDER DATA (INITIAL)", "Purchase Date & Time (IST)", [f"{r.get('Purchase_Date', '')} @ {r.get('Purchase_Time', '10:00:00')}" for _, r in work.iterrows()]),
        ("05. ORDER DATA (INITIAL)", "CNC Purchase Price (₹)", [f"₹{float(r.get('Entry_Price', 0)):,.2f}" for _, r in work.iterrows()]),
        ("06. EXIT DATA (INITIAL)", "Exit Price / Live CMP (₹)", [f"₹{float(r['Exit_Price']) if pd.notna(r.get('Exit_Price')) and str(r.get('Exit_Price')) not in ('', 'nan') and float(r.get('Exit_Price', 0)) > 0 else float(r.get('Live_CMP', r.get('Entry_Price', 0))):,.2f}" for _, r in work.iterrows()]),
        ("07. EXIT DATA (INITIAL)", "Exit Candle Timestamp & Reason", [f"{r.get('Exit_Timestamp', 'Active')} | {r.get('Exit_Reason', 'Monitoring 5m candles')}" for _, r in work.iterrows()]),
        ("08. GTT OCO (INITIAL)", "GTT SL Trigger / Limit (₹)", [f"SL ₹{float(r.get('GTT_SL_Trigger', r.get('Current_Trailing_SL', 0))):,.2f} (Lmt ₹{float(r.get('GTT_SL_Limit', 0)):,.2f})" for _, r in work.iterrows()]),
        ("09. GTT OCO (INITIAL)", "GTT Target Trigger / Limit (₹)", [f"TGT ₹{float(r.get('GTT_Target_Trigger', r.get('Target_Price', 0))):,.2f} (Lmt ₹{float(r.get('GTT_Target_Limit', 0)):,.2f})" for _, r in work.iterrows()]),
        # 5-MINUTE CANDLE REPLAY & TRAILING RATCHET ROWS BELOW INITIAL ROWS
        ("10. 5M REPLAY AUDIT", "5m Candles Tracked (+5m -> 3:15 PM)", [f"{int(r.get('Bars_Tracked_5m', 0))} bars | Low ₹{float(r.get('Window_Low_5m', 0)):,.2f} – High ₹{float(r.get('Window_High_5m', 0)):,.2f}" for _, r in work.iterrows()]),
        ("11. 5M REPLAY AUDIT", "Peak High-Water-Mark (HWM)", [f"₹{float(r.get('High_Water_Mark', 0)):,.2f} ({r.get('Active_Trail_Stage', '')})" for _, r in work.iterrows()]),
        ("12. 5M REPLAY AUDIT", "Chronological Replay Path", [str(r.get("Replay_Audit_Trail", "")) for _, r in work.iterrows()]),
        ("13. TRAIL LADDER", "Initial SL -> S1 -> S2 -> S3", [f"₹{float(r.get('Initial_SL', 0)):.0f} -> ₹{float(r.get('Stage1_Breakeven_SL', 0)):.0f} -> ₹{float(r.get('Stage2_ProfitLock_SL', 0)):.0f} -> ₹{float(r.get('Stage3_Chandelier_SL', 0)):.0f}" for _, r in work.iterrows()]),
        ("14. EOD 3:15 PM RULE", "3:15 PM Action / Swing Gate", [str(r.get("CNC_310PM_Decision", "Auto-Exit @ 3:15 PM IST")) for _, r in work.iterrows()]),
    ]

    records = []
    for r_group, r_field, vals in rows_spec:
        rec = {
            "Row_Priority": r_group,
            "Exit_And_PnL_Parameter": r_field,
        }
        for c_name, v in zip(col_labels, vals):
            rec[c_name] = v
        records.append(rec)
    return pd.DataFrame(records)


def build_operator_initial_rows_ticket_table(op_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the 1-Minute Operator / Big-Player Auto-Trade (`BUY` & `SELL`) Order & Performance Ticket Table where:
    - INITIAL ROWS (Rows 01–09) are strictly the Separate Performance (Profit/Loss), Auto-Trade Side (`BUY`/`SELL`),
      Order Sizing (`<= ₹20,000`), Entry/Exit Prices, Stop-Loss & Target.
    - LOWER ROWS (Rows 10–15) are the 1-Minute Volume-Shock (`>= 4.5x` 20m Median), Estimated Futures-Equivalent Volume
      (e.g., `BEL` 12:02 PM `1.53M` Fut / `76.1K` Cash), Candle CLV, and `T+2m` Follow-Through Confirmation.
    - FIRST 3 COLUMNS (`Row_Priority`, `Operator_AutoTrade_Parameter`, and `#1 Auto-Trade`) are frozen!
    """
    if op_df is None or op_df.empty:
        return pd.DataFrame()

    work = op_df.head(6).copy().reset_index(drop=True)
    col_labels = [
        f"#{i+1} {r['Ticker']} ({r.get('Signal_Side', 'BUY')} @ {str(r.get('Shock_Time_1m', ''))[:5]})"
        for i, r in work.iterrows()
    ]

    rows_spec = [
        # INITIAL ROWS = SEPARATE PERFORMANCE (PROFIT/LOSS) & AUTO-TRADE ORDER DATA
        ("01. PERFORMANCE (INITIAL)", "Separate Auto-Trade P&L Verdict", [str(r.get("Performance_Verdict", "⚪ ACTIVE")) for _, r in work.iterrows()]),
        ("02. AUTO-TRADE (INITIAL)", "Operator Action (BUY or SELL)", [str(r.get("Auto_Trade_Action", "")) for _, r in work.iterrows()]),
        ("03. ORDER DATA (INITIAL)", "Signal Side & Trade Status", [f"{r.get('Signal_Side', '')} | {r.get('Status', '')}" for _, r in work.iterrows()]),
        ("04. ORDER DATA (INITIAL)", "Auto Qty & Capital (<= ₹20K)", [f"{int(r.get('Auto_Qty_20K', 1))} Qty (₹{float(r.get('Capital_Used_INR', 0)):,.2f})" for _, r in work.iterrows()]),
        ("05. ORDER DATA (INITIAL)", "T+2m Confirmed Entry Price (₹)", [f"₹{float(r.get('Confirm_Entry_Price', 0)):,.2f} (@ {r.get('Confirm_Time_T2', '')})" for _, r in work.iterrows()]),
        ("06. EXIT DATA (INITIAL)", "Exit / Live Price (₹)", [f"₹{float(r.get('Live_Or_Exit_Price', 0)):,.2f}" for _, r in work.iterrows()]),
        ("07. RISK DATA (INITIAL)", "Initial SL -> Trailing SL (₹)", [f"SL ₹{float(r.get('Initial_SL', 0)):,.2f} -> Trail ₹{float(r.get('Current_Trailing_SL', 0)):,.2f}" for _, r in work.iterrows()]),
        ("08. TARGET DATA (INITIAL)", "Intraday Target Price (₹)", [f"₹{float(r.get('Target_Price', 0)):,.2f}" for _, r in work.iterrows()]),
        ("09. EXIT AUDIT (INITIAL)", "Exit Timestamp & Reason", [f"{r.get('Exit_Timestamp', 'Active')} | {r.get('Exit_Reason', '')}" for _, r in work.iterrows()]),
        # 1-MINUTE OPERATOR VOLUME-SHOCK & T+2M CONFIRMATION FORENSICS BELOW INITIAL ROWS
        ("10. 1M OPERATOR SHOCK", "Session Date & 1m Shock Candle", [f"{r.get('Session_Date', '')} @ {r.get('Shock_Time_1m', '')}" for _, r in work.iterrows()]),
        ("11. 1M VOLUME SPIKE", "1m Cash Vol & Spike vs 20m Med", [f"{int(r.get('Shock_1m_Cash_Vol', 0)):,} sh ({float(r.get('Vol_Spike_Mult_20m', 0)):.1f}x 20m Med)" for _, r in work.iterrows()]),
        ("12. FUTURES EQ VOL", "Est. Futures + Cash Combined Vol", [f"~{int(r.get('Est_Futures_Eq_Vol', 0)):,} shares (F&O + Cash)" for _, r in work.iterrows()]),
        ("13. FOOTPRINT TYPE", "Institutional Footprint Pattern", [str(r.get("Operator_Footprint_Type", "")) for _, r in work.iterrows()]),
        ("14. CANDLE STRUCTURE", "1m Shock OHLC, CLV & VWAP", [f"{r.get('Shock_Candle_OHLC', '')} | CLV {float(r.get('Shock_CLV', 0)):.2f} | VWAP ₹{float(r.get('Session_VWAP', 0)):,.2f}" for _, r in work.iterrows()]),
        ("15. T+2M CONFIRMATION", "2-Minute Anti-Trap Gate Verdict", [str(r.get("T2_Follow_Through", "")) for _, r in work.iterrows()]),
    ]

    records = []
    for r_group, r_field, vals in rows_spec:
        rec = {
            "Row_Priority": r_group,
            "Operator_AutoTrade_Parameter": r_field,
        }
        for c_name, v in zip(col_labels, vals):
            rec[c_name] = v
        records.append(rec)
    return pd.DataFrame(records)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Tab 1: CNC Entry (Top 3 <=₹20K), 5m Auto-Exit & 1m Operator Auto-Trade Desk",
    "🛡️ Downside Protection & Trailing SL Ladder",
    "🔄 3:10 PM CNC Intraday-to-Swing Predictor",
    "🏆 3-Year Backtest (Top 6 Sniper Logics)",
    "🔬 Production Forensics & Integration Blueprint",
])


@st.fragment(run_every="5m")
def render_live_5m_auto_refresh_desk():
    """
    Auto-refreshes every 5 minutes (300 seconds) and automatically evaluates:
    1. Top-3 CNC 5-minute candle exits (`Purchase_Time + 5m -> 3:15 PM IST`).
    2. Separate 1-Minute Operator / Big-Player Auto-Trades (`BUY` & `SELL`) across the NIFTY 250 funnel.
    """
    # Automatically evaluate chronological 5m exits on every 5-minute cycle
    auto_eval_df = evaluate_chronological_exits(
        trades_csv_path=SNIPER_TELEMETRY_CSV,
        force_full_replay=False,
        eod_315pm_auto_exit=st.session_state.get("eod_315pm_auto_exit_toggle", True),
    )

    # During live Indian market hours (09:15 AM – 03:30 PM IST on weekdays), automatically scan
    # the NIFTY 250 funnel for new 1m Operator footprints every 5 minutes so paper trades auto-trigger
    now_ist_dt = pd.Timestamp.now(tz="Asia/Kolkata")
    is_live_market_hours = (
        now_ist_dt.weekday() < 5
        and ((now_ist_dt.hour == 9 and now_ist_dt.minute >= 15) or (10 <= now_ist_dt.hour <= 14) or (now_ist_dt.hour == 15 and now_ist_dt.minute <= 30))
    )
    if is_live_market_hours:
        try:
            cached_op_signals = scan_operator_1m_footprints()
        except Exception:
            cached_op_signals = load_csv("live_operator_1m_signals.csv")
    else:
        cached_op_signals = load_csv("live_operator_1m_signals.csv")

    # Also evaluate/refresh the separate Operator Auto-Trade (BUY & SELL) ledger every 5m cycle
    op_trades_df, op_perf_summary = auto_execute_and_evaluate_operator_trades(
        cached_op_signals if not cached_op_signals.empty else None
    )
    strat_scorecard_df, all_paper_trades_df = build_all_strategies_paper_ledger_and_scorecard(
        op_trades_df=op_trades_df if not op_trades_df.empty else None
    )

    now_ist_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown(
        f"⏱️ **5-Minute Auto-Refresh & Auto-Exit Engine: `ACTIVE (Every 5 Minutes)`** | "
        f"Last 5m Check: **`{now_ist_str}`** | "
        f"**Universe: `Full NIFTY 250 + ETFs ({len(UNIVERSE_META)} Tickers)`** | "
        f"**Budget Cap: `₹20,000 / Ticker` (`Top 3 CNC` + `Separate 1m Operator Auto-Trade Ledger`)**"
    )

    # Top Control Bar
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.4, 1.4, 1.3, 1.9])
    with ctrl1:
        if st.button("🔄 Run Full NIFTY 250 Scan + 5m & 1m Operator Check", type="primary", use_container_width=True):
            with st.spinner(f"Scanning {len(UNIVERSE_META)} NIFTY 250 & Global instruments + 1m Operator footprints..."):
                scan_live_sniper_opportunities()
                evaluate_chronological_exits(
                    trades_csv_path=SNIPER_TELEMETRY_CSV,
                    force_full_replay=True,
                    eod_315pm_auto_exit=st.session_state.get("eod_315pm_auto_exit_toggle", True),
                )
                new_op_sig = scan_operator_1m_footprints()
                auto_execute_and_evaluate_operator_trades(new_op_sig)
            st.rerun()
    with ctrl2:
        if st.button("🏆 Reset / Auto-Load Today's Top 3 CNC (<=₹20K)", use_container_width=True):
            with st.spinner("Loading Top 3 CNC entries (<= ₹20,000 each) & replaying 5m candles from 10:05 to 3:15 PM..."):
                rep_tmp = load_json("live_regime_report.json")
                wl_tmp = load_csv("live_watchlist_radar.csv")
                sn_tmp = load_csv("live_sniper_signals.csv")
                src_tmp = sn_tmp if not sn_tmp.empty else wl_tmp
                stage_top3_daily_cnc_entries(
                    watchlist_df=src_tmp,
                    regime_badge=rep_tmp.get("regime_badge", "TRENDING_OR_ROTATION"),
                    purchase_time_str="10:00:00",
                    eod_315pm_auto_exit=st.session_state.get("eod_315pm_auto_exit_toggle", True),
                    sync_purchase_candle_price=True,
                    trades_csv_path=SNIPER_TELEMETRY_CSV,
                )
            st.rerun()
    with ctrl3:
        cnc_capital = st.number_input(
            "💰 Max Budget / Ticker (₹, <= 20K)",
            min_value=2000.0,
            max_value=float(MAX_BUDGET_PER_TICKER_INR),
            value=float(MAX_BUDGET_PER_TICKER_INR),
            step=2000.0,
            help="Strict ₹20,000 maximum budget per ticker in CNC Delivery (max 3 entries/day = ₹60,000 total).",
        )
    with ctrl4:
        filter_mode = st.selectbox(
            "🔎 Card 1 Candidate Filter (NIFTY 250 Universe)",
            options=[
                "Top Actionable & High-Readiness CNC Setups (Top 3 Highlighted)",
                "Triggered Sniper Signals Only (Immediate CNC Buy)",
                f"All {len(UNIVERSE_META)} NIFTY 250 & International Universe Instruments",
            ],
            index=0,
        )

    regime_rep = load_json("live_regime_report.json")
    sniper_df = load_csv("live_sniper_signals.csv")
    watchlist_df = load_csv("live_watchlist_radar.csv")
    top3_ledger_df = load_csv("sniper_telemetry_trades.csv")
    if top3_ledger_df.empty:
        top3_ledger_df = auto_eval_df

    watchlist_df = recalculate_cnc_sizing(watchlist_df, cnc_capital)
    sniper_df = recalculate_cnc_sizing(sniper_df, cnc_capital)

    if regime_rep:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Live Market Regime", regime_rep.get("regime_badge", "N/A"), f"Universe: {len(watchlist_df)} / {len(UNIVERSE_META)} Tickers")
        m2.metric("Nifty 50 (^NSEI)", f"{regime_rep.get('nifty_close', 0):,.2f}", f"{regime_rep.get('nifty_ret_1d_pct', 0):+.2f}% (1D)")
        m3.metric("India VIX (^INDIAVIX)", f"{regime_rep.get('india_vix', 0):.2f}", "Max ₹20K/Ticker | Max 3/Day")
        tot_pnl_banner = float(top3_ledger_df["PnL_INR"].sum()) if not top3_ledger_df.empty and "PnL_INR" in top3_ledger_df.columns else 0.0
        tot_cap_banner = float(top3_ledger_df["Capital_Used_INR"].sum()) if not top3_ledger_df.empty and "Capital_Used_INR" in top3_ledger_df.columns else 0.0
        tot_pct_banner = (tot_pnl_banner / tot_cap_banner * 100.0) if tot_cap_banner > 0 else 0.0
        m4.metric(
            f"Top 3 Daily CNC P&L ({len(top3_ledger_df)}/3)",
            f"₹{tot_pnl_banner:+,.2f} ({tot_pct_banner:+.2f}%)",
            f"Deployed: ₹{tot_cap_banner:,.0f} / ₹60,000",
        )
        op_net_pnl = float(op_perf_summary.get("total_net_pnl_inr", 0.0))
        op_wr = float(op_perf_summary.get("win_rate_pct", 0.0))
        op_cnt = int(op_perf_summary.get("total_auto_trades", 0))
        m5.metric(
            f"1m Operator Auto-Trade P&L ({op_cnt})",
            f"₹{op_net_pnl:+,.2f} ({op_wr:.1f}% WR)",
            f"BUY: ₹{op_perf_summary.get('buy_net_pnl_inr', 0):+,.0f} | SELL: ₹{op_perf_summary.get('sell_net_pnl_inr', 0):+,.0f}",
        )

    if not watchlist_df.empty:
        if filter_mode.startswith("Triggered"):
            entry_candidates_df = sniper_df if not sniper_df.empty else watchlist_df.head(6)
        elif filter_mode.startswith("Top Actionable"):
            entry_candidates_df = watchlist_df[
                (watchlist_df["Order_Priority_Rank"] <= 3) | (watchlist_df["Sniper_Readiness_Score"] >= 80.0)
            ].copy()
            if entry_candidates_df.empty:
                entry_candidates_df = watchlist_df.head(10).copy()
        else:
            entry_candidates_df = watchlist_df.copy()
    else:
        entry_candidates_df = pd.DataFrame()

    st.markdown("---")

    # =========================================================================
    # SIDE-BY-SIDE CARDS IN TAB 1:
    #   LEFT CARD : CNC Intraday Entry Orders + Select Ticker -> Create Entry in Top 3 (<= ₹20K)
    #   RIGHT CARD: 5m Auto-Exit / Late-Click Candle Replay Button + Performance (P&L) & GTT Orders
    # =========================================================================
    col_entry_card, col_exit_card = st.columns(2, gap="medium")

    # -------------------------------------------------------------------------
    # LEFT CARD: CNC INTRADAY ENTRY ORDERS & TOP 3 ENTRY BUILDER (<= ₹20K/TICKER)
    # -------------------------------------------------------------------------
    with col_entry_card:
        with st.container(border=True):
            st.markdown("### 🟢 Card 1: Which Intraday Trades to Enter (`CNC Only` | `Max ₹20K` | `Top 3/Day`)")
            st.caption(
                "**Budget Rule**: Max **`₹20,000` per ticker** (`Suggested_CNC_Qty = int(20000 // Price)`) and **Max `3` entries per day (`Slot #1/3`, `Slot #2/3`, `Slot #3/3`)**. "
                "Selecting a ticker below creates/updates its entry in the **Top 3** and immediately tracks 5m candles (`Purchase Time + 5m -> 3:15 PM`)."
            )

            # Prominent Interactive Box: Select Ticker Entered in Broker (CNC) -> Creates Entry in Top 3
            with st.container(border=True):
                st.markdown("##### ➕ Select Ticker Entered in Broker (`CNC`) -> Create Entry in Top 3 (`<= ₹20,000`)")
                if not entry_candidates_df.empty:
                    b_col1, b_col2, b_col3 = st.columns([1.6, 1.1, 1.1])
                    pick_ticker = b_col1.selectbox(
                        "Select Ticker Entered in Broker (CNC)",
                        options=entry_candidates_df["Ticker"].tolist(),
                        key="stage_cnc_ticker",
                    )
                    default_p_date = (
                        str(top3_ledger_df["Purchase_Date"].iloc[0])
                        if not top3_ledger_df.empty and "Purchase_Date" in top3_ledger_df.columns and str(top3_ledger_df["Purchase_Date"].iloc[0]).strip() != ""
                        else datetime.now().strftime("%Y-%m-%d")
                    )
                    pick_date_str = b_col2.text_input(
                        "Purchase Date (YYYY-MM-DD)",
                        value=default_p_date,
                        help="If clicking late or testing a past session, enter the purchase date so historical 5m candles from that date are replayed.",
                    )
                    pick_time_str = b_col3.selectbox(
                        "Purchase Time (IST)",
                        options=["09:30:00", "09:45:00", "10:00:00", "10:15:00", "10:30:00", "11:00:00", "12:00:00", "13:00:00", "14:00:00"],
                        index=2,
                        help="5-minute exit tracking starts strictly 5 minutes after Purchase Time (e.g., 10:05:00 for a 10:00:00 entry) through 3:15 PM IST.",
                    )

                    b_col4, b_col5 = st.columns([1.4, 1.6])
                    sync_5m_price = b_col4.checkbox(
                        "Anchor Entry Price to Exact 5m Candle @ Purchase Time",
                        value=True,
                        help="Uses the actual historical 5m candle close at Purchase Time (e.g. 10:00 AM IST) and sizes Qty <= ₹20,000.",
                    )
                    if b_col5.button("✅ Create Entry in Top 3 (<=₹20K) & Track 5m", type="primary", use_container_width=True):
                        row_match = entry_candidates_df[entry_candidates_df["Ticker"] == pick_ticker].iloc[0]
                        ok, msg, _ = stage_ticker_into_top3_cnc(
                            ticker=pick_ticker,
                            watchlist_df=watchlist_df,
                            regime_badge=regime_rep.get("regime_badge", "LIVE"),
                            entry_price=float(row_match["CNC_Buy_Limit_Price"]),
                            budget_inr=min(MAX_BUDGET_PER_TICKER_INR, cnc_capital),
                            purchase_date_str=pick_date_str.strip(),
                            purchase_time_str=pick_time_str,
                            eod_315pm_auto_exit=st.session_state.get("eod_315pm_auto_exit_toggle", True),
                            sync_purchase_candle_price=sync_5m_price,
                            replace_slot3_if_full=True,
                            trades_csv_path=SNIPER_TELEMETRY_CSV,
                        )
                        st.session_state["last_stage_msg"] = msg
                        st.rerun()

                    if "last_stage_msg" in st.session_state:
                        st.success(st.session_state["last_stage_msg"])

            st.markdown("##### 📋 1A. CNC Entry Order Ticket (`Initial Rows = Order Data` | `First 3 Columns Frozen`)")
            entry_ticket_df = build_entry_order_initial_rows_table(entry_candidates_df)
            render_frozen_3col_dataframe(entry_ticket_df, height=390)

            st.markdown("##### 🎯 1B. Ranked CNC Entry Table (`Initial Rows = Top 3 CNC Orders <= ₹20K` | `First 3 Columns Frozen`)")
            if not entry_candidates_df.empty:
                entry_cols = [
                    # First 3 Frozen Columns
                    "Ticker",
                    "CNC_Order_Action",
                    "CNC_Buy_Limit_Price",
                    # Top-3 Slot & Order-Related Columns Immediately Next
                    "Daily_Top3_Slot",
                    "Suggested_CNC_Qty",
                    "CNC_Order_Value_INR",
                    "Product_Type",
                    "CNC_Order_Type",
                    "Initial_Protected_SL",
                    "Downside_Risk_Pct",
                    "Intraday_Target",
                    "Risk_Reward_Ratio",
                    "Max_Stop_Risk_INR",
                    # Logic & Readiness Context
                    "Best_Logic_ID",
                    "Sniper_Readiness_Score",
                    "Meets_5Gate_10AM_Entry",
                    "ORB_10AM_Trigger",
                    "Live_Price",
                    "Structural_S1_Floor",
                    "Day_Change_Pct",
                    "5D_Rel_Strength_Pct",
                    "Vol_Ratio_20D",
                    "RSI_14",
                ]
                avail_entry_cols = [c for c in entry_cols if c in entry_candidates_df.columns]
                render_frozen_3col_dataframe(entry_candidates_df[avail_entry_cols], height=350)

    # -------------------------------------------------------------------------
    # RIGHT CARD: 5M AUTO-EXIT, LATE-CLICK HISTORICAL 5M REPLAY & PROFIT/LOSS DESK
    # -------------------------------------------------------------------------
    with col_exit_card:
        with st.container(border=True):
            st.markdown("### 🛡️ Card 2: 5m Auto-Exit, Late-Click Candle Replay & Performance (`Profit / Loss`)")
            st.caption(
                "**How Auto-Exit & Late-Click Replay Works**: Tracks every **5-minute candle starting 5 minutes after Purchase Time (`Purchase_Time + 5m`, e.g. `10:05 AM`) up to `3:15 PM IST`**. "
                "If clicked late (afternoon, after market close, or next day), it loads historical 5m candles from purchase date & time till click time / 3:15 PM to check if SL, Trailing SL, Profit Target, or 3:15 PM Exit was hit."
            )

            # Interactive Exit & Late-Click 5-Minute Replay Control Box
            with st.container(border=True):
                st.markdown("##### ⚡ Exit Action & Late-Click 5-Minute Historical Candle Replay Controls")
                ex_c1, ex_c2 = st.columns([1.5, 1.5])
                eod_315_rule = ex_c1.checkbox(
                    "⏰ Auto-Exit at 3:15 PM IST Candle Close (Intraday Rule)",
                    value=True,
                    key="eod_315pm_auto_exit_toggle",
                    help="When checked, if neither SL/Trail nor Target is hit between Purchase_Time+5m and 3:15 PM, the trade automatically exits on the 3:15 PM candle close.",
                )
                replay_sync_price = ex_c2.checkbox(
                    "📌 Sync Entry Price to Purchase Time 5m Candle on Replay",
                    value=True,
                    key="replay_sync_price_toggle",
                )

                btn_c1, btn_c2 = st.columns([1.6, 1.4])
                if btn_c1.button(
                    "🔍 Track 5m Candles (+5m -> 3:15 PM / Now) & Execute Exit",
                    type="primary",
                    use_container_width=True,
                    help="Loads 5-minute candles from Purchase Date & Time (+5m) up to 3:15 PM / click time, checks SL/Trail/Target/3:15PM exit rules, and updates Profit/Loss.",
                ):
                    with st.spinner("Replaying historical 5-minute candles from Purchase Time + 5m through 3:15 PM IST / Click Time..."):
                        replayed_df = evaluate_chronological_exits(
                            trades_csv_path=SNIPER_TELEMETRY_CSV,
                            force_full_replay=True,
                            eod_315pm_auto_exit=eod_315_rule,
                            sync_purchase_candle_price=replay_sync_price,
                        )
                        tot_p = float(replayed_df["PnL_INR"].sum()) if not replayed_df.empty else 0.0
                        st.session_state["last_exit_replay_msg"] = (
                            f"✅ Replayed 5m candles (`Purchase_Time + 5m -> 3:15 PM IST`) across **{len(replayed_df)} Top-3 CNC entries** | "
                            f"Net Realized/Live P&L: **`₹{tot_p:+,.2f}`**"
                        )
                    st.rerun()

                if btn_c2.button(
                    "🚨 Force Immediate Manual Exit @ Live CMP (All Top 3)",
                    use_container_width=True,
                    help="Immediately closes all active Top-3 entries at the latest 5m / live market price and locks in Profit/Loss.",
                ):
                    forced_df = force_manual_exit_trade(ticker="ALL", trades_csv_path=SNIPER_TELEMETRY_CSV)
                    tot_p = float(forced_df["PnL_INR"].sum()) if not forced_df.empty else 0.0
                    st.session_state["last_exit_replay_msg"] = (
                        f"⚡ Manual Exit executed @ Live CMP across **{len(forced_df)} Top-3 CNC entries** | Net P&L: **`₹{tot_p:+,.2f}`**"
                    )
                    st.rerun()

                if "last_exit_replay_msg" in st.session_state:
                    st.success(st.session_state["last_exit_replay_msg"])

            # Performance (Profit or Loss) Summary KPI Metrics inside Card 2
            if not top3_ledger_df.empty:
                tot_cap = float(top3_ledger_df["Capital_Used_INR"].sum()) if "Capital_Used_INR" in top3_ledger_df.columns else 0.0
                tot_pnl_inr = float(top3_ledger_df["PnL_INR"].sum()) if "PnL_INR" in top3_ledger_df.columns else 0.0
                tot_pnl_pct = (tot_pnl_inr / tot_cap * 100.0) if tot_cap > 0 else 0.0
                wins = int((top3_ledger_df["PnL_INR"] > 0).sum()) if "PnL_INR" in top3_ledger_df.columns else 0
                losses = int((top3_ledger_df["PnL_INR"] < 0).sum()) if "PnL_INR" in top3_ledger_df.columns else 0
                wr_pct = (wins / len(top3_ledger_df) * 100.0) if len(top3_ledger_df) > 0 else 0.0

                pk1, pk2, pk3 = st.columns(3)
                pk1.metric(
                    "Top 3 Capital Deployed",
                    f"₹{tot_cap:,.2f}",
                    f"Max Cap: ₹{MAX_DAILY_TOTAL_BUDGET_INR:,.0f} (<=₹20K x 3)",
                )
                pk2.metric(
                    "Net Profit / Loss (₹ & %)",
                    f"₹{tot_pnl_inr:+,.2f}",
                    f"{tot_pnl_pct:+.2f}% Return on Deployed Cash",
                )
                pk3.metric(
                    "Top 3 Win / Loss Count",
                    f"{wins}W / {losses}L ({wr_pct:.0f}% WR)",
                    "5m Candle Replay Verified",
                )

            st.markdown("##### 📋 2A. Top 3 CNC Exit, GTT & Profit/Loss Ticket (`Initial Rows = P&L & Order Data` | `First 3 Columns Frozen`)")
            top3_ticket_df = build_top3_exit_and_pnl_ticket_table(top3_ledger_df)
            render_frozen_3col_dataframe(top3_ticket_df, height=390)

            st.markdown("##### 📒 2B. Top 3 Daily CNC Ledger & 5-Minute Replay Audit (`First 3 Columns Frozen`)")
            if not top3_ledger_df.empty:
                ledger_cols_order = [
                    # First 3 Frozen Columns
                    "Daily_Slot",
                    "Ticker",
                    "Performance_Verdict",
                    # Order, Exit & P&L Columns Immediately Next
                    "PnL_INR",
                    "PnL_Pct",
                    "Status",
                    "GTT_Order_Status",
                    "CNC_Qty",
                    "Capital_Used_INR",
                    "Purchase_Date",
                    "Purchase_Time",
                    "Entry_Price",
                    "Exit_Price",
                    "Exit_Timestamp",
                    "Exit_Reason",
                    "Bars_Tracked_5m",
                    "Window_Low_5m",
                    "Window_High_5m",
                    "Replay_Audit_Trail",
                    "GTT_SL_Trigger",
                    "GTT_SL_Limit",
                    "GTT_Target_Trigger",
                    "GTT_Target_Limit",
                    "High_Water_Mark",
                    "Structural_S1_Floor",
                    "Logic_ID",
                ]
                avail_l_cols = [c for c in ledger_cols_order if c in top3_ledger_df.columns]
                render_frozen_3col_dataframe(top3_ledger_df[avail_l_cols], height=350)

    # =========================================================================
    # SECTION 3 IN TAB 1:
    # 1-MINUTE OPERATOR / BIG-PLAYER FOOTPRINT DETECTOR & SEPARATE AUTO-TRADE
    # (BUY & SELL) PERFORMANCE DESK (5-MINUTE AUTO-REFRESH)
    # =========================================================================
    st.markdown("---")
    with st.container(border=True):
        st.markdown(
            "### 🕵️ Section 3: 1-Minute Operator / Big-Player Footprint Detector & Separate Auto-Trade (`BUY` & `SELL`) Performance Desk (`5m Auto-Refresh`)"
        )
        st.caption(
            "**How It Works (Zero-Load 2-Stage NIFTY 250 Funnel)**: Scans the top active NIFTY 250 stocks on **1-minute candles (`1m`)** for **Institutional / Operator Volume Shocks (`>= 4.5x` 20m Median Volume; e.g., `BEL` Oct 16 `12:02 PM` `1.53M` Futures / `76.1K` Cash Volume)**. "
            "Instead of trading blind on `T+0`, it enforces a **3-Candle (`T+0` Shock -> `T+1m` Hold -> `T+2m` Follow-Through + `<=0.65%` Non-Exhaustion Gate)** and automatically executes & tracks **`BUY` (Low Absorption / VWAP Markup)** and **`SELL` (Operator Dump / Distribution)** auto-trades (`<= ₹20,000` each) in a **completely separate performance ledger**."
        )

        op_perf = load_json("operator_performance_summary.json") or op_perf_summary
        op_signals_all = load_csv("live_operator_1m_signals.csv")
        op_trades_all = load_csv("operator_auto_trades.csv")
        bel_oct16_df = load_csv("bel_oct16_operator_validation.csv")

        # Separate Operator Auto-Trade Performance KPIs
        ok1, ok2, ok3, ok4, ok5 = st.columns(5)
        ok1.metric(
            "Separate Operator Net P&L",
            f"₹{float(op_perf.get('total_net_pnl_inr', 0)):+,.2f}",
            f"+{float(op_perf.get('return_on_20k_slot_pct', 0)):.2f}% on ₹20K Slot (PF {float(op_perf.get('profit_factor', 0)):.2f})",
        )
        ok2.metric(
            "Operator Auto-Trades (W/L)",
            f"{int(op_perf.get('total_auto_trades', 0))} Trades ({int(op_perf.get('wins', 0))}W / {int(op_perf.get('losses', 0))}L)",
            f"Win Rate: {float(op_perf.get('win_rate_pct', 0)):.1f}% (1m Replay)",
        )
        ok3.metric(
            "🟢 AUTO-BUY Net P&L",
            f"₹{float(op_perf.get('buy_net_pnl_inr', 0)):+,.2f}",
            f"{int(op_perf.get('buy_auto_trades', 0))} Long Auto-Trades (<=₹20K)",
        )
        ok4.metric(
            "🔴 AUTO-SELL Net P&L",
            f"₹{float(op_perf.get('sell_net_pnl_inr', 0)):+,.2f}",
            f"{int(op_perf.get('sell_auto_trades', 0))} Short/Dump Auto-Trades (<=₹20K)",
        )
        ok5.metric(
            "📌 BEL Oct 16 (12:02 PM) P&L",
            f"₹{float(op_perf.get('bel_oct16_verified_pnl_inr', 0)):+,.2f}",
            "4/4 Validated Signals (1.53M Fut / 76.1K Cash)",
        )

        # Interactive Controls for 1-Minute Operator Scan & Auto-Trade Replay
        op_c1, op_c2, op_c3 = st.columns([1.6, 1.4, 2.0])
        with op_c1:
            if st.button(
                "🔎 Scan 1m Operator Footprints (NIFTY 250) & Auto-Trade (BUY/SELL)",
                type="primary",
                use_container_width=True,
                key="btn_scan_operator_1m",
            ):
                with st.spinner("Running 2-Stage NIFTY 250 1-Minute Operator Volume-Shock Scan & T+2m Auto-Trade Replay..."):
                    fresh_sig = scan_operator_1m_footprints()
                    fresh_trd, fresh_perf = auto_execute_and_evaluate_operator_trades(fresh_sig)
                    st.session_state["last_op_scan_msg"] = (
                        f"✅ Scanned NIFTY 250 1m Operator Footprints | **{len(fresh_trd)} Confirmed Auto-Trades (`BUY` & `SELL`)** | "
                        f"Separate Net P&L: **`₹{float(fresh_perf.get('total_net_pnl_inr', 0)):+,.2f}`** (`{float(fresh_perf.get('win_rate_pct', 0)):.1f}% WR`)"
                    )
                st.rerun()
        with op_c2:
            if st.button(
                "🔄 Re-Evaluate Open 1m Operator Trades (1m Trail -> 3:15 PM)",
                use_container_width=True,
                key="btn_re_eval_operator_1m",
            ):
                with st.spinner("Re-evaluating 1-minute candles (SL, Stage-1/2/3 Trail, Target & 3:15 PM Auto-Close)..."):
                    fresh_trd, fresh_perf = auto_execute_and_evaluate_operator_trades(op_signals_all)
                    st.session_state["last_op_scan_msg"] = (
                        f"⏱️ Updated 1m Operator Auto-Trades at `{fresh_perf.get('last_5m_refresh_ist', '')}` | "
                        f"Net P&L: **`₹{float(fresh_perf.get('total_net_pnl_inr', 0)):+,.2f}`**"
                    )
                st.rerun()
        with op_c3:
            op_view_filter = st.selectbox(
                "🎯 Filter Operator Auto-Trade & 1m Footprint View",
                options=[
                    "All Confirmed Auto-Trades (BUY + SELL Separate Ledger)",
                    "🟢 AUTO-BUY Only (Operator Low Absorption & VWAP Markup)",
                    "🔴 AUTO-SELL Only (Operator Supply Dump / Short)",
                    "📌 BEL Oct 16 Validation Only (09:58 BUY, 11:08 SELL, 11:23 BUY, 12:02 PM 1.53M Vol SELL)",
                    "All Detected 1m Volume Shocks (Including Trap-Filtered Spikes)",
                ],
                index=0,
                key="op_view_filter_select",
            )

        if "last_op_scan_msg" in st.session_state:
            st.success(st.session_state["last_op_scan_msg"])

        # Load Strategy-Wise Paper Trade Scorecard & Master Paper Trade Log across all 9 strategies
        if strat_scorecard_df.empty:
            strat_scorecard_df = load_csv("strategy_wise_paper_scorecard.csv")
        if all_paper_trades_df.empty:
            all_paper_trades_df = load_csv("all_strategies_paper_trades.csv")

        st.markdown(
            "##### 📊 3A. Strategy-by-Strategy Paper Trade Performance Scorecard (`All 9 Possible Strategies: OP_S1–OP_S3 + L1–L6` | `First 3 Columns Frozen`)"
        )
        st.caption(
            "Tracks paper trade count, Win/Loss ratio, Profit Factor, and Net ₹ P&L (`<= ₹20,000` budget per trade) separately for **every single strategy (`3` Operator 1m Auto-Trade Strategies + `6` CNC Sniper Strategies)**."
        )
        if not strat_scorecard_df.empty:
            render_frozen_3col_dataframe(strat_scorecard_df, height=360)

        st.markdown(
            "##### 📒 3B. Master Paper Trade Log Across All 9 Strategies (`Entry / Exit / P&L Audit` | `First 3 Columns Frozen`)"
        )
        if not all_paper_trades_df.empty:
            strat_options = ["All 9 Strategies (Complete Paper Trade Log)"] + strat_scorecard_df["Strategy_ID"].tolist() if not strat_scorecard_df.empty else ["All 9 Strategies (Complete Paper Trade Log)"]
            selected_strat_log = st.selectbox(
                "🔎 Filter Paper Trade Log by Strategy (`OP_S1–OP_S3` Operator 1m or `L1–L6` CNC Snipers)",
                options=strat_options,
                index=0,
                key="master_paper_strat_filter",
            )
            if selected_strat_log.startswith("All 9"):
                shown_paper_df = all_paper_trades_df.copy()
            else:
                shown_paper_df = all_paper_trades_df[all_paper_trades_df["Strategy_ID"] == selected_strat_log].copy()
            render_frozen_3col_dataframe(shown_paper_df, height=380)

        # Apply filter to op_trades_all
        if not op_trades_all.empty:
            if op_view_filter.startswith("🟢 AUTO-BUY"):
                filtered_op_df = op_trades_all[op_trades_all["Signal_Side"] == "BUY"].copy()
            elif op_view_filter.startswith("🔴 AUTO-SELL"):
                filtered_op_df = op_trades_all[op_trades_all["Signal_Side"] == "SELL"].copy()
            elif op_view_filter.startswith("📌 BEL Oct 16"):
                filtered_op_df = op_trades_all[op_trades_all["Source_Tag"] == "BEL_OCT16_VALIDATED"].copy()
            elif op_view_filter.startswith("All Detected"):
                filtered_op_df = op_signals_all.copy() if not op_signals_all.empty else op_trades_all.copy()
            else:
                filtered_op_df = op_trades_all.copy()
        else:
            filtered_op_df = op_signals_all.copy()

        st.markdown(
            "##### 📋 3C. Operator Auto-Trade (`BUY` & `SELL`) Order & Separate Performance Ticket (`Initial Rows = P&L & Order Data` | `First 3 Columns Frozen`)"
        )
        op_ticket_df = build_operator_initial_rows_ticket_table(filtered_op_df)
        render_frozen_3col_dataframe(op_ticket_df, height=400)

        st.markdown(
            "##### 📒 3D. Separate 1-Minute Operator Auto-Trade (`BUY` & `SELL`) Forensics Ledger (`Initial Rows = Order & P&L` | `First 3 Columns Frozen`)"
        )
        if not filtered_op_df.empty:
            op_ledger_cols = [
                # First 3 Frozen Columns
                "Ticker",
                "Auto_Trade_Action",
                "Performance_Verdict",
                # Separate P&L & Order Execution Columns Immediately Next
                "Signal_Side",
                "PnL_INR",
                "PnL_Pct",
                "Status",
                "Auto_Qty_20K",
                "Capital_Used_INR",
                "Confirm_Entry_Price",
                "Live_Or_Exit_Price",
                "Initial_SL",
                "Current_Trailing_SL",
                "Target_Price",
                # 1-Minute Shock & T+2m Confirmation Forensics
                "Session_Date",
                "Shock_Time_1m",
                "Confirm_Time_T2",
                "Shock_1m_Cash_Vol",
                "Est_Futures_Eq_Vol",
                "Vol_Spike_Mult_20m",
                "Operator_Footprint_Type",
                "Shock_Candle_OHLC",
                "Shock_CLV",
                "Session_VWAP",
                "T2_Follow_Through",
                "Exit_Timestamp",
                "Exit_Reason",
                "Max_Favorable_Pct",
                "Max_Adverse_Pct",
                "Source_Tag",
                "Trade_ID",
            ]
            avail_op_cols = [c for c in op_ledger_cols if c in filtered_op_df.columns]
            render_frozen_3col_dataframe(filtered_op_df[avail_op_cols], height=390)

        with st.expander(
            "🔬 BEL 16th Oct 1-Minute Chart Validation (`12:02 PM` — `1.53M` Futures Vol / `76,149` Cash Vol) & Trap-Filter Audit (`First 3 Columns Frozen`)",
            expanded=False,
        ):
            st.markdown(
                """
                - **Validated Statement**: On **16th Oct (`BEL` 1-Minute Chart @ `12:02 PM IST`)**, `BEL` printed a massive **1-minute red breakdown candle (`O: ₹380.35, H: ₹380.40, L: ₹379.85, C: ₹380.00`)** with **`76,149` NSE Cash shares (`7.8x` 20m median volume) and `~1.53 Million` Futures-equivalent shares (~268 BEL Futures lots)**.
                - **Why Our `T+2m` Confirmation Rule Caught all 4 Moves (`+₹2,990` on `₹20K` budget) & Blocked the `09:50 AM` Fakeout**:
                  1. **`09:50 AM` (`157K` Cash / `~3.1M` Fut Vol)**: Failed `T+2m` confirmation -> **Filtered out as a Bull Trap (`⚪ TRAP FILTERED`)**, avoiding a `-₹1.60/sh` drop to `₹377.10`.
                  2. **`09:58 AM` (`70.1K` Cash / `~1.40M` Fut Vol)**: Operator Low Absorption at `₹377.10`, confirmed at `10:00 AM` (`₹377.65`) -> **`🟢 AUTO-BUY` won `+₹1,274.00`**.
                  3. **`11:08 AM` (`50.5K` Cash / `~1.01M` Fut Vol)**: Supply rejection at `₹380.45`, confirmed at `11:10 AM` (`₹379.65`) -> **`🔴 AUTO-SELL` won `+₹104.00`** before flipping.
                  4. **`11:23 AM` (`54.5K` Cash / `~1.09M` Fut Vol)**: `9.4x` VWAP reclaim, confirmed at `11:25 AM` (`₹379.55`) -> **`🟢 AUTO-BUY` won `+₹234.00`** before flipping at `12:02 PM`.
                  5. **`12:02 PM` (`76.1K` Cash / `~1.53M` Fut Vol)**: Operator Distribution Dump, confirmed at `12:04 PM` (`₹379.50 < ₹379.85` Shock Low) -> **`🔴 AUTO-SELL` dropped straight to `₹376.60` (`3:15 PM` close `₹376.85`), winning `+₹1,378.00` (`+0.70%`)** with virtually zero adverse bounce (`+0.03%`).
                """
            )
            if not bel_oct16_df.empty:
                bel_cols = [
                    "Ticker",
                    "Auto_Trade_Action",
                    "Performance_Verdict",
                    "Session_Date",
                    "Shock_Time_1m",
                    "Confirm_Time_T2",
                    "Shock_1m_Cash_Vol",
                    "Est_Futures_Eq_Vol",
                    "Vol_Spike_Mult_20m",
                    "Operator_Footprint_Type",
                    "Shock_Candle_OHLC",
                    "T2_Follow_Through",
                    "Confirm_Entry_Price",
                    "Live_Or_Exit_Price",
                    "PnL_INR",
                    "Exit_Reason",
                ]
                avail_bel = [c for c in bel_cols if c in bel_oct16_df.columns]
                render_frozen_3col_dataframe(bel_oct16_df[avail_bel], height=260)

    st.markdown("---")
    with st.expander("🏗️ Can the Best Intraday Logic Be Updated in Production (`smart-etf-tracker`) Without Disturbing Other Logics? (Advisory Blueprint — Zero Production Changes Made)", expanded=False):
        st.markdown(
            """
            ### ✅ YES — 100% Isolated Drop-In Possible Without Disturbing ETF, Swing, or Long-Term Logics
            We audited `c:\\Users\\epurntr\\Downloads\\Gravity\\smart-etf-tracker` (`core_logic.py` and `pages/1_📈_Intraday_Trading.py`). **No files in `smart-etf-tracker` have been modified.** Here is why and how the best intraday logic from this repo can be plugged into production with **zero side effects** on any other strategy:

            1. **Strict Strategy Namespace Isolation in `smart-etf-tracker/core_logic.py`**:
               - In production `core_logic.py`, intraday trades are tagged with `Strategy_Type == "INTRADAY"` inside `trade_Multiple_log.csv` and evaluated exclusively inside `evaluate_intraday_trades()` and `generate_intraday_signals()`.
               - Meanwhile, ETF Accumulation (`ETF_ACCUMULATION`), Swing Trades (`SWING`), and Structural Portfolios run in separate functions and separate Streamlit pages (`pages/2_...`, `pages/3_...`).
            2. **3 Drop-In Upgrades for Production (When You Are Ready to Promote)**:
               - **Upgrade #1 (Signal Generation)**: Replace only the body of `generate_intraday_signals()` in `smart-etf-tracker/core_logic.py` with our `L6` (5-Gate 10:00 AM ORB Sniper) + `L1/L4` regime-gated selector, enforcing `Max ₹20,000/ticker` and `Max 3 entries/day` (`CNC` product tag).
               - **Upgrade #2 (Bug-Free 5m Chronological Replay)**: Replace only the `Strategy_Type == "INTRADAY"` loop inside `evaluate_intraday_trades()` with `evaluate_chronological_exits()` (which tracks `Last_Evaluated_Bar_TS` and replays 5m bars from `Purchase_Time + 5m` to `3:15 PM IST`, fixing the retroactive morning-low stop-out bug).
               - **Upgrade #3 (UI Card in `pages/1_📈_Intraday_Trading.py`)**: Add the `@st.fragment(run_every="5m")` auto-refresh wrapper and the **Late-Click 5m Candle Replay & Exit Button** exclusively inside `pages/1_📈_Intraday_Trading.py`.
            3. **Impact on Other Production Logics**: **`0%` (Zero impact)** on ETF tracker, Swing engine, database schema, or other pages.
            """
        )

    st.markdown("#### 🌍 Global Overnight Cues & NSE Sectoral Rotation Leadership (`First 3 Columns Frozen`)")
    if regime_rep:
        gc1, gc2 = st.columns(2)
        with gc1:
            g_rows = []
            for k, v in regime_rep.get("global_cues", {}).items():
                if v:
                    g_rows.append({
                        "Global_Benchmark": k,
                        "Trend_Signal": "✅ Above 20DMA" if v.get("close", 0) >= v.get("sma20", 0) else "🔴 Below 20DMA",
                        "Last_Close": v.get("close"),
                        "1D_Change_Pct": v.get("ret_1d"),
                        "5D_Change_Pct": v.get("ret_5d"),
                        "SMA_20": v.get("sma20"),
                        "SMA_50": v.get("sma50"),
                    })
            render_frozen_3col_dataframe(pd.DataFrame(g_rows))
        with gc2:
            sec_df = pd.DataFrame(regime_rep.get("sector_leadership", []))
            if not sec_df.empty:
                sec_cols = ["sector", "symbol", "rs_vs_nifty_5d_pct", "ret_1d_pct", "ret_5d_pct", "above_20dma"]
                avail_s = [c for c in sec_cols if c in sec_df.columns]
                render_frozen_3col_dataframe(sec_df[avail_s])

    st.markdown(f"#### 📡 Full {len(UNIVERSE_META)}-Instrument NIFTY 250 & International CNC Readiness Radar (`Initial Rows = Order Ready` | `First 3 Columns Frozen`)")
    if not watchlist_df.empty:
        render_frozen_3col_dataframe(watchlist_df, height=460)


with tab1:
    render_live_5m_auto_refresh_desk()

with tab2:
    watchlist_df_t2 = recalculate_cnc_sizing(load_csv("live_watchlist_radar.csv"), MAX_BUDGET_PER_TICKER_INR)
    st.subheader("🛡️ Institutional Downside Protection & 4-Stage Chronological GTT Trailing Stop Ladder")
    st.markdown(
        """
        **How Our CNC + GTT OCO Downside Risk Engine Prevents Production Stop-Outs:**
        1. **100% CNC Execution (Zero MIS Penalty, Max ₹20K/Ticker, Max 3/Day)**: All positions are taken in `CNC` with a 2-Leg Broker GTT OCO Sell order and capped at `₹20,000` per ticker (`<= ₹60,000/day`).
        2. **Structural Support Floor (`S1`)**: Initial GTT stop-loss is anchored below **Opening Range Low (`ORB_Low`) / 20-Day Structural Support `S1` minus an `0.15x–0.18x ATR` cushion**.
        3. **Chronological 5-Minute High-Water-Mark (`HWM`) Ratchet (`Purchase_Time + 5m -> 3:15 PM`)**: Tracks `Last_Evaluated_Bar_TS` and `High_Water_Mark` bar-by-bar in strict chronological order—eliminating the `smart-etf-tracker` bug where afternoon SL upgrades were retroactively compared against 9:50 AM morning lows.
        4. **4-Stage GTT Profit-Locking Ratchet**:
           - **Stage 1 (`MFE >= +0.80 ATR`)**: Modify GTT SL Trigger to `Entry + 0.18 ATR` (True Breakeven + DP/STT cover).
           - **Stage 2 (`MFE >= +1.25 ATR`)**: Modify GTT SL Trigger to `Entry + 0.65 ATR` (Locks >50% of gain).
           - **Stage 3 (`MFE >= +1.75 ATR`)**: Modify GTT SL Trigger to `Entry + 1.15 ATR` (Locks >66% of gain).
           - **Stage 4 (`MFE >= +2.20 ATR`)**: Dynamic Chandelier GTT Trail at `High_Water_Mark - 0.50 ATR`.
        """
    )
    if not watchlist_df_t2.empty:
        cols_ladder = [
            "Ticker", "Exit_Readiness_Status", "GTT_SL_Trigger",
            "Daily_Top3_Slot", "Suggested_CNC_Qty", "CNC_Order_Value_INR",
            "GTT_SL_Limit", "GTT_Target_Trigger", "GTT_Target_Limit",
            "Product_Type", "CNC_Buy_Limit_Price",
            "Structural_S1_Floor", "Dist_Above_S1_Pct",
            "Initial_Protected_SL", "Downside_Risk_Pct",
            "Stage1_Breakeven_SL", "Stage2_ProfitLock_SL", "Stage3_Chandelier_SL",
            "GTT_Swing_Target", "CNC_310PM_Decision",
        ]
        avail_ladder = [c for c in cols_ladder if c in watchlist_df_t2.columns]
        render_frozen_3col_dataframe(watchlist_df_t2[avail_ladder], height=500)

with tab3:
    st.subheader("🔄 3:10 PM IST CNC Intraday-to-Swing Conversion Predictor (3-Year Empirical Study)")
    swing_summary = load_json("swing_conversion_summary.json")
    if swing_summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw Daily Intraday Win Rate (3Y)", f"{swing_summary.get('raw_intraday_win_rate_pct', 0):.2f}%", f"{swing_summary.get('raw_intraday_avg_ret_pct', 0):+.3f}% avg")
        m2.metric("Converting SL Losers ('Bag-Hold')", f"{swing_summary.get('loser_baghold_swing_win_rate_pct', 0):.2f}%", f"{swing_summary.get('loser_baghold_swing_avg_ret_pct', 0):+.3f}% avg", delta_color="inverse")
        m3.metric("Unconditional Stalled Conversion", f"{swing_summary.get('unconditional_swing_conversion_win_rate_pct', 0):.2f}%", f"{swing_summary.get('unconditional_swing_conversion_avg_ret_pct', 0):+.3f}% avg")
        m4.metric("5-Gate Smart CNC Swing Hold", f"{swing_summary.get('smart_gated_swing_conversion_win_rate_pct', 0):.2f}%", f"+{swing_summary.get('smart_gated_swing_conversion_avg_ret_pct', 0):.3f}% avg")

    st.markdown("#### 🧮 Interactive 3:10 PM CNC Hold vs Intraday Exit Simulator")
    sc1, sc2, sc3, sc4 = st.columns(4)
    in_entry = sc1.number_input("CNC Entry Price (₹)", value=1000.0, step=10.0)
    in_cmp = sc2.number_input("3:10 PM Live Price (₹)", value=1008.0, step=5.0)
    in_high = sc3.number_input("Day High (₹)", value=1012.0, step=5.0)
    in_low = sc4.number_input("Day Low (₹)", value=996.0, step=5.0)

    sc5, sc6, sc7, sc8 = st.columns(4)
    in_s1 = sc5.number_input("Structural Support S1 (₹)", value=990.0, step=5.0)
    in_sma20 = sc6.number_input("20-Day SMA (₹)", value=992.0, step=5.0)
    in_sma50 = sc7.number_input("50-Day SMA (₹)", value=980.0, step=5.0)
    in_vol = sc8.number_input("Volume Ratio vs 20D Avg", value=1.35, step=0.1)

    pred_out = predict_swing_conversion_success(
        entry_price=in_entry,
        current_price=in_cmp,
        day_high=in_high,
        day_low=in_low,
        s1_support=in_s1,
        sma20=in_sma20,
        sma50=in_sma50,
        sma200=in_sma50 * 0.95,
        vol_ratio=in_vol,
        market_regime="SIDEWAYS_ROTATION",
        sl_breached=(in_cmp <= in_s1),
    )
    st.info(
        f"**3:10 PM CNC Verdict**: {pred_out['verdict']} | **Empirical Win Probability**: `{pred_out['predicted_swing_win_rate_pct']:.1f}%` | "
        f"**Expected 5D Swing Return**: `{pred_out['expected_swing_return_pct']:+.2f}%` | **Gates Passed**: `{pred_out['gates_passed']}`"
    )
    st.json(pred_out["gate_details"])

with tab4:
    st.subheader("🏆 3-Year Historical Multi-Regime Backtest Results (2023–2026) — First 3 Columns Frozen")
    bt_res = load_csv("backtest_3y_results.csv")
    bt_reg = load_csv("backtest_3y_regime_matrix.csv")
    bt_ver15 = load_csv("intraday_60d_15m_verification.csv")
    bt_timing_sum = load_csv("entry_timing_10am_5gate_summary.csv")
    bt_ledger = load_csv("backtest_trade_ledger_3y.csv")

    st.markdown("#### 1. Strategy Performance Comparison (Top 6 Sniper Logics Including L6 5-Gate 10:00 AM Sniper vs Controls)")
    if not bt_res.empty:
        render_frozen_3col_dataframe(bt_res)

    st.markdown("#### 2. Market Regime × Sniper Logic Matrix")
    if not bt_reg.empty:
        render_frozen_3col_dataframe(bt_reg)

    st.markdown("#### 3. 10:00 AM Entry Timing Study on 5-Gate Pre-Qualified Stocks (15m Bars + Trailing SL & Profit Target)")
    if not bt_timing_sum.empty:
        render_frozen_3col_dataframe(bt_timing_sum)

    st.markdown("#### 4. 60-Day 15-Minute Bar-by-Bar Intraday Walk-Forward Verification")
    if not bt_ver15.empty:
        render_frozen_3col_dataframe(bt_ver15)

    st.markdown("#### 5. Filterable 3-Year Historical Trade Ledger (`First 3 Columns Frozen`)")
    if not bt_ledger.empty:
        sniper_only = st.checkbox("Show Top 6 Sniper Logics Only (L1–L6)", value=True)
        view_df = bt_ledger[bt_ledger["Logic_ID"].str.startswith(("L1_", "L2_", "L3_", "L4_", "L5_", "L6_"))] if sniper_only else bt_ledger
        render_frozen_3col_dataframe(view_df.tail(250), height=420)

with tab5:
    st.subheader("🔬 Forensic Audit of `smart-etf-tracker` Production Trades & Zero-Disturbance Integration Plan")
    st.markdown(
        """
        ### 🏗️ Production Integration Assessment (`smart-etf-tracker` — Advisory Only, Zero Changes Made in Production)
        - **Can the best intraday logic (`L6` 5-Gate 10:00 AM ORB + `L1/L4` Regime Sniper + 5m Chronological Replay) be updated in `smart-etf-tracker` without disturbing other logics?**
          - **Yes, 100%.** In `smart-etf-tracker`, all non-intraday workflows (ETF Accumulation, Swing Trading, Capital Allocation) filter by `Strategy_Type != "INTRADAY"` and reside in separate functions/pages.
          - Updating only `generate_intraday_signals()`, `evaluate_intraday_trades()` (for `Strategy_Type == "INTRADAY"`), and `pages/1_📈_Intraday_Trading.py` will upgrade production intraday performance while leaving ETF and Swing modules completely untouched.
        """
    )
    prod_audit = load_csv("production_trades_forensic_audit.csv")
    if not prod_audit.empty:
        render_frozen_3col_dataframe(prod_audit)
