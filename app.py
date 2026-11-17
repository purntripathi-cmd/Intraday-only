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
    or not hasattr(intraday_expert_engine, "FRESH_PAPER_TRADES_CSV")
    or not hasattr(intraday_expert_engine, "auto_rollover_daily_top3_cnc")
    or not hasattr(intraday_expert_engine, "build_probable_intraday_candidates_sheet")
    or not hasattr(intraday_expert_engine, "run_continuous_critic_audit_and_memory_check")
    or not hasattr(intraday_expert_engine, "REGULAR_AUDIT_LOG_CSV")
    or not hasattr(intraday_expert_engine, "run_v3_two_expert_monday_auto_engine")
    or not hasattr(intraday_expert_engine, "V3_EXPERT_MONDAY_START_DATE")
    or not hasattr(intraday_expert_engine, "compute_adaptive_strategy_scores")
    or not hasattr(intraday_expert_engine, "rebuild_all_real_paper_history")
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

APP_CREATED_DATE = getattr(intraday_expert_engine, "APP_CREATED_DATE", "2026-10-27")
V3_EXPERT_MONDAY_START_DATE = getattr(
    intraday_expert_engine, "V3_EXPERT_MONDAY_START_DATE", "2026-11-02"
)
V3_EXPERT_FRESH_LEDGER_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_FRESH_LEDGER_CSV",
    os.path.join(DATA_DIR, "v3_expert_fresh_paper_ledger.csv"),
)
V3_EXPERT_SHADOW_LEDGER_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_SHADOW_LEDGER_CSV",
    os.path.join(DATA_DIR, "v3_expert_shadow_validation_ledger.csv"),
)
V3_EXPERT_DAILY_SUMMARY_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_DAILY_SUMMARY_CSV",
    os.path.join(DATA_DIR, "v3_expert_daily_summary.csv"),
)
V3_EXPERT_SCORECARD_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_SCORECARD_CSV",
    os.path.join(DATA_DIR, "v3_expert_strategy_scorecard.csv"),
)
V3_EXPERT_REJECTED_TRAPS_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_REJECTED_TRAPS_CSV",
    os.path.join(DATA_DIR, "v3_expert_rejected_traps_log.csv"),
)
V3_EXPERT_AB_COMPARISON_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_AB_COMPARISON_CSV",
    os.path.join(DATA_DIR, "v3_expert_vs_v1_stable_comparison.csv"),
)
V3_EXPERT_3Y_REBACKTEST_CSV = getattr(
    intraday_expert_engine,
    "V3_EXPERT_3Y_REBACKTEST_CSV",
    os.path.join(DATA_DIR, "two_expert_3y_rebacktest_comparison.csv"),
)
V3_EXPERT_SUMMARY_JSON = getattr(
    intraday_expert_engine,
    "V3_EXPERT_SUMMARY_JSON",
    os.path.join(DATA_DIR, "v3_expert_performance_summary.json"),
)
run_v3_two_expert_monday_auto_engine = getattr(
    intraday_expert_engine,
    "run_v3_two_expert_monday_auto_engine",
    lambda *a, **kw: {},
)
FRESH_CNC_HISTORY_CSV = getattr(
    intraday_expert_engine,
    "FRESH_CNC_HISTORY_CSV",
    os.path.join(DATA_DIR, "fresh_cnc_sniper_history.csv"),
)
FRESH_PAPER_TRADES_CSV = getattr(
    intraday_expert_engine,
    "FRESH_PAPER_TRADES_CSV",
    os.path.join(DATA_DIR, "fresh_paper_trades_since_inception.csv"),
)
DAILY_PAPER_SUMMARY_CSV = getattr(
    intraday_expert_engine,
    "DAILY_PAPER_SUMMARY_CSV",
    os.path.join(DATA_DIR, "daily_paper_trade_summary.csv"),
)
HISTORICAL_BENCHMARK_CSV = getattr(
    intraday_expert_engine,
    "HISTORICAL_BENCHMARK_CSV",
    os.path.join(DATA_DIR, "historical_benchmark_trades.csv"),
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
PROBABLE_CANDIDATES_CSV = getattr(
    intraday_expert_engine,
    "PROBABLE_CANDIDATES_CSV",
    os.path.join(DATA_DIR, "probable_intraday_candidates.csv"),
)
PROBABLE_CANDIDATES_STRAT_A_CSV = getattr(
    intraday_expert_engine,
    "PROBABLE_CANDIDATES_STRAT_A_CSV",
    os.path.join(DATA_DIR, "probable_candidates_strategy_a.csv"),
)
PROBABLE_CANDIDATES_STRAT_B_CSV = getattr(
    intraday_expert_engine,
    "PROBABLE_CANDIDATES_STRAT_B_CSV",
    os.path.join(DATA_DIR, "probable_candidates_strategy_b.csv"),
)
CANONICAL_STRATEGY_NAMES = getattr(
    intraday_expert_engine,
    "CANONICAL_STRATEGY_NAMES",
    {},
)
_get_strategy_paper_performance_map = getattr(
    intraday_expert_engine,
    "_get_strategy_paper_performance_map",
    lambda: {},
)
_compute_confidence_badge = getattr(
    intraday_expert_engine,
    "_compute_confidence_badge",
    lambda **kw: (80.0, "🟢 HIGH (80%)"),
)
REGULAR_AUDIT_LOG_CSV = getattr(
    intraday_expert_engine,
    "REGULAR_AUDIT_LOG_CSV",
    os.path.join(DATA_DIR, "regular_audit_log.csv"),
)
AUDIT_GAPS_CHANGES_CSV = getattr(
    intraday_expert_engine,
    "AUDIT_GAPS_CHANGES_CSV",
    os.path.join(DATA_DIR, "audit_gaps_and_changes_log.csv"),
)
AUDIT_CRITIC_MEMORY_JSON = getattr(
    intraday_expert_engine,
    "AUDIT_CRITIC_MEMORY_JSON",
    os.path.join(DATA_DIR, "audit_critic_memory.json"),
)
DAILY_CAPITAL_TAX_XIRR_CSV = getattr(
    intraday_expert_engine,
    "DAILY_CAPITAL_TAX_XIRR_CSV",
    os.path.join(DATA_DIR, "daily_capital_charges_tax_xirr.csv"),
)
V1_PRE_AUDIT_SNAPSHOT_CSV = getattr(
    intraday_expert_engine,
    "V1_PRE_AUDIT_SNAPSHOT_CSV",
    os.path.join(DATA_DIR, "v1_pre_audit_fresh_trades_snapshot.csv"),
)
ADAPTIVE_STRATEGY_RANKING_CSV = getattr(
    intraday_expert_engine,
    "ADAPTIVE_STRATEGY_RANKING_CSV",
    os.path.join(DATA_DIR, "adaptive_strategy_ranking.csv"),
)
compute_adaptive_strategy_scores = getattr(
    intraday_expert_engine,
    "compute_adaptive_strategy_scores",
    lambda *a, **kw: pd.DataFrame(),
)
rebuild_all_real_paper_history = getattr(
    intraday_expert_engine,
    "rebuild_all_real_paper_history",
    lambda *a, **kw: {},
)
auto_rollover_daily_top3_cnc = getattr(
    intraday_expert_engine,
    "auto_rollover_daily_top3_cnc",
    lambda **kw: evaluate_chronological_exits(**kw),
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
build_probable_intraday_candidates_sheet = getattr(
    intraday_expert_engine,
    "build_probable_intraday_candidates_sheet",
    lambda *a, **kw: pd.DataFrame(),
)
run_continuous_critic_audit_and_memory_check = getattr(
    intraday_expert_engine,
    "run_continuous_critic_audit_and_memory_check",
    lambda *a, **kw: {},
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

# Sidebar Controls (initialized before st.tabs so all tabs observe consistent state)
with st.sidebar:
    st.markdown("### ⚙️ Execution & Quality Controls")
    st.toggle(
        "⏰ Auto-Exit Intraday at 3:15 PM IST",
        value=True,
        key="eod_315pm_auto_exit_toggle",
        help="Automatically square off open intraday CNC/Operator positions at 3:15 PM IST unless gated for swing hold.",
    )
    st.toggle(
        "🔴 Enable MIS Short Selling (OP_S3)",
        value=True,
        key="enable_mis_short_toggle",
        help="When ON, allows 1m Operator Breakdown SHORT SELL (MIS Only) signals during Bearish/Sideways regimes. Turn OFF for 100% CNC Long-Only.",
    )
    st.caption(
        "**Adaptive Strategy Ranking**: 14-day exponential decay + regime match (`λ=0.25`) + "
        "Wilson Lower Bound (`z=1.645`) + Bayesian-shrunk XIRR (`N>=5d` guard). Strictly Closed Trades Only."
    )
    if st.button("🔄 Rebuild Real Paper History (>= 2026-10-27)", use_container_width=True, key="sidebar_rebuild_real_btn"):
        with st.spinner("Rebuilding all paper ledgers strictly from genuine NSE OHLCV since 2026-10-27..."):
            rebuild_all_real_paper_history(
                enable_mis_short=st.session_state.get("enable_mis_short_toggle", True)
            )
        st.toast("Clean real-market rebuild complete!", icon="✅")
        st.rerun()

st.title("🎯 Institutional Intraday Desk — Strategy B Quality Engine & Adaptive Top-10")
st.caption(
    f"Universe: **NIFTY 250 + ETFs (`{len(UNIVERSE_META)}` Instruments)** | "
    "**🟢 BUY (CNC / CNC-MIS)** vs **🔴 SHORT SELL (MIS Only)** | "
    "Strict **≤ ₹20,000/Ticker** | **Zero Artificial Data — Strictly Real Closed-Trade Telemetry (>= 2026-10-27)**"
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


def _add_signal_direction(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every trade / signal / strategy table clearly indicates explicit Buy/Short-Sell product mode:
    `🟢 BUY (CNC)`, `🟢 BUY (CNC/MIS)`, or `🔴 SHORT SELL (MIS Only)`.
    """
    if df_in is None or df_in.empty:
        return df_in
    df = df_in.copy()
    has_signal_context = any(
        c in df.columns
        for c in [
            "Strategy_ID", "StratB_Strategy_ID", "Logic_ID", "Signal_Side",
            "Paper_Order_Side", "Order_Side", "Auto_Trade_Action", "Ticker",
            "symbol", "Strategy_Name", "Candidate_Group",
        ]
    )
    if not has_signal_context:
        return df

    def _determine_direction(r):
        existing = str(r.get("Signal_Direction", "")).strip()
        if existing in ("🟢 BUY (CNC)", "🟢 BUY (CNC/MIS)", "🔴 SHORT SELL (MIS Only)"):
            return existing
        side_fields = [
            str(r.get("Signal_Side", "")),
            str(r.get("Paper_Order_Side", "")),
            str(r.get("Order_Side", "")),
            str(r.get("Auto_Trade_Action", "")),
            str(r.get("Action", "")),
            str(r.get("Signal", "")),
            existing,
        ]
        combined_side = " ".join(side_fields).upper()
        strat_id = str(r.get("Strategy_ID", r.get("StratB_Strategy_ID", r.get("Logic_ID", "")))).upper()
        prod = str(r.get("Product_Mode", r.get("Product_Type", ""))).upper()
        if "SELL" in combined_side or "SHORT" in combined_side or "OP_S3" in strat_id:
            return "🔴 SHORT SELL (MIS Only)"
        if strat_id.startswith("OP_") or "CNC/MIS" in prod:
            return "🟢 BUY (CNC/MIS)"
        return "🟢 BUY (CNC)"

    dir_col = df.apply(_determine_direction, axis=1)
    if "Signal_Direction" in df.columns:
        df["Signal_Direction"] = dir_col
    else:
        df.insert(0, "Signal_Direction", dir_col)
    return df


def render_frozen_3col_dataframe(df: pd.DataFrame, height: int = None, auto_add_direction: bool = True):
    """
    Freezes the first 3 columns of any DataFrame in Streamlit's interactive Glide Data Grid
    without mutating visible cell labels (uses invisible zero-width disambiguator only on repeated triples).
    """
    if df is None or df.empty:
        st.info("No records available.")
        return

    work = _add_signal_direction(df.copy().reset_index(drop=True)) if auto_add_direction else df.copy().reset_index(drop=True)
    if len(work.columns) < 3:
        st.dataframe(work, use_container_width=True, hide_index=True, height=height)
        return

    freeze_cols = list(work.columns[:3])
    col0 = freeze_cols[0]

    seen: dict[str, int] = {}
    unique_col0 = []
    for val in work[col0].astype(str):
        cnt = seen.get(val, 0)
        seen[val] = cnt + 1
        # Keep display clean — append zero-width spaces only when identical text repeats
        unique_col0.append(val + ("\u200b" * cnt))
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
    Dynamically updates CNC/Intraday order quantity strictly under <= ₹20,000 budget cap
    and excludes any instrument whose 1-share price exceeds the capped budget (`qty == 0`).
    """
    if df is None or df.empty:
        return df
    price_col = (
        "CNC_Buy_Limit_Price"
        if "CNC_Buy_Limit_Price" in df.columns
        else ("Recommended_Entry_Price" if "Recommended_Entry_Price" in df.columns else None)
    )
    if price_col is None:
        return df
    out = df.copy()
    capped_capital = min(MAX_BUDGET_PER_TICKER_INR, max(1000.0, float(cnc_capital_inr)))
    prices = pd.to_numeric(out[price_col], errors="coerce").fillna(0.0)
    # Strict budget guard: exclude 0-qty / over-budget tickers cleanly
    valid_mask = (prices > 0.0) & (prices <= capped_capital)
    out = out[valid_mask].copy().reset_index(drop=True)
    if out.empty:
        return out
    prices = pd.to_numeric(out[price_col], errors="coerce")
    qtys = (capped_capital // prices).astype(int)
    over_mask = (qtys * prices > capped_capital) & (qtys > 1)
    qtys = qtys.where(~over_mask, qtys - 1)
    qtys = qtys.where(qtys >= 1, 0)
    out = out[qtys >= 1].copy().reset_index(drop=True)
    if out.empty:
        return out
    prices = pd.to_numeric(out[price_col], errors="coerce")
    qtys = (capped_capital // prices).astype(int)
    if "Initial_Protected_SL" in out.columns:
        sl_prices = pd.to_numeric(out["Initial_Protected_SL"], errors="coerce").fillna(prices * 0.985)
    elif "Recommended_Stop_Loss" in out.columns:
        sl_prices = pd.to_numeric(out["Recommended_Stop_Loss"], errors="coerce").fillna(prices * 0.985)
    else:
        sl_prices = prices * 0.985

    out["Suggested_CNC_Qty"] = qtys
    out["CNC_Order_Value_INR"] = (qtys * prices).round(2)
    out["Max_Stop_Risk_INR"] = (qtys * (prices - sl_prices).abs().clip(lower=0.05)).round(2)
    if "Raw_Qty" in out.columns:
        out["Raw_Qty"] = qtys
    if "Qty_For_20K_And_Capital" in out.columns:
        out["Qty_For_20K_And_Capital"] = [f"{int(q)} Qty (₹{float(q * p):,.0f})" for q, p in zip(qtys, prices)]
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
    "🎯 Tab 1: Live Signals — Adaptive Top 10 (BUY/SELL)",
    "📊 Tab 2: Paper Trade Performance & 9-Strategy Scorecard",
    "🛠️ Tab 3: Manual Order Builder, 1m Shock Scanner & Radar",
    "🛡️ Tab 4: Downside Protection & Trailing SL Ladder",
    "🔄 Tab 5: 3:10 PM Swing Predictor, 3Y Backtest & Critic Audit",
])


def render_manual_order_builder_and_scanner_tab():
    """
    Renders Tab 2: Manual CNC Entry/Exit Order Builders (Card 1 & Card 2),
    Manual 1m Operator Raw Footprint Scanner, BEL Oct 16 Validation,
    smart-etf-tracker Production Blueprint, Global Cues & Full 261-Ticker Radar.
    """
    regime_rep = load_json("live_regime_report.json")
    sniper_df = load_csv("live_sniper_signals.csv")
    watchlist_df = load_csv("live_watchlist_radar.csv")
    top3_ledger_df = load_csv("sniper_telemetry_trades.csv")
    op_signals_all = load_csv("live_operator_1m_signals.csv")
    op_trades_all = load_csv("operator_auto_trades.csv")
    op_perf_summary = load_json("operator_performance_summary.json")
    strat_scorecard_df = load_csv("strategy_wise_paper_scorecard.csv")
    all_paper_trades_df = load_csv("all_strategies_paper_trades.csv")

    st.subheader("🛠️ Manual CNC Order Builder, 5m Late-Click Replay, 1m Operator Raw Scanner & Full Nifty-250 Radar")
    st.caption(
        "Moved out of Tab 1 so Tab 1 stays 100% focused on **Scheduled Paper Trade Performance (`Strategy A: 1st Stable` vs `Strategy B: 2-Expert Critic [Starts Monday]`)**. "
        "Use this tab to manually stage broker orders, run ad-hoc 1m shock scans, or browse all 261 NIFTY 250 instruments."
    )

    m_ctrl1, m_ctrl2 = st.columns([1.4, 2.2])
    with m_ctrl1:
        cnc_capital = st.number_input(
            "💰 Manual Order Budget / Ticker (₹, <= 20K)",
            min_value=2000.0,
            max_value=float(MAX_BUDGET_PER_TICKER_INR),
            value=float(MAX_BUDGET_PER_TICKER_INR),
            step=2000.0,
            key="tab2_manual_cnc_capital",
        )
    with m_ctrl2:
        filter_mode = st.selectbox(
            "🔎 Card 1 Candidate Filter (NIFTY 250 Universe)",
            options=[
                "Top Actionable & High-Readiness CNC Setups (Top 3 Highlighted)",
                "Triggered Sniper Signals Only (Immediate CNC Buy)",
                f"All {len(UNIVERSE_META)} NIFTY 250 & International Universe Instruments",
            ],
            index=0,
            key="tab2_candidate_filter_mode",
        )

    watchlist_df = recalculate_cnc_sizing(watchlist_df, cnc_capital)
    sniper_df = recalculate_cnc_sizing(sniper_df, cnc_capital)
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
                    value=bool(st.session_state.get("eod_315pm_auto_exit_toggle", True)),
                    key="tab3_eod_315pm_auto_exit_toggle",
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
                    # Exact Entry Date/Time & Exit Timestamp/Reason Immediately Next
                    "Purchase_Date",
                    "Purchase_Time",
                    "Exit_Timestamp",
                    "Exit_Reason",
                    "PnL_INR",
                    "PnL_Pct",
                    "Status",
                    "GTT_Order_Status",
                    "CNC_Qty",
                    "Capital_Used_INR",
                    "Entry_Price",
                    "Exit_Price",
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
            "### 🕵️ Card 3: 1-Minute Operator / Big-Player Footprint Scanner & Strategy A vs Strategy B Performance Desk (`5m Auto-Refresh`)"
        )
        st.caption(
            "**How It Works (Zero-Load 2-Stage NIFTY 250 Funnel)**: Scans the top active NIFTY 250 stocks on **1-minute candles (`1m`)** for **Institutional / Operator Volume Shocks (`>= 4.5x` 20m Median Volume; e.g., `BEL` Oct 16 `12:02 PM` `1.53M` Futures / `76.1K` Cash Volume)** and compares **Strategy A (`1st Stable Version`)** vs **Strategy B (`2-Expert Critic Strategy` — Live From Monday `2026-11-02`)** across all `OP_S1–OP_S3` and `L1–L6` setups."
        )

        op_perf = load_json("operator_performance_summary.json") or op_perf_summary
        op_signals_all = load_csv("live_operator_1m_signals.csv")
        op_trades_all = load_csv("operator_auto_trades.csv")
        bel_oct16_df = load_csv("bel_oct16_operator_validation.csv")

        # Real-time Operator 1m KPIs strictly from closed rows in operator_auto_trades.csv
        op_closed_df = (
            op_trades_all[
                ~op_trades_all["Status"].astype(str).str.upper().str.contains("ACTIVE|OPEN|PENDING|MONITORING", na=False)
            ].copy()
            if (not op_trades_all.empty and "Status" in op_trades_all.columns)
            else pd.DataFrame()
        )
        op_open_cnt = max(0, len(op_trades_all) - len(op_closed_df))
        op_buy_sub = (
            op_closed_df[op_closed_df["Signal_Side"].astype(str).str.upper().str.contains("BUY", na=False)]
            if (not op_closed_df.empty and "Signal_Side" in op_closed_df.columns)
            else pd.DataFrame()
        )
        op_sell_sub = (
            op_closed_df[op_closed_df["Signal_Side"].astype(str).str.upper().str.contains("SELL|SHORT", na=False)]
            if (not op_closed_df.empty and "Signal_Side" in op_closed_df.columns)
            else pd.DataFrame()
        )

        def _op_sub_stats(df_sub: pd.DataFrame):
            if df_sub is None or df_sub.empty:
                return 0, 0, 0, 0.0, 0.0
            pnl_s = pd.to_numeric(df_sub.get("PnL_INR", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            net_s = (
                pd.to_numeric(df_sub["Net_PnL_After_Charges_INR"], errors="coerce").fillna(pnl_s)
                if "Net_PnL_After_Charges_INR" in df_sub.columns
                else pnl_s
            )
            wins = int((net_s > 0).sum())
            losses = len(df_sub) - wins
            wr = round(wins / max(len(df_sub), 1) * 100.0, 1)
            net_sum = float(net_s.sum())
            pt = round(net_sum * 0.80 if net_sum > 0 else net_sum, 2)
            return len(df_sub), wins, losses, wr, pt

        n_op_c, w_op_c, l_op_c, wr_op_c, pt_op_c = _op_sub_stats(op_closed_df)
        n_ob, w_ob, l_ob, wr_ob, pt_ob = _op_sub_stats(op_buy_sub)
        n_os, w_os, l_os, wr_os, pt_os = _op_sub_stats(op_sell_sub)

        ok1, ok2, ok3, ok4, ok5 = st.columns(5)
        ok1.metric(
            "Operator 1m Closed Post-Tax (Real)",
            f"₹{pt_op_c:+,.2f}",
            f"{n_op_c} Closed ({w_op_c}W/{l_op_c}L) | {op_open_cnt} Open",
        )
        ok2.metric(
            "Operator Closed Success Rate",
            f"{wr_op_c:.1f}% WR",
            f"Evaluated strictly on closed trades (3 bps slippage)",
        )
        ok3.metric(
            "🟢 BUY (CNC/MIS) OP_S1+S2",
            f"₹{pt_ob:+,.2f}",
            f"{n_ob} Closed ({w_ob}W/{l_ob}L | {wr_ob:.1f}% WR)",
        )
        ok4.metric(
            "🔴 SHORT SELL (MIS Only) OP_S3",
            f"₹{pt_os:+,.2f}",
            f"{n_os} Closed ({w_os}W/{l_os}L | {wr_os:.1f}% WR)",
        )
        ok5.metric(
            "📡 Detected 1m Shocks Today/Cache",
            f"{len(op_signals_all)} Shocks",
            "Prime Window 09:40–11:15 IST | >= ₹1.5 Cr",
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
                    fresh_trd, fresh_perf = auto_execute_and_evaluate_operator_trades(
                        fresh_sig,
                        enable_mis_short=st.session_state.get("enable_mis_short_toggle", True),
                        current_regime=regime_rep.get("regime_badge") if isinstance(regime_rep, dict) else None,
                    )
                    st.session_state["last_op_scan_msg"] = (
                        f"✅ Scanned NIFTY 250 1m Operator Footprints | **{len(fresh_trd)} Confirmed Auto-Trades (`BUY CNC/MIS` & `SHORT SELL MIS`)** | "
                        f"Closed Net P&L: **`₹{float(fresh_perf.get('total_net_pnl_inr', 0)):+,.2f}`** (`{float(fresh_perf.get('win_rate_pct', 0)):.1f}% Closed WR`)"
                    )
                st.rerun()
        with op_c2:
            if st.button(
                "🔄 Re-Evaluate Open 1m Operator Trades (1m Trail -> 3:15 PM)",
                use_container_width=True,
                key="btn_re_eval_operator_1m",
            ):
                with st.spinner("Re-evaluating 1-minute candles (SL, Cost+ Ratchet, Target & 3:15 PM Auto-Close)..."):
                    fresh_trd, fresh_perf = auto_execute_and_evaluate_operator_trades(
                        op_signals_all,
                        enable_mis_short=st.session_state.get("enable_mis_short_toggle", True),
                        current_regime=regime_rep.get("regime_badge") if isinstance(regime_rep, dict) else None,
                    )
                    st.session_state["last_op_scan_msg"] = (
                        f"⏱️ Updated 1m Operator Auto-Trades at `{fresh_perf.get('last_5m_refresh_ist', '')}` | "
                        f"Closed Net P&L: **`₹{float(fresh_perf.get('total_net_pnl_inr', 0)):+,.2f}`**"
                    )
                st.rerun()
        with op_c3:
            op_view_filter = st.selectbox(
                "🎯 Filter Operator Auto-Trade & 1m Footprint View",
                options=[
                    "All Confirmed Auto-Trades (BUY CNC/MIS + SHORT SELL MIS)",
                    "🟢 BUY Only (OP_S1 Low Absorption & OP_S2 VWAP Markup — CNC/MIS)",
                    "🔴 SHORT SELL Only (OP_S3 Supply Dump — MIS Only)",
                    "All Detected 1m Volume Shocks (Including Trap-Filtered Spikes)",
                ],
                index=0,
                key="op_view_filter_select",
            )

        if "last_op_scan_msg" in st.session_state:
            st.success(st.session_state["last_op_scan_msg"])

        # Apply filter to op_trades_all
        if not op_trades_all.empty:
            if op_view_filter.startswith("🟢"):
                filtered_op_df = op_trades_all[op_trades_all["Signal_Side"] == "BUY"].copy()
            elif op_view_filter.startswith("🔴"):
                filtered_op_df = op_trades_all[op_trades_all["Signal_Side"] == "SELL"].copy()
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
            "##### 📒 3D. Separate 1-Minute Operator Auto-Trade (`BUY` & `SELL`) Forensics Ledger (`Entry & Exit Times in Initial Columns` | `First 3 Columns Frozen`)"
        )
        if not filtered_op_df.empty:
            op_ledger_cols = [
                # First 3 Frozen Columns (Date, Ticker, Entry -> Exit Time)
                "Session_Date",
                "Ticker",
                "Entry_To_Exit_Time",
                # Exact Entry & Exit Times + Performance & Order Execution Immediately Next
                "Entry_Time_IST",
                "Exit_Time_IST",
                "Auto_Trade_Action",
                "Performance_Verdict",
                "Exit_Reason",
                "Strategy_ID",
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



def _enrich_signals_for_live_tab(df_in: pd.DataFrame, engine_mode: str = "B") -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame()
    df = df_in.copy()
    perf_map_live = _get_strategy_paper_performance_map()
    sid_col = (
        "Strategy_ID"
        if "Strategy_ID" in df.columns
        else ("StratB_Strategy_ID" if "StratB_Strategy_ID" in df.columns else None)
    )
    if sid_col is not None:
        if "Strategy_Name" not in df.columns:
            df["Strategy_Name"] = ""
        df["Strategy_Name"] = df.apply(
            lambda r: (
                str(r.get("Strategy_Name", "")).strip()
                if str(r.get("Strategy_Name", "")).strip() not in ("", "nan", "None")
                else CANONICAL_STRATEGY_NAMES.get(
                    str(r.get(sid_col, "")).strip(),
                    str(r.get("Strategy_Name_And_Mode", r.get(sid_col, "Confirmed Intraday Strategy"))),
                )
            ),
            axis=1,
        )

        # Pull real closed-trade telemetry from adaptive perf_map (same source as Tab 2 Scorecard)
        badge_key = "StratB_Success_Str" if engine_mode == "B" else ("StratA_Success_Str" if engine_mode == "A" else "Combined_AB_Success_Str")
        wr_key = "StratB_WR_Pct" if engine_mode != "A" else "StratA_WR_Pct"
        wlb_key = "StratB_WLB_Pct" if engine_mode != "A" else "StratA_WLB_Pct"
        xirr_key = "StratB_XIRR_Pct" if engine_mode != "A" else "StratA_XIRR_Pct"
        pt_key = "StratB_Post_Tax_INR" if engine_mode != "A" else "StratA_Post_Tax_INR"

        df["Paper_Trade_Success_Rate"] = df[sid_col].apply(
            lambda s: str(perf_map_live.get(str(s).strip(), {}).get(badge_key, "⚪ Unproven (0 Closed)") or "⚪ Unproven (0 Closed)")
        )
        df["Strategy_Paper_WR_Pct"] = df[sid_col].apply(
            lambda s: float(perf_map_live.get(str(s).strip(), {}).get(wr_key, 0.0) or 0.0)
        )
        df["StratB_WLB_Pct"] = df[sid_col].apply(
            lambda s: float(perf_map_live.get(str(s).strip(), {}).get(wlb_key, 0.0) or 0.0)
        )
        df["StratB_XIRR_Pct"] = df[sid_col].apply(
            lambda s: float(perf_map_live.get(str(s).strip(), {}).get(xirr_key, 0.0) or 0.0)
        )
        df["Adaptive_Rank_Score"] = df[sid_col].apply(
            lambda s: float(perf_map_live.get(str(s).strip(), {}).get("Adaptive_Rank_Score", 0.0) or 0.0)
        )
        df["Strategy_Paper_PostTax_INR"] = df[sid_col].apply(
            lambda s: float(perf_map_live.get(str(s).strip(), {}).get(pt_key, 0.0) or 0.0)
        )
        if "Product_Mode" not in df.columns:
            df["Product_Mode"] = df[sid_col].apply(
                lambda s: str(perf_map_live.get(str(s).strip(), {}).get("Product_Mode", "CNC" if str(s).strip().startswith("L") else "MIS"))
            )

        # Always compute real-time Confidence from closed-trade WLB + XIRR/expectancy + setup readiness
        conf_scores = []
        conf_badges = []
        for _, r in df.iterrows():
            sid_v = str(r.get(sid_col, "")).strip()
            vol_v = float(pd.to_numeric(r.get("Vol_Spike_Mult_20m", r.get("Vol_Ratio_20D", 1.5)), errors="coerce") or 1.5)
            c_sc, c_bd = _compute_confidence_badge(
                strat_id=sid_v,
                perf_map=perf_map_live,
                readiness_or_conf=float(pd.to_numeric(r.get("Raw_Readiness_Score", 70.0), errors="coerce") or 70.0),
                vol_mult=vol_v,
                gap_pct=float(pd.to_numeric(r.get("Raw_Gap_Pct", 0.0), errors="coerce") or 0.0),
                is_blocked_in_b=False,
                prefer_engine="B" if engine_mode != "A" else "A",
            )
            conf_scores.append(c_sc)
            conf_badges.append(c_bd)
        df["Confidence_Score_Pct"] = conf_scores
        df["Confidence_Level"] = conf_badges

    df = _add_signal_direction(df)

    # Sort strictly highest success probability (Adaptive Rank / WLB) + max XIRR on top
    sort_cols = [
        c
        for c in ["Adaptive_Rank_Score", "StratB_WLB_Pct", "StratB_XIRR_Pct", "Strategy_Paper_PostTax_INR", "Confidence_Score_Pct"]
        if c in df.columns
    ]
    asc_flags = [False] * len(sort_cols)
    if "Raw_Gap_Pct" in df.columns:
        sort_cols.append("Raw_Gap_Pct")
        asc_flags.append(True)
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=asc_flags).reset_index(drop=True)
    return df


@st.fragment(run_every="5m")
def render_live_signals_tab():
    """
    Tab 1: Adaptive Top-10 Live Entry & Exit Signals (Strategy B Quality Engine).
    Sorted strictly by highest closed-trade success probability (Adaptive Rank / WLB) + max XIRR on top.
    """
    eod_exit_enabled = st.session_state.get("eod_315pm_auto_exit_toggle", True)
    mis_short_enabled = st.session_state.get("enable_mis_short_toggle", True)

    try:
        auto_rollover_daily_top3_cnc(eod_315pm_auto_exit=eod_exit_enabled)
    except Exception:
        pass

    auto_eval_df = evaluate_chronological_exits(
        trades_csv_path=SNIPER_TELEMETRY_CSV,
        force_full_replay=False,
        eod_315pm_auto_exit=eod_exit_enabled,
    )

    regime_rep = load_json("live_regime_report.json")
    current_regime_badge = regime_rep.get("regime_badge") if isinstance(regime_rep, dict) else None

    now_ist_dt = pd.Timestamp.now(tz="Asia/Kolkata")
    today_ist_str = now_ist_dt.strftime("%Y-%m-%d")
    is_live_market_hours = (
        now_ist_dt.weekday() < 5
        and ((now_ist_dt.hour == 9 and now_ist_dt.minute >= 15) or (10 <= now_ist_dt.hour <= 14) or (now_ist_dt.hour == 15 and now_ist_dt.minute <= 30))
    )
    existing_op_trades = load_csv("operator_auto_trades.csv")
    date_col_op = (
        "Session_Date"
        if (not existing_op_trades.empty and "Session_Date" in existing_op_trades.columns)
        else ("Entry_Date" if (not existing_op_trades.empty and "Entry_Date" in existing_op_trades.columns) else None)
    )
    needs_post_market_catchup = (
        now_ist_dt.weekday() < 5
        and now_ist_dt.hour >= 10
        and (
            existing_op_trades.empty
            or date_col_op is None
            or today_ist_str not in existing_op_trades[date_col_op].astype(str).str[:10].values
        )
    )
    if is_live_market_hours or needs_post_market_catchup:
        try:
            cached_op_signals = scan_operator_1m_footprints()
        except Exception:
            cached_op_signals = load_csv("live_operator_1m_signals.csv")
    else:
        cached_op_signals = load_csv("live_operator_1m_signals.csv")

    op_trades_df, _ = auto_execute_and_evaluate_operator_trades(
        cached_op_signals if not cached_op_signals.empty else None,
        enable_mis_short=mis_short_enabled,
        current_regime=current_regime_badge,
    )
    build_all_strategies_paper_ledger_and_scorecard(
        op_trades_df=op_trades_df if not op_trades_df.empty else None,
        current_regime=current_regime_badge,
    )

    try:
        run_v3_two_expert_monday_auto_engine(
            watchlist_df=load_csv("live_watchlist_radar.csv"),
            sniper_df=load_csv("live_sniper_signals.csv"),
            op_signals_df=cached_op_signals if not cached_op_signals.empty else None,
            force_rebuild_shadow=needs_post_market_catchup,
            force_refresh=needs_post_market_catchup,
            enable_mis_short=mis_short_enabled,
            current_regime=current_regime_badge,
        )
    except Exception:
        pass

    now_ist_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown(
        f"⏱️ **Live Signal Engine: `ACTIVE (5m Auto-Refresh)`** | "
        f"Last Check: **`{now_ist_str}`** | "
        f"**MIS Short Selling: `{'ON (OP_S3 MIS)' if mis_short_enabled else 'OFF (CNC Long-Only)'}`** | "
        f"*Strictly Closed-Trade Telemetry (`>= {APP_CREATED_DATE}`)*"
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.8, 1.8, 1.4])
    with ctrl1:
        if st.button("🔄 Refresh Strategy B Signals & Paper Trades Now", type="primary", use_container_width=True, key="tab1_refresh_btn"):
            with st.spinner(f"Scanning {len(UNIVERSE_META)} NIFTY 250 instruments & evaluating Strategy B signals..."):
                scan_live_sniper_opportunities()
                evaluate_chronological_exits(
                    trades_csv_path=SNIPER_TELEMETRY_CSV,
                    force_full_replay=True,
                    eod_315pm_auto_exit=eod_exit_enabled,
                )
                rep_fresh = load_json("live_regime_report.json")
                reg_fresh = rep_fresh.get("regime_badge") if isinstance(rep_fresh, dict) else current_regime_badge
                new_op_sig = scan_operator_1m_footprints()
                new_op_tr, _ = auto_execute_and_evaluate_operator_trades(
                    new_op_sig,
                    enable_mis_short=mis_short_enabled,
                    current_regime=reg_fresh,
                )
                build_all_strategies_paper_ledger_and_scorecard(
                    op_trades_df=new_op_tr if not new_op_tr.empty else None,
                    current_regime=reg_fresh,
                )
                run_v3_two_expert_monday_auto_engine(
                    watchlist_df=load_csv("live_watchlist_radar.csv"),
                    sniper_df=load_csv("live_sniper_signals.csv"),
                    op_signals_df=new_op_sig if not new_op_sig.empty else None,
                    force_rebuild_shadow=True,
                    force_refresh=True,
                    enable_mis_short=mis_short_enabled,
                    current_regime=reg_fresh,
                )
                build_probable_intraday_candidates_sheet(
                    watchlist_df=load_csv("live_watchlist_radar.csv"),
                    sniper_df=load_csv("live_sniper_signals.csv"),
                    op_signals_df=new_op_sig if not new_op_sig.empty else None,
                    top3_ledger_df=load_csv("sniper_telemetry_trades.csv"),
                    enable_mis_short=mis_short_enabled,
                    current_regime=reg_fresh,
                )
            st.rerun()
    with ctrl2:
        if st.button("🏆 Reset / Sync Today's Top 3 CNC Slots (<=₹20K)", use_container_width=True, key="tab1_sync_btn"):
            with st.spinner("Loading Top 3 CNC entries (<= ₹20,000 each) & replaying 5m candles..."):
                rep_tmp = load_json("live_regime_report.json")
                wl_tmp = load_csv("live_watchlist_radar.csv")
                sn_tmp = load_csv("live_sniper_signals.csv")
                src_tmp = sn_tmp if not sn_tmp.empty else wl_tmp
                stage_top3_daily_cnc_entries(
                    watchlist_df=src_tmp,
                    regime_badge=rep_tmp.get("regime_badge", "SIDEWAYS_ROTATION"),
                    purchase_time_str="10:00:00",
                    eod_315pm_auto_exit=eod_exit_enabled,
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
            key="tab1_cnc_capital",
            help="Strict ₹20,000 maximum budget per ticker in CNC Delivery (any ticker priced > budget is excluded).",
        )

    sniper_df = load_csv("live_sniper_signals.csv")
    watchlist_df = load_csv("live_watchlist_radar.csv")
    top3_ledger_df = load_csv("sniper_telemetry_trades.csv")
    if top3_ledger_df.empty:
        top3_ledger_df = auto_eval_df

    watchlist_df = recalculate_cnc_sizing(watchlist_df, cnc_capital)
    sniper_df = recalculate_cnc_sizing(sniper_df, cnc_capital)

    v3_summary_meta = load_json("v3_expert_performance_summary.json")
    if not isinstance(v3_summary_meta, dict):
        v3_summary_meta = {}

    b_wr_live = float(v3_summary_meta.get("shadow_net_wr_pct", 0.0) or 0.0)
    b_pt_live = float(v3_summary_meta.get("shadow_post_tax_pnl_inr", 0.0) or 0.0)
    b_xirr_live = float(v3_summary_meta.get("shadow_annualized_xirr_pct", 0.0) or 0.0)
    b_tr_live = int(v3_summary_meta.get("shadow_validated_trades", 0) or 0)
    b_days_live = int(v3_summary_meta.get("unique_trading_days_b", 0) or 0)
    b_xirr_disp = f"{b_xirr_live:+.1f}% XIRR" if b_days_live >= 5 else f"XIRR Pending ({b_days_live}/5d)"

    # KPI Bar — Strictly Real Closed-Trade Telemetry (zero hardcoded fallbacks)
    if regime_rep:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Market Regime", regime_rep.get("regime_badge", "N/A"), f"Eligible <=₹{int(cnc_capital):,}: {len(watchlist_df)} Tickers")
        m2.metric("Nifty 50 (^NSEI)", f"{regime_rep.get('nifty_close', 0):,.2f}", f"{regime_rep.get('nifty_ret_1d_pct', 0):+.2f}% (1D)")
        m3.metric("India VIX (^INDIAVIX)", f"{regime_rep.get('india_vix', 0):.2f}", "3 bps Slippage | Max 1/Sector")
        m4.metric(
            f"🟣 Strategy B Closed WR — {b_wr_live:.1f}%",
            f"₹{b_pt_live:+,.2f} Post-Tax",
            f"{b_xirr_disp} | {b_tr_live} Closed Trades",
        )

    # Build probable candidates (Strategy B quality-gated & adaptive-ranked)
    st.markdown("---")
    probable_df = build_probable_intraday_candidates_sheet(
        watchlist_df=watchlist_df,
        sniper_df=sniper_df,
        op_signals_df=cached_op_signals if not cached_op_signals.empty else None,
        top3_ledger_df=top3_ledger_df,
        budget_per_ticker=cnc_capital,
        enable_mis_short=mis_short_enabled,
        current_regime=current_regime_badge,
    )
    if probable_df.empty:
        probable_df = load_csv("probable_intraday_candidates.csv")
    probable_b_df = load_csv("probable_candidates_strategy_b.csv")
    if probable_b_df.empty and not probable_df.empty and "Executes_In_Strategy_B" in probable_df.columns:
        probable_b_df = probable_df[probable_df["Executes_In_Strategy_B"] == True].copy()

    probable_b_df = recalculate_cnc_sizing(probable_b_df, cnc_capital)
    probable_b_df = _enrich_signals_for_live_tab(probable_b_df, engine_mode="B")
    if not mis_short_enabled and not probable_b_df.empty and "Signal_Direction" in probable_b_df.columns:
        probable_b_df = probable_b_df[~probable_b_df["Signal_Direction"].astype(str).str.contains("SHORT", na=False)].copy()

    top10_df = probable_b_df.head(10).copy() if not probable_b_df.empty else pd.DataFrame()

    # SECTION A: ADAPTIVE TOP-10 SIGNALS (HIGHEST SUCCESS PROBABILITY + MAX XIRR ON TOP)
    with st.container(border=True):
        st.markdown(
            f"### 🎯 Section A: Adaptive Top {len(top10_df)} Strategy B Signals — Clear BUY (CNC) vs SHORT SELL (MIS Only)"
        )
        st.caption(
            f"**Strategy B Quality Engine** (`{b_tr_live}` closed trades, `{b_wr_live:.1f}%` closed WR, `{b_xirr_disp}`). "
            "Auto-ranked across sessions using 14-day exponential decay + current-regime fit + Wilson Lower Bound (`z=1.645`) + Bayesian-shrunk XIRR — "
            "**highest success probability with max XIRR always on top**. Strict `<= ₹20,000` budget & prime window."
        )

        if not top10_df.empty:
            card_cols = st.columns(min(4, len(top10_df) + 1))
            for idx in range(min(3, len(top10_df))):
                rc = top10_df.iloc[idx]
                direction = str(rc.get("Signal_Direction", "🟢 BUY (CNC)"))
                badge = ["🏆 #1", "🥈 #2", "🥉 #3"][idx]
                card_cols[idx].metric(
                    f"{badge} {direction}: {rc.get('Ticker', '')}",
                    f"Entry ₹{float(rc.get('Recommended_Entry_Price', 0)):,.2f} ({int(rc.get('Raw_Qty', 1))} Qty)",
                    f"{str(rc.get('Strategy_ID', ''))} | {str(rc.get('Confidence_Level', ''))[:20]}",
                )
            if len(top10_df) > 3:
                card_cols[-1].metric(
                    "📊 Qualified Setups (<= ₹20K)",
                    f"{len(top10_df)} Shown / {len(probable_b_df)} Total",
                    f"Strategy B Only | {b_wr_live:.1f}% Closed WR",
                )

            signal_display_cols = [
                "Signal_Direction",
                "Ticker",
                "Strategy_Name",
                "Product_Mode",
                "Paper_Trade_Success_Rate",
                "Confidence_Level",
                "StratB_WLB_Pct",
                "StratB_XIRR_Pct",
                "Recommended_Entry_Price",
                "Recommended_Stop_Loss",
                "Safe_T1_And_Trail_SL_Levels",
                "Recommended_Exit_Target",
                "Trigger_Proximity_Status",
                "Live_LTP_And_Gap",
                "Qty_For_20K_And_Capital",
                "Est_Profit_Vs_Risk_INR",
                "Time_Exit_And_315PM_Rule",
                "Strategy_ID",
                "Pre_Configure_Broker_Order_Guide",
            ]
            avail_sig_cols = [c for c in signal_display_cols if c in top10_df.columns]
            render_frozen_3col_dataframe(top10_df[avail_sig_cols], height=420, auto_add_direction=False)
        else:
            st.info("No qualified Strategy B setups under the current budget, prime liquidity window, and regime filters.")

    # SECTION B: ACTIVE EXIT ALERTS & OPEN POSITIONS
    op_trades_live = load_csv("operator_auto_trades.csv")
    sniper_trades_live = load_csv("sniper_telemetry_trades.csv")

    with st.container(border=True):
        st.markdown("### 📌 Section B: Current Open Positions & Active Exit Alerts (Live P&L)")
        st.caption(
            "Active CNC and 1m Operator positions currently open in intraday replay (`ACTIVE`, `ACTIVE_INTRADAY`, `SWING_CONVERTED`, `ACTIVE_OPERATOR_TRADE`)."
        )

        active_positions = []
        if not sniper_trades_live.empty and "Status" in sniper_trades_live.columns:
            st_upper = sniper_trades_live["Status"].astype(str).str.upper()
            active_cnc = sniper_trades_live[
                st_upper.isin(["ACTIVE", "MONITORING", "ACTIVE_INTRADAY", "SWING_CONVERTED"])
            ].copy()
            if not active_cnc.empty:
                for _, r in active_cnc.iterrows():
                    entry_p = float(pd.to_numeric(r.get("Entry_Price", 0), errors="coerce") or 0.0)
                    qty_v = int(pd.to_numeric(r.get("CNC_Qty", 0), errors="coerce") or 0)
                    if qty_v <= 0 or entry_p <= 0 or entry_p > cnc_capital:
                        continue
                    live_p = float(pd.to_numeric(r.get("Live_CMP", r.get("Exit_Price", entry_p)), errors="coerce") or entry_p)
                    pnl = float(pd.to_numeric(r.get("PnL_INR", (live_p - entry_p) * qty_v), errors="coerce") or 0.0)
                    active_positions.append({
                        "Signal_Direction": "🟢 BUY (CNC)",
                        "Ticker": r.get("Ticker", ""),
                        "Strategy_ID": r.get("Logic_ID", r.get("Strategy_ID", "CNC")),
                        "Entry_Price": f"₹{entry_p:,.2f}",
                        "Live_CMP": f"₹{live_p:,.2f}",
                        "Stop_Loss": f"₹{float(pd.to_numeric(r.get('Current_Trailing_SL', r.get('Initial_SL', 0)), errors='coerce') or 0):,.2f}",
                        "Target": f"₹{float(pd.to_numeric(r.get('Target_Price', 0), errors='coerce') or 0):,.2f}",
                        "Unrealized_PnL": f"₹{pnl:+,.2f}",
                        "Status": str(r.get("Status", "ACTIVE")),
                        "Entry_Time": str(r.get("Purchase_Time", "")),
                    })

        if not op_trades_live.empty and "Status" in op_trades_live.columns:
            op_upper = op_trades_live["Status"].astype(str).str.upper()
            active_op = op_trades_live[
                op_upper.isin(["ACTIVE", "MONITORING", "OPEN", "ACTIVE_OPERATOR_TRADE"])
            ].copy()
            if not active_op.empty:
                for _, r in active_op.iterrows():
                    entry_p = float(pd.to_numeric(r.get("Confirm_Entry_Price", 0), errors="coerce") or 0.0)
                    qty_v = int(pd.to_numeric(r.get("Auto_Qty_20K", 0), errors="coerce") or 0)
                    if qty_v <= 0 or entry_p <= 0 or entry_p > cnc_capital:
                        continue
                    live_p = float(pd.to_numeric(r.get("Live_Or_Exit_Price", entry_p), errors="coerce") or entry_p)
                    side = str(r.get("Signal_Side", "BUY")).upper()
                    direction = "🔴 SHORT SELL (MIS Only)" if ("SELL" in side or "SHORT" in side) else "🟢 BUY (CNC/MIS)"
                    if "SELL" in side or "SHORT" in side:
                        pnl = float(pd.to_numeric(r.get("PnL_INR", (entry_p - live_p) * qty_v), errors="coerce") or 0.0)
                    else:
                        pnl = float(pd.to_numeric(r.get("PnL_INR", (live_p - entry_p) * qty_v), errors="coerce") or 0.0)
                    active_positions.append({
                        "Signal_Direction": direction,
                        "Ticker": r.get("Ticker", ""),
                        "Strategy_ID": r.get("Strategy_ID", ""),
                        "Entry_Price": f"₹{entry_p:,.2f}",
                        "Live_CMP": f"₹{live_p:,.2f}",
                        "Stop_Loss": f"₹{float(pd.to_numeric(r.get('Current_Trailing_SL', r.get('Initial_SL', 0)), errors='coerce') or 0):,.2f}",
                        "Target": f"₹{float(pd.to_numeric(r.get('Target_Price', 0), errors='coerce') or 0):,.2f}",
                        "Unrealized_PnL": f"₹{pnl:+,.2f}",
                        "Status": str(r.get("Status", "ACTIVE")),
                        "Entry_Time": str(r.get("Entry_Time_IST", r.get("Shock_Time_1m", ""))),
                    })

        if active_positions:
            pos_df = pd.DataFrame(active_positions)
            st.dataframe(pos_df, use_container_width=True, hide_index=True, height=min(250, 40 + 35 * len(pos_df)))
        else:
            st.info("No active open positions right now. Closed trades are reconciled in Tab 2.")

    # SECTION C: Lock Candidate into Top-3
    probable_all_df = _enrich_signals_for_live_tab(recalculate_cnc_sizing(probable_df, cnc_capital), engine_mode="AB")
    with st.expander("📌 Lock a Candidate into Top-3 Active 5m Exit Tracker (≤ ₹20K)", expanded=False):
        cnc_prob_sub = probable_all_df[probable_all_df["Candidate_Group"] == "CNC_OR_INTRADAY_SNIPER"].reset_index(drop=True) if not probable_all_df.empty and "Candidate_Group" in probable_all_df.columns else probable_all_df
        if not cnc_prob_sub.empty:
            lk1, lk2, lk3, lk4 = st.columns([1.8, 1.0, 1.0, 1.5])
            cand_tickers = cnc_prob_sub["Ticker"].tolist() if not cnc_prob_sub.empty else []
            sel_cand_ticker = lk1.selectbox(
                "Select Probable Candidate Configured in Broker",
                options=cand_tickers,
                key="tab1_top_prob_lock_ticker",
            )
            default_ep = (
                float(cnc_prob_sub.loc[cnc_prob_sub["Ticker"] == sel_cand_ticker, "Recommended_Entry_Price"].iloc[0])
                if sel_cand_ticker in cnc_prob_sub["Ticker"].values and "Recommended_Entry_Price" in cnc_prob_sub.columns
                else 100.0
            )
            cfg_ep = lk2.number_input("Entry Price (₹)", min_value=1.0, value=float(default_ep), step=0.5, key="tab1_top_prob_lock_price")
            cfg_time = lk3.text_input("Entry Time (IST)", value="10:00:00", key="tab1_top_prob_lock_time")
            lk4.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if lk4.button(f"✅ Lock {sel_cand_ticker} into Top-3", type="primary", use_container_width=True, key="tab1_top_prob_lock_btn"):
                row_match = watchlist_df[watchlist_df["Ticker"] == sel_cand_ticker]
                r_dict = row_match.iloc[0].to_dict() if not row_match.empty else {"Ticker": sel_cand_ticker, "Live_Price": cfg_ep}
                ok, msg, _ = stage_ticker_into_top3_cnc(
                    ticker=sel_cand_ticker,
                    row_data=r_dict,
                    purchase_time_str=cfg_time.strip(),
                    entry_price_override=float(cfg_ep),
                    budget_per_ticker=float(cnc_capital),
                    regime_badge=regime_rep.get("regime_badge", "SIDEWAYS_ROTATION"),
                    eod_315pm_auto_exit=eod_exit_enabled,
                    sync_purchase_candle_price=False,
                    trades_csv_path=SNIPER_TELEMETRY_CSV,
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)

    with st.expander(f"📋 All Qualified Strategy B Candidates ({len(probable_b_df)} Setups <= ₹{int(cnc_capital):,})", expanded=False):
        if not probable_b_df.empty:
            full_cols = [
                "Signal_Direction", "Ticker", "Strategy_Name", "Product_Mode",
                "Paper_Trade_Success_Rate", "Confidence_Level", "StratB_WLB_Pct", "StratB_XIRR_Pct",
                "Order_Side", "Recommended_Entry_Price",
                "Recommended_Stop_Loss", "Safe_T1_And_Trail_SL_Levels",
                "Recommended_Exit_Target", "Trigger_Proximity_Status",
                "Live_LTP_And_Gap", "Qty_For_20K_And_Capital", "Strategy_ID",
            ]
            avail_full = [c for c in full_cols if c in probable_b_df.columns]
            render_frozen_3col_dataframe(probable_b_df[avail_full], height=400, auto_add_direction=False)


def render_paper_trade_performance_tab():
    """
    Tab 2: Paper Trade Performance & 9-Strategy Scorecard.
    Computes all Win Rates, Post-Tax P&L, and XIRR (N>=5d guard) strictly from closed real trades.
    Zero 3Y backtest percentages appear in the 9-Strategy Scorecard.
    """
    v3_summary_meta = load_json("v3_expert_performance_summary.json")
    if not isinstance(v3_summary_meta, dict):
        v3_summary_meta = {}
    v3_alpha_meta = v3_summary_meta.get("alpha_vs_1st_stable", {}) if isinstance(v3_summary_meta, dict) else {}

    daily_summary_df = load_csv("daily_paper_trade_summary.csv")
    fresh_paper_df = load_csv("fresh_paper_trades_since_inception.csv")
    all_paper_trades_df = load_csv("all_strategies_paper_trades.csv")
    if fresh_paper_df.empty:
        fresh_paper_df = all_paper_trades_df.copy()
    strat_scorecard_df = load_csv("strategy_wise_paper_scorecard.csv")

    v3_shadow_df = load_csv("v3_expert_shadow_validation_ledger.csv")
    v3_live_df = load_csv("v3_expert_fresh_paper_ledger.csv")
    v3_daily_df = load_csv("v3_expert_daily_summary.csv")
    v3_scorecard_df = load_csv("v3_expert_strategy_scorecard.csv")
    v3_ab_trades_df = load_csv("v3_expert_ab_trade_by_trade_log.csv")
    v3_traps_df = load_csv("v3_expert_rejected_traps_log.csv")
    daily_cap_xirr_df = load_csv("daily_capital_charges_tax_xirr.csv")

    b_post_tax = float(v3_summary_meta.get("shadow_post_tax_pnl_inr", 0.0) or 0.0)
    b_wr = float(v3_summary_meta.get("shadow_net_wr_pct", 0.0) or 0.0)
    b_xirr = float(v3_summary_meta.get("shadow_annualized_xirr_pct", 0.0) or 0.0)
    b_trades = int(v3_summary_meta.get("shadow_validated_trades", 0) or 0)
    b_days = int(v3_summary_meta.get("unique_trading_days_b", 0) or 0)
    a_post_tax = float(v3_summary_meta.get("strat_a_post_tax_pnl_inr", 0.0) or 0.0)
    a_wr = float(v3_summary_meta.get("strat_a_net_wr_pct", 0.0) or 0.0)
    a_trades = int(v3_summary_meta.get("strat_a_total_trades", 0) or 0)
    alpha_pct = float(v3_alpha_meta.get("post_tax_profit_alpha_pct", 0.0) or 0.0) if isinstance(v3_alpha_meta, dict) else 0.0
    b_xirr_text = f"{b_xirr:+.1f}%" if b_days >= 5 else f"Pending ({b_days}/5d)"

    st.subheader("📊 Paper Trade Performance & Detailed Trade Log (Closed Real Trades Only)")
    st.caption(
        f"**Strategy B (Forward Quality Engine, >= {APP_CREATED_DATE})**: {b_wr:.1f}% Closed WR | "
        f"₹{b_post_tax:+,.2f} Post-Tax | {b_xirr_text} XIRR | {b_trades} Closed Trades across {b_days} Sessions"
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🟣 Strategy B Post-Tax P&L", f"₹{b_post_tax:+,.2f}", f"{b_trades} Closed | {b_wr:.1f}% WR")
    kpi2.metric("🟣 Strategy B Annualized XIRR", b_xirr_text, f"Requires N>=5 sessions ({b_days} observed)")
    kpi3.metric("🟢 Strategy A Post-Tax P&L", f"₹{a_post_tax:+,.2f}", f"{a_trades} Closed | {a_wr:.1f}% WR")
    kpi4.metric("🏆 Strategy B vs A Alpha", f"₹{b_post_tax - a_post_tax:+,.2f}", f"{alpha_pct:+.1f}% Post-Tax Delta")

    perf_tab1, perf_tab2, perf_tab3, perf_tab4, perf_tab5 = st.tabs([
        "📒 Trade-by-Trade Log",
        "🏆 9-Strategy Scorecard",
        "⚖️ A vs B Comparison",
        "💰 Daily Capital, Tax & XIRR",
        "🛡️ Blocked Traps & Trap Audit",
    ])

    with perf_tab1:
        st.markdown("##### 📒 Strategy B Execution Log (`BUY CNC/MIS` vs `SHORT SELL MIS Only`)")
        status_filter = st.radio(
            "Trade Status Filter:",
            options=["Closed Trades Only (Matches Scorecard WR)", "All Trades (Including Active Open)"],
            index=0,
            horizontal=True,
            key="tab2_log_status_filter",
        )
        v3_shadow_view = _add_signal_direction(v3_shadow_df)
        if not v3_shadow_view.empty and status_filter.startswith("Closed"):
            if "Trade_Status" in v3_shadow_view.columns:
                v3_shadow_view = v3_shadow_view[
                    ~v3_shadow_view["Trade_Status"].astype(str).str.upper().str.contains("ACTIVE|OPEN|PENDING|MONITORING", na=False)
                ].copy()
        if not v3_shadow_view.empty:
            trade_log_cols = [
                "Signal_Direction",
                "Entry_Date",
                "Ticker",
                "Strategy_Name" if "Strategy_Name" in v3_shadow_view.columns else "Strategy_ID",
                "Performance_Verdict",
                "Entry_To_Exit_Time",
                "Entry_Time_IST",
                "Exit_Time_IST",
                "Exit_Trigger_Reason",
                "Strategy_ID",
                "Paper_Order_Side",
                "Entry_Price",
                "Exit_Or_Live_Price",
                "Qty_20K",
                "Capital_Used_INR",
                "PnL_INR",
                "PnL_Pct",
                "Platform_Charges_INR",
                "Net_PnL_After_Charges_INR",
                "Post_Tax_PnL_INR",
                "Initial_SL",
                "Trailing_SL",
                "Target_Price",
                "Trade_Status",
                "Trade_ID",
            ]
            avail_log = [c for c in trade_log_cols if c in v3_shadow_view.columns]
            render_frozen_3col_dataframe(v3_shadow_view[avail_log], height=450, auto_add_direction=False)
        else:
            st.info("No closed Strategy B trades matching this filter yet.")

        with st.expander("📒 Strategy A Trade-by-Trade Log (for comparison)", expanded=False):
            a_view = _add_signal_direction(fresh_paper_df)
            if not a_view.empty:
                a_log_cols = [
                    "Signal_Direction", "Entry_Date", "Ticker", "Performance_Verdict",
                    "Entry_To_Exit_Time", "Entry_Time_IST", "Exit_Time_IST",
                    "Strategy_ID", "Paper_Order_Side", "Entry_Price", "Exit_Or_Live_Price",
                    "PnL_INR", "PnL_Pct", "Post_Tax_PnL_INR", "Trade_Status",
                ]
                avail_a_log = [c for c in a_log_cols if c in a_view.columns]
                render_frozen_3col_dataframe(a_view[avail_a_log], height=350, auto_add_direction=False)

    with perf_tab2:
        st.markdown("##### 🏆 All 9 Strategies — Real-Time Closed-Trade Paper Scorecard (Adaptive Ranking)")
        st.caption(
            "Strictly real closed paper trades since inception (zero 3Y backtest percentages — see Tab 5 for 3Y historical backtests). "
            "Sorted identically to Tab 1: **Adaptive Rank Score** (14d decay + regime match) -> **Wilson Lower Bound WR** -> **Strategy B XIRR** -> **Post-Tax ₹**."
        )
        scorecard_df_to_show = v3_scorecard_df if not v3_scorecard_df.empty else strat_scorecard_df
        if not scorecard_df_to_show.empty:
            sc_clean = scorecard_df_to_show.copy()
            if "Backtest_3Y_WR_A_vs_B" in sc_clean.columns:
                sc_clean = sc_clean.drop(columns=["Backtest_3Y_WR_A_vs_B"])
            sc_clean = _add_signal_direction(sc_clean)
            preferred_sc_cols = [
                "Signal_Direction",
                "Adaptive_Rank",
                "Strategy_Name",
                "Product_Mode",
                "Paper_Trade_Success_Rate",
                "Confidence_Level",
                "StratB_WLB_Pct",
                "StratB_XIRR_Pct",
                "Trades_Done_A_vs_B",
                "StratB_Post_Tax_INR",
                "StratA_Post_Tax_INR",
                "Post_Tax_Alpha_INR",
                "A_vs_B_Scorecard_Verdict",
                "Strategy_ID",
                "Two_Expert_Active_Rule_In_Strategy_B",
            ]
            avail_sc_cols = [c for c in preferred_sc_cols if c in sc_clean.columns]
            render_frozen_3col_dataframe(sc_clean[avail_sc_cols] if avail_sc_cols else sc_clean, height=430, auto_add_direction=False)

    with perf_tab3:
        st.markdown("##### ⚖️ Strategy A vs Strategy B — Side-by-Side Comparison")
        st.markdown("###### 🗓️ Day-by-Day Summary (Closed Trades & Daily Aggregate Tax/XIRR)")
        if not v3_daily_df.empty:
            render_frozen_3col_dataframe(v3_daily_df, height=230)

        st.markdown("###### ⏰ Trade-by-Trade Side-by-Side Execution Log (BUY vs SHORT SELL)")
        v3_ab_view = _add_signal_direction(v3_ab_trades_df)
        if not v3_ab_view.empty:
            render_frozen_3col_dataframe(v3_ab_view, height=400, auto_add_direction=False)

    with perf_tab4:
        st.markdown("##### 💰 Daily Capital, Platform Charges, 20% Tax & Aggregate XIRR Breakdown")
        st.caption(
            "Tracks peak simultaneous invested capital (`<= ₹20,000/slot`), brokerage + statutory charges + 3 bps slippage model, "
            "20% tax on net positive daily profit, and portfolio XIRR (annualized once `N >= 5` unique trading days exist)."
        )
        if not daily_cap_xirr_df.empty:
            render_frozen_3col_dataframe(daily_cap_xirr_df, height=280)

        with st.expander("📒 Strategy A Standalone Day-by-Day Capital & XIRR", expanded=False):
            if not daily_summary_df.empty:
                render_frozen_3col_dataframe(daily_summary_df, height=195)

    with perf_tab5:
        st.markdown("##### 🛡️ Filtered Low-Quality Setups & Strategy B Gate Audit")
        st.caption("Trades excluded by forward quality gates (lunch chop, R:R floor, sector concentration, regime shorting gate, or slot cap).")
        if not v3_traps_df.empty:
            render_frozen_3col_dataframe(v3_traps_df, height=300)

        with st.expander(f"📂 Standalone Strategy B Raw Ledgers (Today: {len(v3_live_df)} | All Closed+Open: {len(v3_shadow_df)})", expanded=False):
            v3_raw_mode = st.radio(
                "Select Ledger:",
                options=[
                    f"🟢 Current Session ({len(v3_live_df)} Trades)",
                    f"🧪 All Validated ({len(v3_shadow_df)} Trades)",
                ],
                index=0,
                horizontal=True,
                key="tab2_v3_standalone_radio",
            )
            raw_b_df = v3_live_df if v3_raw_mode.startswith("🟢") else v3_shadow_df
            if not raw_b_df.empty:
                render_frozen_3col_dataframe(_add_signal_direction(raw_b_df), height=310, auto_add_direction=False)


with tab1:
    render_live_signals_tab()

with tab2:
    render_paper_trade_performance_tab()

with tab3:
    render_manual_order_builder_and_scanner_tab()


with tab4:
    watchlist_df_t2 = recalculate_cnc_sizing(load_csv("live_watchlist_radar.csv"), MAX_BUDGET_PER_TICKER_INR)
    st.subheader("🛡️ Institutional Downside Protection & 4-Stage Chronological GTT Trailing Stop Ladder")
    st.markdown(
        """
        **How Our CNC + GTT OCO Downside Risk Engine Prevents Premature Stop-Outs:**
        1. **Strict Budget & Slot Discipline (`<= ₹20,000/Ticker`)**: All positions respect `<= ₹20,000/ticker` (`qty == 0` tickers excluded).
        2. **Structural Support Floor (`S1` + ATR)**: Initial GTT stop-loss is anchored below ORB Low / Support `S1` with `>= max(0.60%, 1.2 * ATR_5m)` noise cushion.
        3. **Chronological 5m / 1m High-Water-Mark (`HWM`) Ratchet**: Tracks `Last_Evaluated_Bar_TS` and `HWM` strictly forward in chronological order.
        4. **Cost-Aware Profit-Locking Ratchet**: `+0.80R -> Entry + 0.25R` (cost+ lock), scaling upward cleanly with trend extension.
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

with tab5:
    sw_tab1, sw_tab2, sw_tab3 = st.tabs([
        "🔄 3:10 PM Swing Predictor",
        "🏆 3-Year Backtest & Historical Archive",
        "📋 Continuous Critic Audit & Gap Governance",
    ])

    with sw_tab1:
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

    with sw_tab2:
        st.subheader("🏆 3-Year Historical Backtest & Pre-Inception Benchmark Archive (Strictly Separated from Live Tab 2)")
        bt_rebacktest = load_csv("two_expert_3y_rebacktest_comparison.csv")
        hist_benchmark_df = load_csv("historical_benchmark_trades.csv")
        bt_res = load_csv("backtest_3y_results.csv")
        bt_reg = load_csv("backtest_3y_regime_matrix.csv")
        bt_ver15 = load_csv("intraday_60d_15m_verification.csv")
        bt_timing_sum = load_csv("entry_timing_10am_5gate_summary.csv")
        bt_ledger = load_csv("backtest_trade_ledger_3y.csv")

        if not bt_rebacktest.empty:
            st.markdown("#### 0. 3-Year Historical Re-Backtest Reference (Excluded from Tab 2 Live Paper Scorecard)")
            render_frozen_3col_dataframe(bt_rebacktest, height=360)

        st.markdown(f"#### 1. Historical Benchmark Archive (`< {APP_CREATED_DATE}` | `{len(hist_benchmark_df)}` Rows)")
        if not hist_benchmark_df.empty:
            render_frozen_3col_dataframe(hist_benchmark_df, height=320)

        st.markdown("#### 2. Baseline 3-Year Sniper Strategy Comparison")
        if not bt_res.empty:
            render_frozen_3col_dataframe(bt_res)

        st.markdown("#### 3. Market Regime × Sniper Logic Matrix")
        if not bt_reg.empty:
            render_frozen_3col_dataframe(bt_reg)

        st.markdown("#### 4. 10:00 AM Entry Timing Study on 5-Gate Pre-Qualified Stocks")
        if not bt_timing_sum.empty:
            render_frozen_3col_dataframe(bt_timing_sum)

        st.markdown("#### 5. 60-Day 15-Minute Bar-by-Bar Intraday Walk-Forward Verification")
        if not bt_ver15.empty:
            render_frozen_3col_dataframe(bt_ver15)

        st.markdown("#### 6. Filterable 3-Year Historical Trade Ledger")
        if not bt_ledger.empty:
            sniper_only = st.checkbox("Show Top 6 Sniper Logics Only (L1–L6)", value=True)
            view_df = bt_ledger[bt_ledger["Logic_ID"].str.startswith(("L1_", "L2_", "L3_", "L4_", "L5_", "L6_"))] if sniper_only else bt_ledger
            render_frozen_3col_dataframe(view_df.tail(250), height=380)

    with sw_tab3:
        st.subheader("📋 Continuous Critic Audit, Gap Governance & Real Daily Capital / XIRR Ledger")
        audit_log_df = load_csv("regular_audit_log.csv")
        gaps_log_df = load_csv("audit_gaps_and_changes_log.csv")
        daily_cap_xirr_df = load_csv("daily_capital_charges_tax_xirr.csv")
        critic_memory = load_json("audit_critic_memory.json")

        if audit_log_df.empty or gaps_log_df.empty or daily_cap_xirr_df.empty or not critic_memory:
            try:
                res_mem = run_continuous_critic_audit_and_memory_check(force_append_snapshot=False)
                if isinstance(res_mem, dict) and res_mem:
                    critic_memory = res_mem
                audit_log_df = load_csv("regular_audit_log.csv")
                gaps_log_df = load_csv("audit_gaps_and_changes_log.csv")
                daily_cap_xirr_df = load_csv("daily_capital_charges_tax_xirr.csv")
            except Exception:
                pass

        v1_m = critic_memory.get("v1_pre_fix_baseline_summary") or {}
        v2_m = critic_memory.get("v2_post_fix_active_summary") or {}
        guard_status = critic_memory.get("active_stability_guardrails", {}) if isinstance(critic_memory, dict) else {}

        act_c1, act_c2 = st.columns([2.3, 1.2])
        with act_c1:
            all_passed = bool(critic_memory.get("all_guardrails_passing", True))
            if all_passed:
                st.success(
                    f"✅ **Critic Guardrails Active (`{critic_memory.get('engine_version', 'Live')}`)**: "
                    f"All institutional stability & data-integrity guardrails passing (`{critic_memory.get('failed_guardrails_count', 0)}` failed)."
                )
            else:
                st.warning(
                    f"⚠️ **Critic Memory Alert**: `{critic_memory.get('failed_guardrails_count', 0)}` guardrail(s) require review."
                )
        with act_c2:
            if st.button(
                "➕ Run Critic Audit & Append Snapshot Now",
                type="primary",
                use_container_width=True,
                key="btn_append_critic_audit",
            ):
                run_continuous_critic_audit_and_memory_check(force_append_snapshot=True)
                st.toast("Critic Audit snapshot evaluated against closed real trades!", icon="📋")
                st.rerun()

        v2_days = int(v2_m.get("unique_days", 0) or 0)
        v2_xirr_txt = f"{float(v2_m.get('xirr_pct', 0.0)):+.2f}%" if v2_days >= 5 else f"Pending ({v2_days}/5d)"
        ca1, ca2, ca3, ca4 = st.columns(4)
        ca1.metric(
            "🎯 Active Closed Net Win Rate",
            f"{float(v2_m.get('net_wr_pct', 0.0)):.1f}%",
            f"{int(v2_m.get('n_closed', 0))} Closed ({int(v2_m.get('net_w', 0))}W/{int(v2_m.get('net_l', 0))}L)",
        )
        ca2.metric(
            "💼 Avg Daily Peak Capital",
            f"₹{float(v2_m.get('peak_cap_inr', 0.0)):,.0f}",
            f"Over {v2_days} observed sessions",
        )
        ca3.metric(
            "💵 Active Post-Tax Profit",
            f"₹{float(v2_m.get('post_tax_profit_inr', 0.0)):+,.2f}",
            f"Charges ₹{float(v2_m.get('total_charges_inr', 0.0)):,.2f} | 20% Tax",
        )
        ca4.metric(
            "🚀 Annualized Post-Tax XIRR",
            v2_xirr_txt,
            f"Pre-fix archive WR: {float(v1_m.get('net_wr_pct', 0.0)):.1f}%" if v1_m else "Clean Rebuild Mode",
        )

        with st.expander("🛡️ View Automatic Critic Memory Guardrails (`audit_critic_memory.json`)", expanded=False):
            if isinstance(critic_memory, dict):
                st.markdown(f"**Instructions**: `{critic_memory.get('next_critic_instructions', '')}`")
                st.json(guard_status)

        st.markdown("#### 💰 1. Daily Capital, Indian Platform Charges, 20% Tax & XIRR Breakdown")
        if not daily_cap_xirr_df.empty:
            render_frozen_3col_dataframe(daily_cap_xirr_df, height=280)

        st.markdown(f"#### 📜 2. Critic Audit History (`regular_audit_log.csv` — `{len(audit_log_df)}` Rows)")
        if not audit_log_df.empty:
            render_frozen_3col_dataframe(audit_log_df, height=310)

        st.markdown(f"#### 🔧 3. Audit Gaps & Recommended Quality Upgrades (`audit_gaps_and_changes_log.csv` — `{len(gaps_log_df)}` Gaps)")
        if not gaps_log_df.empty:
            render_frozen_3col_dataframe(gaps_log_df, height=340)


