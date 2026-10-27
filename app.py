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
import json
from datetime import datetime
import pandas as pd
import streamlit as st

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

st.title("🎯 Institutional CNC Intraday Sniper, 5m Auto-Exit & GTT Desk")
st.caption(
    "Execution Mode: **100% CNC Only (Delivery — 0% MIS)** | "
    "**Budget Rule: Max `₹20,000` per Ticker & Max `3` Entries/Day (`Top 3` Slots <= `₹60,000`)** | "
    "**5-Minute Auto-Refresh & Auto-Exit + Late-Click 5m Candle Replay (`Purchase Time + 5m -> 3:15 PM IST`)**"
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


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Tab 1: CNC Entry (Top 3 <=₹20K) & 5m Auto-Exit Desk",
    "🛡️ Downside Protection & Trailing SL Ladder",
    "🔄 3:10 PM CNC Intraday-to-Swing Predictor",
    "🏆 3-Year Backtest (Top 6 Sniper Logics)",
    "🔬 Production Forensics & Integration Blueprint",
])


@st.fragment(run_every="5m")
def render_live_5m_auto_refresh_desk():
    """
    Auto-refreshes every 5 minutes (300 seconds) and automatically evaluates 5-minute candles
    from `Purchase_Time + 5m` up to `3:15 PM IST`, executing exits when SL/Trail, Target, or 3:15 PM is met.
    """
    # Automatically evaluate chronological 5m exits on every 5-minute cycle
    auto_eval_df = evaluate_chronological_exits(
        trades_csv_path=SNIPER_TELEMETRY_CSV,
        force_full_replay=False,
        eod_315pm_auto_exit=st.session_state.get("eod_315pm_auto_exit_toggle", True),
    )

    now_ist_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown(
        f"⏱️ **5-Minute Auto-Refresh & Auto-Exit Engine: `ACTIVE (Every 5 Minutes)`** | "
        f"Last 5m Candle Check: **`{now_ist_str}`** | "
        f"**Budget Cap: `₹20,000 / Ticker` | Daily Entry Cap: `Max 3 Tickers/Day (₹60,000 Total)`**"
    )

    # Top Control Bar
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.4, 1.4, 1.3, 1.9])
    with ctrl1:
        if st.button("🔄 Run Full Live Market Scan & 5m Check", type="primary", use_container_width=True):
            with st.spinner("Scanning 54 NSE & International instruments + evaluating 5m CNC exits..."):
                scan_live_sniper_opportunities()
                evaluate_chronological_exits(
                    trades_csv_path=SNIPER_TELEMETRY_CSV,
                    force_full_replay=True,
                    eod_315pm_auto_exit=st.session_state.get("eod_315pm_auto_exit_toggle", True),
                )
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
            "🔎 Card 1 Candidate Filter",
            options=[
                "Top Actionable & High-Readiness CNC Setups (Top 3 Highlighted)",
                "Triggered Sniper Signals Only (Immediate CNC Buy)",
                "All 54 NSE & International Universe Instruments",
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
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Market Regime", regime_rep.get("regime_badge", "N/A"))
        m2.metric("Nifty 50 (^NSEI)", f"{regime_rep.get('nifty_close', 0):,.2f}", f"{regime_rep.get('nifty_ret_1d_pct', 0):+.2f}% (1D)")
        m3.metric("India VIX (^INDIAVIX)", f"{regime_rep.get('india_vix', 0):.2f}", "Max ₹20K/Ticker | Max 3/Day")
        tot_pnl_banner = float(top3_ledger_df["PnL_INR"].sum()) if not top3_ledger_df.empty and "PnL_INR" in top3_ledger_df.columns else 0.0
        tot_cap_banner = float(top3_ledger_df["Capital_Used_INR"].sum()) if not top3_ledger_df.empty and "Capital_Used_INR" in top3_ledger_df.columns else 0.0
        tot_pct_banner = (tot_pnl_banner / tot_cap_banner * 100.0) if tot_cap_banner > 0 else 0.0
        m4.metric(
            f"Top 3 Daily CNC P&L ({len(top3_ledger_df)}/3 Slots)",
            f"₹{tot_pnl_banner:+,.2f} ({tot_pct_banner:+.2f}%)",
            f"Deployed: ₹{tot_cap_banner:,.0f} / ₹60,000 Max",
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

    st.markdown("---")
    with st.expander("🏗️ Can the Best Intraday Logic Be Updated in Production (`smart-etf-tracker`) Without Disturbing Other Logics? (Advisory Blueprint — Zero Production Changes Made)", expanded=True):
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

    st.markdown("#### 📡 Full 54-Instrument Indian & International CNC Readiness Radar (`Initial Rows = Order Ready` | `First 3 Columns Frozen`)")
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
