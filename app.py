"""
Institutional Intraday Trading Expert Dashboard (`app.py`)
==========================================================
Interactive Streamlit platform for Indian (NSE) & International Markets:
- Strictly CNC-Only Execution (100% Cash & Carry / Delivery, Zero MIS Auto-Squareoff Penalty)
- Tab 1 Side-by-Side Cards:
  * Left Card: CNC Intraday Entry Orders (Best Empirically Proven Sniper Logics L1–L6)
  * Right Card: CNC Exit Readiness & Broker GTT (OCO) Exit Orders (4-Stage Trail + 3:10 PM 5-Gate Decision)
- All tables display Order-Related Data in the initial rows and freeze the first 3 columns.
"""

import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st

from intraday_expert_engine import (
    DATA_DIR,
    SNIPER_TRADES_CSV,
    TOP_5_LOGICS_METADATA,
    scan_live_sniper_opportunities,
    evaluate_chronological_exits,
    predict_swing_conversion_success,
)

st.set_page_config(
    page_title="Intraday-Only CNC | Institutional Sniper & GTT Exit Desk",
    page_icon="🎯",
    layout="wide",
)

# Custom CSS to reinforce card styling and sticky first-3-columns in any rendered HTML tables
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

st.title("🎯 Institutional CNC Intraday Sniper & GTT Exit Desk")
st.caption(
    "Execution Mode: **100% CNC Only (Cash & Carry / Delivery — Zero MIS)** | "
    "Top 6 Multi-Regime Sniper Logics (`L1`–`L6`) | 10:00 AM Entry Window | "
    "4-Stage Chronological GTT OCO Trailing Stop & 3:10 PM 5-Gate Swing Hold Predictor"
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

    # Ensure col0 values are unique strings so MultiIndex never collapses adjacent rows
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
    Dynamically updates CNC order quantity, order value, and rupee stop risk based on user's CNC capital.
    """
    if df.empty or "CNC_Buy_Limit_Price" not in df.columns:
        return df
    out = df.copy()
    prices = pd.to_numeric(out["CNC_Buy_Limit_Price"], errors="coerce").fillna(1.0).clip(lower=1.0)
    sl_prices = pd.to_numeric(out["Initial_Protected_SL"], errors="coerce").fillna(prices * 0.98)
    qtys = (cnc_capital_inr // prices).clip(lower=1).astype(int)
    out["Suggested_CNC_Qty"] = qtys
    out["CNC_Order_Value_INR"] = (qtys * prices).round(2)
    out["Max_Stop_Risk_INR"] = (qtys * (prices - sl_prices).clip(lower=0.05)).round(2)
    return out


def build_entry_order_initial_rows_table(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the CNC Entry Order Specification Table where:
    - INITIAL ROWS (Rows 01–08) are strictly the Broker CNC Order Execution Data.
    - LOWER ROWS (Rows 09–14) are the Sniper Logic, 5-Gate, and Structural Support metrics.
    - FIRST 3 COLUMNS (`Row_Group`, `CNC_Order_Field`, and `#1 Primary CNC Pick`) are frozen!
    """
    if candidates_df.empty:
        return pd.DataFrame()

    top_n = candidates_df.head(6).copy().reset_index(drop=True)
    col_Labels = [f"#{i+1} {row['Ticker']}" for i, row in top_n.iterrows()]

    rows_spec = [
        # INITIAL ROWS = ORDER-RELATED DATA
        ("01. ORDER DATA (INITIAL)", "CNC Order Action", [r["CNC_Order_Action"] for _, r in top_n.iterrows()]),
        ("02. ORDER DATA (INITIAL)", "Broker Product Type", [r["Product_Type"] for _, r in top_n.iterrows()]),
        ("03. ORDER DATA (INITIAL)", "Order Type & Timing", [r["CNC_Order_Type"] for _, r in top_n.iterrows()]),
        ("04. ORDER DATA (INITIAL)", "CNC Buy Limit Price (₹)", [f"₹{float(r['CNC_Buy_Limit_Price']):,.2f}" for _, r in top_n.iterrows()]),
        ("05. ORDER DATA (INITIAL)", "Suggested CNC Qty (Shares)", [f"{int(r['Suggested_CNC_Qty'])} Qty" for _, r in top_n.iterrows()]),
        ("06. ORDER DATA (INITIAL)", "Total CNC Cash Required (₹)", [f"₹{float(r['CNC_Order_Value_INR']):,.0f}" for _, r in top_n.iterrows()]),
        ("07. ORDER DATA (INITIAL)", "Initial Protected SL (₹ / %)", [f"₹{float(r['Initial_Protected_SL']):,.2f} (-{float(r['Downside_Risk_Pct']):.2f}%)" for _, r in top_n.iterrows()]),
        ("08. ORDER DATA (INITIAL)", "Intraday Target & R:R", [f"₹{float(r['Intraday_Target']):,.2f} ({r['Risk_Reward_Ratio']})" for _, r in top_n.iterrows()]),
        # TECHNICAL & LOGIC CONTEXT ROWS BELOW INITIAL ORDER ROWS
        ("09. SNIPER LOGIC", "Best Sniper Logic ID", [r["Best_Logic_ID"] for _, r in top_n.iterrows()]),
        ("10. SNIPER LOGIC", "Sniper Readiness / Conviction", [f"{float(r['Sniper_Readiness_Score']):.1f} / 100" for _, r in top_n.iterrows()]),
        ("11. 5-GATE 10AM CHECK", "5-Gate Pre-Qualified (10 AM)", ["✅ YES (5/5)" if bool(r["Meets_5Gate_10AM_Entry"]) else f"Gates: {r['Swing_Conversion_Gates']}" for _, r in top_n.iterrows()]),
        ("12. SUPPORT & CMP", "Live CMP vs Structural S1", [f"CMP ₹{float(r['Live_Price']):,.2f} | S1 ₹{float(r['Structural_S1_Floor']):,.2f}" for _, r in top_n.iterrows()]),
        ("13. RELATIVE STRENGTH", "Stock 5D RS / Sector 5D RS", [f"{float(r['5D_Rel_Strength_Pct']):+.2f}% / {float(r['Sector_5D_RS_Pct']):+.2f}%" for _, r in top_n.iterrows()]),
        ("14. MOMENTUM & VOL", "RSI(14) & 20D Vol Ratio", [f"RSI {float(r['RSI_14']):.1f} | Vol {float(r['Vol_Ratio_20D']):.2f}x" for _, r in top_n.iterrows()]),
    ]

    records = []
    for r_group, r_field, vals in rows_spec:
        rec = {
            "Row_Priority": r_group,
            "CNC_Order_Parameter": r_field,
        }
        for c_name, v in zip(col_Labels, vals):
            rec[c_name] = v
        records.append(rec)
    return pd.DataFrame(records)


def build_exit_gtt_initial_rows_table(exit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the CNC Exit & Broker GTT (OCO) Order Specification Table where:
    - INITIAL ROWS (Rows 01–08) are strictly the Exit Readiness & Broker GTT OCO Order Parameters.
    - LOWER ROWS (Rows 09–14) are the 4-Stage Trailing SL Ratchet Ladder & 5-Gate Swing Conversion metrics.
    - FIRST 3 COLUMNS (`Row_Priority`, `GTT_Exit_Parameter`, and `#1 Monitored CNC Position`) are frozen!
    """
    if exit_df.empty:
        return pd.DataFrame()

    top_n = exit_df.head(6).copy().reset_index(drop=True)
    col_labels = [f"#{i+1} {row['Ticker']}" for i, row in top_n.iterrows()]

    rows_spec = [
        # INITIAL ROWS = EXIT & GTT ORDER-RELATED DATA
        ("01. GTT ORDER (INITIAL)", "When Ready for Exit / Action", [r["Exit_Readiness_Status"] for _, r in top_n.iterrows()]),
        ("02. GTT ORDER (INITIAL)", "Broker GTT Order Mode", ["GTT OCO (2-Leg Sell CNC)" for _ in range(len(top_n))]),
        ("03. GTT ORDER (INITIAL)", "Product & Sell Qty (CNC)", [f"CNC Delivery | {int(r['Suggested_CNC_Qty'])} Qty" for _, r in top_n.iterrows()]),
        ("04. GTT ORDER (INITIAL)", "GTT Leg-1: SL Trigger Price (₹)", [f"₹{float(r['GTT_SL_Trigger']):,.2f}" for _, r in top_n.iterrows()]),
        ("05. GTT ORDER (INITIAL)", "GTT Leg-1: SL Limit Price (₹)", [f"₹{float(r['GTT_SL_Limit']):,.2f}" for _, r in top_n.iterrows()]),
        ("06. GTT ORDER (INITIAL)", "GTT Leg-2: Target Trigger (₹)", [f"₹{float(r['GTT_Target_Trigger']):,.2f}" for _, r in top_n.iterrows()]),
        ("07. GTT ORDER (INITIAL)", "GTT Leg-2: Target Limit (₹)", [f"₹{float(r['GTT_Target_Limit']):,.2f}" for _, r in top_n.iterrows()]),
        ("08. GTT ORDER (INITIAL)", "3:10 PM CNC Exit vs Swing Hold", [r["CNC_310PM_Decision"] for _, r in top_n.iterrows()]),
        # TRAILING STOP LADDER & SWING CONVERSION CONTEXT BELOW INITIAL ROWS
        ("09. TRAIL RATCHET", "Active Trailing SL Stage", [r["Active_Trail_Stage"] for _, r in top_n.iterrows()]),
        ("10. TRAIL RATCHET", "Stage 1 Breakeven+ SL (+0.18 ATR)", [f"₹{float(r['Stage1_Breakeven_SL']):,.2f}" for _, r in top_n.iterrows()]),
        ("11. TRAIL RATCHET", "Stage 2 Profit-Lock SL (+0.65 ATR)", [f"₹{float(r['Stage2_ProfitLock_SL']):,.2f}" for _, r in top_n.iterrows()]),
        ("12. TRAIL RATCHET", "Stage 3 66%-Lock SL (+1.15 ATR)", [f"₹{float(r['Stage3_Chandelier_SL']):,.2f}" for _, r in top_n.iterrows()]),
        ("13. SWING GTT TARGET", "5-Day Swing GTT Target (+3.2 ATR)", [f"₹{float(r['GTT_Swing_Target']):,.2f}" for _, r in top_n.iterrows()]),
        ("14. 5-GATE WIN PROB", "3:10 PM Swing Win Prob & Gates", [f"{float(r['Swing_Conversion_Prob_Pct']):.1f}% ({r['Swing_Conversion_Gates']})" for _, r in top_n.iterrows()]),
    ]

    records = []
    for r_group, r_field, vals in rows_spec:
        rec = {
            "Row_Priority": r_group,
            "GTT_Exit_Parameter": r_field,
        }
        for c_name, v in zip(col_labels, vals):
            rec[c_name] = v
        records.append(rec)
    return pd.DataFrame(records)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Tab 1: CNC Entry & GTT Exit Desk (Side-by-Side)",
    "🛡️ Downside Protection & Trailing SL Ladder",
    "🔄 3:10 PM CNC Intraday-to-Swing Predictor",
    "🏆 3-Year Backtest (Top 6 Sniper Logics)",
    "🔬 Production Forensics (smart-etf-tracker)",
])

with tab1:
    # Top Control Bar
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.4, 1.3, 1.5, 1.8])
    with ctrl1:
        if st.button("🔄 Run Live Market CNC Sniper Scan", type="primary", use_container_width=True):
            with st.spinner("Scanning 54 NSE & International instruments + evaluating 5m CNC GTT exits..."):
                scan_live_sniper_opportunities()
                evaluate_chronological_exits(SNIPER_TRADES_CSV)
            st.rerun()
    with ctrl2:
        if st.button("⚡ Refresh 5m Chronological GTT Status", use_container_width=True):
            with st.spinner("Checking chronological 5-minute bars for GTT SL/Target & Trail upgrades..."):
                evaluate_chronological_exits(SNIPER_TRADES_CSV)
            st.rerun()
    with ctrl3:
        cnc_capital = st.number_input(
            "💰 Per-Trade CNC Cash Capital (₹, 0% MIS)",
            min_value=10000.0,
            max_value=5000000.0,
            value=100000.0,
            step=25000.0,
            help="All trades are executed in CNC (Cash & Carry / Delivery) with 1x cash and zero MIS auto-squareoff risk.",
        )
    with ctrl4:
        filter_mode = st.selectbox(
            "🔎 Card 1 & 2 Candidate Filter",
            options=[
                "Top Actionable & High-Readiness CNC Setups (Score >= 80)",
                "Triggered Sniper Signals Only (Immediate CNC Buy)",
                "All 54 NSE & International Universe Instruments",
            ],
            index=0,
        )

    regime_rep = load_json("live_regime_report.json")
    sniper_df = load_csv("live_sniper_signals.csv")
    watchlist_df = load_csv("live_watchlist_radar.csv")
    active_ledger_df = load_csv("intraday_sniper_trades.csv")

    watchlist_df = recalculate_cnc_sizing(watchlist_df, cnc_capital)
    sniper_df = recalculate_cnc_sizing(sniper_df, cnc_capital)

    if regime_rep:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Market Regime", regime_rep.get("regime_badge", "N/A"))
        m2.metric("Nifty 50 (^NSEI)", f"{regime_rep.get('nifty_close', 0):,.2f}", f"{regime_rep.get('nifty_ret_1d_pct', 0):+.2f}% (1D)")
        m3.metric("India VIX (^INDIAVIX)", f"{regime_rep.get('india_vix', 0):.2f}", "Execution: CNC Only (No MIS)")
        m4.metric(
            "Triggered CNC Sniper Trades Today",
            f"{len(sniper_df)}",
            "Sniper Standby (Capital Protected)" if sniper_df.empty else f"Top Pick: {sniper_df.iloc[0]['Ticker']} ({sniper_df.iloc[0]['Best_Logic_ID']})",
        )

    # Filter candidate rows based on user selection, keeping Order_Priority_Rank sorted so actionable rows are always in initial rows
    if not watchlist_df.empty:
        if filter_mode.startswith("Triggered"):
            entry_candidates_df = sniper_df if not sniper_df.empty else watchlist_df.head(5)
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
    #   LEFT CARD : Which Intraday Trades to Enter in CNC (Best Logics L1–L6)
    #   RIGHT CARD: When Ready for Exit & Broker GTT (OCO) Exit Order
    # =========================================================================
    col_entry_card, col_exit_card = st.columns(2, gap="medium")

    # -------------------------------------------------------------------------
    # LEFT CARD: CNC INTRADAY ENTRY ORDERS (BEST LOGICS L1–L6)
    # -------------------------------------------------------------------------
    with col_entry_card:
        with st.container(border=True):
            st.markdown("### 🟢 Card 1: Which Intraday Trades to Enter (`CNC Only`)")
            st.caption(
                "**Product: `CNC` (Delivery / Cash & Carry — Never `MIS`)** | "
                "**Best Logics**: `L6` (5-Gate 10:00 AM Sniper, 89.4% WR), `L1` (Sector ORB, 78.7% WR), "
                "`L4` (Defensive RS Fortress, 78.0% WR), `L5` (Global Macro, 77.8% WR), `L2` (Support Spring, 76.1% WR), `L3` (Catalyst VWAP, 75.6% WR)"
            )

            if not sniper_df.empty:
                top_s = sniper_df.iloc[0]
                st.success(
                    f"**🔥 ACTIONABLE CNC ENTRY TRIGGERED**: Buy **`{top_s['Ticker']}`** in **`CNC`** "
                    f"(Qty: **`{int(top_s['Suggested_CNC_Qty'])}`** @ Limit **`₹{float(top_s['CNC_Buy_Limit_Price']):,.2f}`**) | "
                    f"Logic: `{top_s['Best_Logic_Name']}` | Initial Protected SL: `₹{float(top_s['Initial_Protected_SL']):,.2f}` | "
                    f"Intraday Target: `₹{float(top_s['Intraday_Target']):,.2f}`"
                )
            else:
                st.info(
                    "🛡️ **0 Immediate Market Orders Forced Right Now**: Below are the highest-priority **CNC Limit / 10:00 AM Trigger Orders** "
                    "pre-calculated from our Top 6 Sniper Logics. Enter ONLY if price triggers `CNC_Buy_Limit_Price`."
                )

            st.markdown("##### 📋 1A. CNC Entry Order Ticket (`Initial Rows = Order Data` | `First 3 Columns Frozen`)")
            entry_ticket_df = build_entry_order_initial_rows_table(entry_candidates_df)
            render_frozen_3col_dataframe(entry_ticket_df, height=390)

            st.markdown("##### 🎯 1B. Ranked CNC Entry Table (`Initial Rows = Actionable CNC Orders` | `First 3 Columns Frozen`)")
            if not entry_candidates_df.empty:
                entry_cols = [
                    # First 3 Frozen Columns
                    "Ticker",
                    "CNC_Order_Action",
                    "CNC_Buy_Limit_Price",
                    # Order-Related Columns Immediately Next
                    "Product_Type",
                    "CNC_Order_Type",
                    "Suggested_CNC_Qty",
                    "CNC_Order_Value_INR",
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
                    "Best_Logic_Name",
                ]
                avail_entry_cols = [c for c in entry_cols if c in entry_candidates_df.columns]
                render_frozen_3col_dataframe(entry_candidates_df[avail_entry_cols], height=360)

                # Quick Action to Stage/Log a CNC Entry into Active GTT Exit Monitor
                with st.expander("➕ Log / Stage a CNC Entry Order into Active GTT Exit Monitor (Card 2)"):
                    sel_col1, sel_col2 = st.columns([2, 1])
                    pick_ticker = sel_col1.selectbox(
                        "Select Ticker Entered in Broker (CNC)",
                        options=entry_candidates_df["Ticker"].tolist(),
                        key="stage_cnc_ticker",
                    )
                    if sel_col2.button("✅ Add to Active CNC Monitor", use_container_width=True):
                        row_match = entry_candidates_df[entry_candidates_df["Ticker"] == pick_ticker].iloc[0]
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        sl_t = float(row_match["GTT_SL_Trigger"])
                        tgt_t = float(row_match["GTT_Target_Trigger"])
                        new_trade = {
                            "Ticker": row_match["Ticker"],
                            "GTT_Order_Status": f"🛡️ GTT OCO ARMED: SL ₹{sl_t:.2f} | TGT ₹{tgt_t:.2f}",
                            "GTT_SL_Trigger": sl_t,
                            "Product_Type": "CNC (Delivery - 0% MIS)",
                            "CNC_Qty": int(row_match["Suggested_CNC_Qty"]),
                            "Entry_Price": float(row_match["CNC_Buy_Limit_Price"]),
                            "Live_CMP": float(row_match["Live_Price"]),
                            "GTT_SL_Limit": float(row_match["GTT_SL_Limit"]),
                            "GTT_Target_Trigger": tgt_t,
                            "GTT_Target_Limit": float(row_match["GTT_Target_Limit"]),
                            "Active_Trail_Stage": row_match["Active_Trail_Stage"],
                            "CNC_310PM_Decision": row_match["CNC_310PM_Decision"],
                            "PnL_Pct": 0.0,
                            "Trade_ID": f"CNC_{int(datetime.now().timestamp())}_{row_match['Ticker']}",
                            "Entry_Timestamp": now_str,
                            "Market": row_match["Market"],
                            "Sector_Group": row_match["Sector_Group"],
                            "Logic_ID": row_match["Best_Logic_ID"],
                            "Logic_Name": row_match["Best_Logic_Name"],
                            "Market_Regime_At_Entry": regime_rep.get("regime_badge", "LIVE"),
                            "Conviction_Score": float(row_match["Sniper_Readiness_Score"]),
                            "Status": "ACTIVE_INTRADAY",
                            "High_Water_Mark": float(row_match["Live_Price"]),
                            "Structural_S1_Floor": float(row_match["Structural_S1_Floor"]),
                            "Initial_SL": float(row_match["Initial_Protected_SL"]),
                            "Current_Trailing_SL": sl_t,
                            "Target_Price": tgt_t,
                            "ATR_At_Entry": round((tgt_t - float(row_match["CNC_Buy_Limit_Price"])) / 2.0, 2),
                            "Swing_Conversion_Prob_Pct": float(row_match["Swing_Conversion_Prob_Pct"]),
                            "Swing_Conversion_Gates": row_match["Swing_Conversion_Gates"],
                            "Swing_Conversion_Verdict": row_match["Swing_Conversion_Verdict"],
                            "Last_Evaluated_Bar_TS": now_str,
                            "Exit_Timestamp": "",
                            "Exit_Price": 0.0,
                            "Exit_Reason": "",
                        }
                        curr_ledger = load_csv("intraday_sniper_trades.csv")
                        curr_ledger = pd.concat([curr_ledger, pd.DataFrame([new_trade])], ignore_index=True)
                        curr_ledger.to_csv(SNIPER_TRADES_CSV, index=False)
                        st.success(f"Added `{pick_ticker}` (CNC) to Active GTT Exit Monitor!")
                        st.rerun()

    # -------------------------------------------------------------------------
    # RIGHT CARD: WHEN READY FOR EXIT & BROKER GTT (OCO) EXIT ORDERS (CNC ONLY)
    # -------------------------------------------------------------------------
    with col_exit_card:
        with st.container(border=True):
            st.markdown("### 🛡️ Card 2: When Ready for Exit & Broker `GTT (OCO)` Orders")
            st.caption(
                "**Post-Buy CNC Rule**: Immediately after CNC Buy execution, place a **2-Leg Broker GTT OCO Sell Order** "
                "(`Leg 1: GTT_SL_Trigger / Limit` + `Leg 2: GTT_Target_Trigger / Limit`). "
                "Ratchet GTT SL across Stages 1–4; at **3:10 PM IST**, hold CNC overnight ONLY if 5-Gate Swing Criteria pass."
            )

            # Combine active CNC trades from ledger + top entry candidates so Card 2 always shows live GTT orders
            if not active_ledger_df.empty and "Status" in active_ledger_df.columns:
                active_only = active_ledger_df[active_ledger_df["Status"].isin(["ACTIVE_INTRADAY", "SWING_CONVERTED"])].copy()
            else:
                active_only = pd.DataFrame()

            if not active_only.empty:
                top_act = active_only.iloc[0]
                st.warning(
                    f"**🛡️ ACTIVE CNC POSITION MONITORED**: **`{top_act['Ticker']}`** | "
                    f"Status: **{top_act.get('GTT_Order_Status', 'GTT OCO ARMED')}** | "
                    f"GTT SL Trigger: **`₹{float(top_act.get('GTT_SL_Trigger', top_act['Current_Trailing_SL'])):,.2f}`** "
                    f"(Limit `₹{float(top_act.get('GTT_SL_Limit', float(top_act['Current_Trailing_SL'])*0.997)):,.2f}`) | "
                    f"GTT Target Trigger: **`₹{float(top_act.get('GTT_Target_Trigger', top_act['Target_Price'])):,.2f}`** | "
                    f"3:10 PM Rule: {top_act.get('CNC_310PM_Decision', '')}"
                )
            else:
                st.info(
                    "📋 **No Open CNC Position in Ledger Yet**: Displaying live **GTT OCO Exit Triggers, 4-Stage Trailing Stop Ratchets, "
                    "and 3:10 PM CNC Exit/Hold Signals** for today's top CNC candidates."
                )

            st.markdown("##### 📋 2A. Broker GTT (OCO) Exit Ticket (`Initial Rows = GTT Order Data` | `First 3 Columns Frozen`)")
            gtt_ticket_df = build_exit_gtt_initial_rows_table(entry_candidates_df)
            render_frozen_3col_dataframe(gtt_ticket_df, height=390)

            st.markdown("##### 🚨 2B. Live Exit Readiness & GTT Ratchet Table (`Initial Rows = Actionable Exits/GTT` | `First 3 Columns Frozen`)")
            if not entry_candidates_df.empty:
                exit_cols = [
                    # First 3 Frozen Columns (Ticker + When Ready for Exit + GTT SL Trigger)
                    "Ticker",
                    "Exit_Readiness_Status",
                    "GTT_SL_Trigger",
                    # Remaining GTT OCO & CNC Order Columns Immediately Next
                    "GTT_SL_Limit",
                    "GTT_Target_Trigger",
                    "GTT_Target_Limit",
                    "Active_Trail_Stage",
                    "CNC_310PM_Decision",
                    "Product_Type",
                    "Suggested_CNC_Qty",
                    "Stage1_Breakeven_SL",
                    "Stage2_ProfitLock_SL",
                    "Stage3_Chandelier_SL",
                    "GTT_Swing_Target",
                    "Swing_Conversion_Prob_Pct",
                    "Swing_Conversion_Gates",
                    "Live_Price",
                    "CNC_Buy_Limit_Price",
                    "Initial_Protected_SL",
                    "Structural_S1_Floor",
                ]
                avail_exit_cols = [c for c in exit_cols if c in entry_candidates_df.columns]
                render_frozen_3col_dataframe(entry_candidates_df[avail_exit_cols], height=360)

            if not active_ledger_df.empty:
                with st.expander("📒 View Tracked CNC Positions Ledger (`data/intraday_sniper_trades.csv` — First 3 Columns Frozen)", expanded=True):
                    ledger_cols_order = [
                        "Ticker", "GTT_Order_Status", "GTT_SL_Trigger",
                        "GTT_SL_Limit", "GTT_Target_Trigger", "GTT_Target_Limit",
                        "Active_Trail_Stage", "CNC_310PM_Decision", "Product_Type",
                        "CNC_Qty", "Entry_Price", "Live_CMP", "PnL_Pct",
                        "High_Water_Mark", "Structural_S1_Floor", "Logic_ID",
                        "Status", "Entry_Timestamp", "Last_Evaluated_Bar_TS",
                    ]
                    avail_l_cols = [c for c in ledger_cols_order if c in active_ledger_df.columns]
                    render_frozen_3col_dataframe(active_ledger_df[avail_l_cols], height=220)

    st.markdown("---")
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

with tab2:
    st.subheader("🛡️ Institutional Downside Protection & 4-Stage Chronological GTT Trailing Stop Ladder")
    st.markdown(
        """
        **How Our CNC + GTT OCO Downside Risk Engine Prevents Production Stop-Outs:**
        1. **100% CNC Execution (Zero MIS Penalty)**: All positions are taken in `CNC` with a 2-Leg Broker GTT OCO Sell order. You never face forced 3:20 PM broker auto-squareoff penalties, and winners passing the 5-Gate check at 3:10 PM convert effortlessly into 5-Day Swing trades.
        2. **Structural Support Floor (`S1`)**: Initial GTT stop-loss is anchored below **Opening Range Low (`ORB_Low`) / 20-Day Structural Support `S1` minus an `0.15x–0.18x ATR` cushion**.
        3. **Chronological High-Water-Mark (`HWM`) Ratchet**: Tracks `Last_Evaluated_Bar_TS` and `High_Water_Mark` bar-by-bar in strict chronological order—eliminating the `smart-etf-tracker` bug where afternoon SL upgrades were retroactively compared against 9:50 AM morning lows.
        4. **4-Stage GTT Profit-Locking Ratchet**:
           - **Stage 1 (`MFE >= +0.80 ATR`)**: Modify GTT SL Trigger to `Entry + 0.18 ATR` (True Breakeven + DP/STT cover).
           - **Stage 2 (`MFE >= +1.25 ATR`)**: Modify GTT SL Trigger to `Entry + 0.65 ATR` (Locks >50% of gain).
           - **Stage 3 (`MFE >= +1.75 ATR`)**: Modify GTT SL Trigger to `Entry + 1.15 ATR` (Locks >66% of gain).
           - **Stage 4 (`MFE >= +2.20 ATR`)**: Dynamic Chandelier GTT Trail at `High_Water_Mark - 0.50 ATR`.
        """
    )
    if not watchlist_df.empty:
        cols_ladder = [
            "Ticker", "Exit_Readiness_Status", "GTT_SL_Trigger",
            "GTT_SL_Limit", "GTT_Target_Trigger", "GTT_Target_Limit",
            "Product_Type", "Suggested_CNC_Qty", "CNC_Buy_Limit_Price",
            "Structural_S1_Floor", "Dist_Above_S1_Pct",
            "Initial_Protected_SL", "Downside_Risk_Pct",
            "Stage1_Breakeven_SL", "Stage2_ProfitLock_SL", "Stage3_Chandelier_SL",
            "GTT_Swing_Target", "CNC_310PM_Decision",
        ]
        avail_ladder = [c for c in cols_ladder if c in watchlist_df.columns]
        render_frozen_3col_dataframe(watchlist_df[avail_ladder], height=500)

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
    st.subheader("🔬 Forensic Audit of `smart-etf-tracker` Production Trades (`First 3 Columns Frozen`)")
    prod_audit = load_csv("production_trades_forensic_audit.csv")
    if not prod_audit.empty:
        render_frozen_3col_dataframe(prod_audit)
