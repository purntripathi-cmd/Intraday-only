"""
Intraday Sniper Execution Daemon (`intraday_daemon.py`)
=======================================================
Automated CLI & GitHub Actions runner for `Intraday-only`:
- Enforces strict ₹20,000 max budget per ticker (`MAX_BUDGET_PER_TICKER_INR = 20000.0`)
  and Max 3 entries per day (`MAX_DAILY_ENTRIES = 3` -> `Slot #1/3`, `Slot #2/3`, `Slot #3/3`,
  max ₹60,000 daily deployment, 100% CNC Delivery, 0% MIS).
- `--mode MORNING_SNIPER_SCAN` (09:45–10:00 IST):
  Scans Indian (NSE) & International markets using `intraday_expert_engine.py`.
- `--mode CHRONOLOGICAL_GTT_CHECK` (Every 5 mins intraday):
  Runs `evaluate_chronological_exits` from `Purchase_Time + 5m` through `3:15 PM IST`
  and auto-exits when Stop Loss / Trailing Stop, Profit Target, or 3:15 PM IST is hit.
- `--mode EOD_310PM_SWING_GATE` (15:10–15:15 IST):
  Evaluates active intraday trades at 3:15 PM IST and records realized Profit/Loss (`PnL_INR`, `PnL_Pct`).
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from intraday_expert_engine import (
    DATA_DIR,
    SNIPER_TRADES_CSV,
    SNIPER_TELEMETRY_CSV,
    OPERATOR_SIGNALS_CSV,
    OPERATOR_TRADES_CSV,
    OPERATOR_PERF_JSON,
    MAX_BUDGET_PER_TICKER_INR,
    MAX_DAILY_ENTRIES,
    UNIVERSE_META,
    scan_live_sniper_opportunities,
    evaluate_chronological_exits,
    stage_top3_daily_cnc_entries,
    scan_operator_1m_footprints,
    auto_execute_and_evaluate_operator_trades,
)


def init_trades_ledger():
    for path in (SNIPER_TRADES_CSV, SNIPER_TELEMETRY_CSV):
        if not os.path.exists(path):
            cols = [
                "Daily_Slot", "Ticker", "GTT_Order_Status", "GTT_SL_Trigger",
                "Product_Type", "CNC_Qty", "Capital_Used_INR", "Max_Budget_INR",
                "Purchase_Date", "Purchase_Time", "Entry_Price", "Live_CMP",
                "PnL_INR", "PnL_Pct", "Performance_Verdict",
                "GTT_SL_Limit", "GTT_Target_Trigger", "GTT_Target_Limit",
                "Active_Trail_Stage", "CNC_310PM_Decision",
                "Bars_Tracked_5m", "Window_Low_5m", "Window_High_5m", "Replay_Audit_Trail",
                "Entry_Timestamp", "Logic_ID", "Logic_Name", "Market_Regime",
                "Status", "High_Water_Mark", "Structural_S1_Floor",
                "Initial_SL", "Current_Trailing_SL", "Target_Price",
                "ATR_At_Entry", "Swing_Conversion_Prob_Pct", "Swing_Conversion_Verdict",
                "Last_Evaluated_Bar_TS", "Exit_Timestamp", "Exit_Price", "Exit_Reason"
            ]
            pd.DataFrame(columns=cols).to_csv(path, index=False)


def run_daemon(mode: str = "FULL_CYCLE"):
    init_trades_ledger()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== Intraday CNC Sniper Daemon | Mode: {mode} | Timestamp: {now_str} IST ===")
    print(
        f"=== Capital Rules: Max ₹{MAX_BUDGET_PER_TICKER_INR:,.0f}/ticker | "
        f"Max {MAX_DAILY_ENTRIES} entries/day (<= ₹{MAX_BUDGET_PER_TICKER_INR*MAX_DAILY_ENTRIES:,.0f} total) | CNC Only ==="
    )

    # Step 1: Live Market Regime & Sniper Opportunity Scan
    if mode in ("FULL_CYCLE", "MORNING_SNIPER_SCAN", "EOD_310PM_SWING_GATE"):
        report, snipers_df, watchlist_df = scan_live_sniper_opportunities()
        print(
            f"[Regime Radar] {report['regime_badge']} | Nifty: {report['nifty_close']} "
            f"({report['nifty_ret_1d_pct']:+.2f}%) | VIX: {report['india_vix']}"
        )
        print(f"[Authorized Logics] {', '.join(report['authorized_sniper_logics'])}")

        if mode in ("FULL_CYCLE", "MORNING_SNIPER_SCAN"):
            # Stage Top 3 CNC entries (<= ₹20,000 per ticker, max 3 entries/day) if ledger is empty or has legacy >3 rows
            curr_df = pd.read_csv(SNIPER_TELEMETRY_CSV) if os.path.exists(SNIPER_TELEMETRY_CSV) else pd.DataFrame()
            needs_top3_seed = (
                curr_df.empty
                or len(curr_df) != MAX_DAILY_ENTRIES
                or "Daily_Slot" not in curr_df.columns
                or ("Capital_Used_INR" not in curr_df.columns)
                or (curr_df["Capital_Used_INR"].max() > MAX_BUDGET_PER_TICKER_INR)
            )
            if needs_top3_seed:
                seeded = stage_top3_daily_cnc_entries(
                    watchlist_df=watchlist_df,
                    regime_badge=report["regime_badge"],
                    purchase_time_str="10:00:00",
                    eod_315pm_auto_exit=True,
                    sync_purchase_candle_price=True,
                    trades_csv_path=SNIPER_TELEMETRY_CSV,
                )
                print(f"[Top 3 Daily CNC Entries Seeded & 5m Replayed (10:05 -> 3:15 PM)] {len(seeded)} entries:")
                if not seeded.empty:
                    print(
                        seeded[
                            [
                                "Daily_Slot", "Ticker", "CNC_Qty", "Capital_Used_INR",
                                "Entry_Price", "Status", "Exit_Price", "PnL_INR", "PnL_Pct", "Performance_Verdict"
                            ]
                        ].to_string(index=False)
                    )

    # Step 2: Chronological 5m Candle Replay & Auto-Exit Check (Purchase_Time + 5m -> 3:15 PM IST)
    if mode in ("FULL_CYCLE", "CHRONOLOGICAL_GTT_CHECK", "EOD_310PM_SWING_GATE"):
        updated_ledger = evaluate_chronological_exits(
            trades_csv_path=SNIPER_TELEMETRY_CSV,
            force_full_replay=(mode == "FULL_CYCLE"),
            eod_315pm_auto_exit=True,
        )
        print(f"[5m Auto-Exit & Performance Check] Evaluated {len(updated_ledger)} Top-3 CNC entries.")
        if not updated_ledger.empty:
            tot_cap = updated_ledger["Capital_Used_INR"].sum()
            tot_pnl = updated_ledger["PnL_INR"].sum()
            print(f"[Portfolio Summary] Total Capital Deployed: ₹{tot_cap:,.2f} (<= ₹60,000 cap) | Net P&L: ₹{tot_pnl:+,.2f}")

    # Step 3: 1-Minute Operator / Big-Player Footprint Detector & Separate Auto-Trade (BUY & SELL) Tracker
    wl_for_op = watchlist_df if "watchlist_df" in locals() else None
    op_signals = scan_operator_1m_footprints(watchlist_df=wl_for_op)
    op_trades, op_perf = auto_execute_and_evaluate_operator_trades(op_signals)
    print(
        f"[Operator 1m Auto-Trade Engine (Separate Tracker)] Universe: {op_perf['universe_size_tickers']} tickers | "
        f"Auto-Trades: {op_perf['total_auto_trades']} ({op_perf['buy_auto_trades']} BUY / {op_perf['sell_auto_trades']} SELL) | "
        f"Win Rate: {op_perf['win_rate_pct']}% | Net P&L: ₹{op_perf['total_net_pnl_inr']:+,.2f} "
        f"(Buy P&L: ₹{op_perf['buy_net_pnl_inr']:+,.2f} | Sell P&L: ₹{op_perf['sell_net_pnl_inr']:+,.2f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Intraday Sniper Daemon")
    parser.add_argument(
        "--mode",
        type=str,
        default="FULL_CYCLE",
        choices=["FULL_CYCLE", "MORNING_SNIPER_SCAN", "CHRONOLOGICAL_GTT_CHECK", "EOD_310PM_SWING_GATE"],
    )
    args = parser.parse_args()
    run_daemon(args.mode)
