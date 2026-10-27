# 🎯 Intraday-Only: Institutional Sniper & Downside Risk Protection Expert

An institutional-grade **Intraday Trading & Downside Risk Protection Platform** for **Indian (NSE Equities & Sector ETFs)** and **International Markets (US Nasdaq 100, S&P 500, Global Gold & Macro Cues)**.

Engineered after a full forensic audit of production trades in [`smart-etf-tracker`](https://github.com/purntripathi-cmd/smart-etf-tracker) and validated across **3 full years of actual historical market data (`2023–2026`, ~740 trading days + 60-day 15-minute bar-by-bar verification)**.

---

## 📌 Core Philosophy: Sniper Cadence Over Daily Overtrading

- **No Mandatory Daily Trades**: High-frequency daily intraday trading (`~6.3 trades/week`) suffers from a `45.1%–55.1%` win rate and `-25%` to `-99%` drawdowns due to sideways market chop and opening auction noise.
- **High-Conviction Sniper Cadence**: This engine deploys **5 Regime-Specialized Sniper Logics (`L1`–`L5`)** that fire **only when Market Regime + Sectoral/Macro Shift + Structural Support (`S1`) + Intraday Volume Expansion align** (`~0.13 to 0.43 trades/week` per logic, or **`~1.26 trades/week` combined**).
- **Multi-Layer Downside Risk Protection**:
  1. **Structural Support Floor (`S1`)**: Initial stop-loss is anchored below the **9:15–9:45 Opening Range Low (`ORB_Low`) / 20-Day Structural Support `S1` minus a `0.15x–0.18x ATR` cushion**—never inside normal 15-minute bid-ask noise.
  2. **Chronological 4-Stage High-Water-Mark (`HWM`) Trailing Stop**: Tracks `Last_Evaluated_Bar_TS` and `High_Water_Mark` strictly forward in time (`Stage 1: +0.80 ATR -> +0.18 ATR Breakeven+`, `Stage 2: +1.25 ATR -> +0.65 ATR`, `Stage 3: +1.75 ATR -> +1.15 ATR`, `Stage 4: +2.20 ATR -> HWM - 0.50 ATR Chandelier`).
  3. **3:10 PM Smart Intraday-to-Swing Conversion Gate**: Evaluates stalled 3:10 PM intraday positions against 5 institutional criteria (`84.90%` 3-year empirical win rate).

---

## 🔬 1. Why Production Trades Failed in `smart-etf-tracker`

Full forensic report: [`PRODUCTION_FORENSICS.md`](PRODUCTION_FORENSICS.md) | Raw audit ledger: [`data/production_trades_forensic_audit.csv`](data/production_trades_forensic_audit.csv)

1. **Bug #1 — Retroactive 5-Minute Trailing Stop State Corruption (`evaluate_trade_exits`)**:
   - On `2026-10-27`, `DIVISLAB` (entered at `₹9,305.50`) rallied to **`₹9,505.50` (`+2.15%`)** and `NYKAA` (entered at `₹334.60`) rallied to **`₹339.40` (`+1.43%`)**.
   - At `15:01:40 IST`, `PAPER_TRADE_3PM` upgraded and overwrote `Stop_Loss` in `paper_trades.csv` to `₹9,416.61` (`DIVISLAB`) and `₹336.11` (`NYKAA`).
   - Ten minutes later at `15:11:32 IST`, `INTRADAY_SQUAREOFF` re-walked all 5m bars starting from `09:50 AM` using the afternoon `Stop_Loss` (`₹9,416.61` and `₹336.11`), comparing it against the **09:50 AM morning candle low** (`₹9,273.50` and `₹334.50`) and falsely stopping out two winning trades for `-0.34%` and `-0.03%`!
2. **Flaw #2 — Premature Day-1 Breakeven Choke Inside Intraday Noise**:
   - `MAFANG` (Short) was choked out at `12:51 IST` on Day 1 for `+0.21%` before moving **`+14.20%` MFE**; `AJANTPHARM` (`-0.09%` exit -> **`+6.00%` MFE**), `DIVISLAB` (`+0.51%` exit -> **`+3.95%` MFE**), and `MCX` (`-0.06%` exit -> **`+1.97%` MFE`) suffered the same premature ratchet choke.
3. **Flaw #3 — Forced Daily Candidate Quota in Sideways/Bear Regimes**:
   - `get_top_conviction_candidates` forced fallback picks to fill daily quotas during `🟡 Sideways` and `🔴 Bear Market` regimes.
4. **Flaw #4 — Arbitrary ATR Stops Above Structural Support `S1`**:
   - `CHENNPETRO` was stopped out at `₹1,378.93` (`-3.80%`) just `₹1.23` above its intraday low (`₹1,377.70`, which held above 20-day `S1` support at `₹1,362`) and immediately surged **`+10.37%`**.

---

## 🔄 2. Prediction of Success: Converting Intraday Trades into Swing Trades (3-Year Study)

We analyzed **4,955 historical intraday signals over 3 years** ([`data/production_forensics_swing_study.csv`](data/production_forensics_swing_study.csv) & [`data/swing_conversion_summary.json`](data/swing_conversion_summary.json)):

| Intraday-to-Swing Policy at 3:10 PM IST | 3Y Sample Size | Win Rate (%) | Avg Return (%) | Institutional Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **1. Raw Daily Intraday (No Conversion)** | `4,955` | `45.11%` | `-0.055%` | ❌ High-frequency daily intraday loses money |
| **2. Converting Stopped-Out Intraday Losers ("Bag-Holding")** | `354` | `35.38%` | `-1.528%` | 🚨 **Never convert an intraday loser that broke `S1` support** |
| **3. Unconditional Conversion of All Stalled 3:10 PM Trades** | `4,601` | `60.70%` | `+0.382%` | ⚠️ Mediocre edge; exposes capital to overnight gaps |
| **4. 5-Gate Smart Structural-Support Swing Conversion** | **`298`** | **`84.90%`** | **`+1.960%`** | ✅ **High-Conviction Rescue (`85.03%` Bull / `84.68%` Sideways)** |

---

## 🏆 3. Top 5 Institutional Sniper Logics (3-Year Backtest: 2023–2026)

Full report: [`BACKTEST_REPORT_3Y.md`](BACKTEST_REPORT_3Y.md) | Summary CSV: [`data/backtest_3y_results.csv`](data/backtest_3y_results.csv) | 3Y Trade Ledger: [`data/backtest_trade_ledger_3y.csv`](data/backtest_trade_ledger_3y.csv)

| Rank | Logic ID | Strategy & Shift Focus | Best Market Regime | 3Y Trades | Cadence | Intraday Win Rate | Avg Return | Profit Factor | Max DD |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `L1_SECTOR_ORB_SNIPER` | **Sectoral Shift + NR7 Compression ORB Sniper** | `BULL_EXPANSION` & `SECTOR_SHIFT` | `21` | `0.13 / wk` | **`95.24%`** | **`+1.570%`** | **`375.55`** | **`-0.09%`** |
| **#2** | `L4_DEFENSIVE_RS_SNIPER` | **Bear/High-VIX Defensive RS Fortress Sniper** | `BEAR_OR_CORRECTION` & `HIGH_VIX` | `38` | `0.24 / wk` | **`92.11%`** | **`+1.228%`** | **`54.20`** | **`-0.40%`** |
| **#3** | `L5_GLOBAL_MACRO_SNIPER` | **International & Overnight Global Macro Sniper** | `INTERNATIONAL_CUE` & `BULL` | `40` | `0.26 / wk` | **`95.00%`** | **`+1.500%`** | **`27.49`** | **`-1.27%`** |
| **#4** | `L2_SUPPORT_SPRING_SNIPER` | **Structural S1 + 50DMA Institutional Spring Sniper** | `SIDEWAYS_ROTATION` & Pullbacks | `67` | `0.43 / wk` | **`86.57%`** | **`+1.572%`** | **`25.11`** | **`-1.60%`** |
| **#5** | `L3_CATALYST_VWAP_SNIPER` | **Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper** | `CATALYST_PEAD` (All Regimes) | `31` | `0.20 / wk` | **`83.87%`** | **`+1.194%`** | **`21.47`** | **`-0.53%`** |

---

## 🚀 Quickstart & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Live Intraday Sniper Scan + Chronological GTT Check
python intraday_daemon.py --mode FULL_CYCLE

# 3. Launch Interactive Streamlit Dashboard
streamlit run app.py

# 4. Re-run 3-Year Historical Multi-Regime Backtest & Forensic Study
python forensic_analyzer.py
python backtest_engine_3y.py
```
