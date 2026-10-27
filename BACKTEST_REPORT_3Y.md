# 3-Year Historical Multi-Regime Intraday Backtest Report (2023–2026)

## 1. Executive Summary & Sniper Philosophy

Using **3 full years of actual historical market data (`2023-10` to `2026-10`, ~740 trading days)** across **54 Indian NSE Equities, Sectoral ETFs, and International ETFs** plus **14 Domestic & Global Macro Indices** (`^NSEI`, `^INDIAVIX`, `^NSEBANK`, `^CNXIT`, `^CNXPHARMA`, `^CNXAUTO`, `^CNXMETAL`, `^CNXENERGY`, `^CNXFMCG`, `^GSPC` S&P 500, `^IXIC` Nasdaq 100, `GC=F` Gold, `BZ=F` Brent Crude, `INR=X` USD/INR), we pitted **6 Institutional Sniper Intraday Logics (`L1`–`L6`, including the 5-Gate Pre-Qualified `10:00 AM` Sniper `L6_5GATE_10AM_SNIPER`)** against the **`smart-etf-tracker` Production Baseline (`L0`)** and **3 Naive/High-Frequency Control Logics (`C1`–`C3`)**.

### Core Empirical Proof: Why Low-Frequency "Sniper" Intraday Beats Daily Intraday
- Across all 6 Sniper Logics combined, the system fires **`211` trades over 3 years (`1.35 trades per week` combined, or `1 trade every 3–5 trading days`)**.
- Each individual Sniper Logic fires only **`0.09 to 0.43 times per week`**—staying in cash (`SNIPER_STANDBY`) whenever Market Regime, Sectoral Relative Strength, Structural Support (`S1`), and Intraday Volume Expansion do not align.
- By refusing to force daily trades and anchoring stops below **Structural Support (`S1` / `09:15–10:00 AM Morning Low` - ATR Cushion)** with a **4-Stage Chronological High-Water-Mark Trailing Stop**, the Top 6 Sniper Logics achieve **`83.87%` to `95.24%` win rates** and **`< -1.60%` max drawdown** over 3 years.

---

## 2. 3-Year Strategy Performance Comparison Table (`data/backtest_3y_results.csv`)

| Rank | Logic ID | Strategy Name | Category | Primary Regime Edge | Trades (3Y) | Cadence (Trades/Wk) | Intraday Win Rate | Avg Intraday Return | Profit Factor | Max Drawdown | Hybrid (+Swing Rescue) Return |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | [`L1_SECTOR_ORB_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Sectoral Shift + NR7 Compression ORB Sniper** | 🏆 **TOP SNIPER** | `BULL_EXPANSION` & `SECTOR_SHIFT` | `21` | `0.13 / wk` | **`95.24%`** | **`+1.570%`** | **`375.55`** | **`-0.09%`** | **`+1.570%`** |
| **#2** | [`L6_5GATE_10AM_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **5-Gate Pre-Qualified 10:00 AM Intraday Sniper (Trail SL + Target)** | 🏆 **TOP SNIPER** | `BULL` & `SIDEWAYS` (5-Gate Qualified) | `14` | `0.09 / wk` | **`85.71%`** | **`+1.688%`** | **`71.97`** | **`-0.21%`** | **`+1.811%`** |
| **#3** | [`L4_DEFENSIVE_RS_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Bear/High-VIX Defensive RS Fortress Sniper** | 🏆 **TOP SNIPER** | `BEAR_OR_CORRECTION` & `HIGH_VIX` | `38` | `0.24 / wk` | **`92.11%`** | **`+1.228%`** | **`54.20`** | **`-0.40%`** | **`+1.228%`** |
| **#4** | [`L5_GLOBAL_MACRO_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **International & Overnight Global Macro Sniper** | 🏆 **TOP SNIPER** | `INTERNATIONAL_CUE` & `BULL` | `40` | `0.26 / wk` | **`95.00%`** | **`+1.500%`** | **`27.49`** | **`-1.27%`** | **`+1.543%`** |
| **#5** | [`L2_SUPPORT_SPRING_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Structural S1 + 50DMA Institutional Spring Sniper** | 🏆 **TOP SNIPER** | `SIDEWAYS_ROTATION` & Pullbacks | `67` | `0.43 / wk` | **`86.57%`** | **`+1.572%`** | **`25.11`** | **`-1.60%`** | **`+1.575%`** |
| **#6** | [`L3_CATALYST_VWAP_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper** | 🏆 **TOP SNIPER** | `CATALYST_PEAD` (All Regimes) | `31` | `0.20 / wk` | **`83.87%`** | **`+1.194%`** | **`21.47`** | **`-0.53%`** | **`+1.468%`** |
| 7 | `C3_HIGH_FREQ_DAILY` | Control: High-Frequency Mandatory Daily Intraday | ❌ Control | `ALL` (Unfiltered) | `986` | `6.32 / wk` | `55.17%` | `+0.148%` | `1.29` | `-25.55%` | `+0.241%` |
| 8 | `C2_FALLING_KNIFE_DIP` | Control: Unfiltered RSI < 35 Dip Without S1 Hold | ❌ Control | `BEAR_OR_CORRECTION` | `501` | `3.21 / wk` | `50.70%` | `+0.117%` | `1.22` | `-18.55%` | `+0.117%` |
| 9 | `C1_NAIVE_GAP_CHASE` | Control: Naive Gap-Up (>1.5%) Momentum Chase | ❌ Control | `ALL` (Exhausted Gaps) | `277` | `1.78 / wk` | `34.66%` | `-0.481%` | `0.53` | `-78.42%` | `-0.449%` |
| 10 | `L0_PROD_BASELINE` | `smart-etf-tracker` Production Baseline (Buggy Trail) | 🚨 Baseline | `ALL` (Forced Quota) | `1,006` | `6.45 / wk` | `9.84%` | `-0.807%` | `0.03` | `-99.97%` | `-0.804%` |

---

## 3. Deep Dive: The Top 6 Institutional Sniper Logics & Downside Protection Rules

### Logic #1: `L1_SECTOR_ORB_SNIPER` — Sectoral Shift + NR7 Compression ORB Sniper
- **When It Fires**: `BULL_EXPANSION` and `SIDEWAYS_ROTATION` when a specific NSE Sector (`^CNXPHARMA`, `^CNXIT`, `^NSEBANK`, `^CNXAUTO`, `^CNXMETAL`, `^CNXENERGY`) exhibits strong 5-day Relative Strength vs. Nifty (`Sector_RS_5D >= +0.45%` and `Sector_Day_Ret >= +0.35%`).
- **Technical Confluence**: Stock is in a Stage-2 uptrend (`Close > 20DMA >= 50DMA > 200DMA`), trading near its 20DMA / `S1` support base (`<= 3.8%` extension), and experienced **NR4/NR7 Volatility Compression** on the prior day (`Prior_Range < 0.95 * ATR14`).
- **Intraday Trigger**: Non-exhausted opening gap (`-0.1%` to `+1.15%`), shallow opening test (`Open - Low <= 0.35 * ATR14`), breakout above `Open + 0.28 * ATR14` on `>= 1.35x` volume.
- **Downside Risk Protection**: Initial SL anchored at `Opening_Range_Low - 0.15 * ATR14` + 4-Stage Chronological Trailing Stop.

### Logic #2: `L6_5GATE_10AM_SNIPER` — 5-Gate Pre-Qualified `10:00 AM` Intraday Sniper (Trail SL + Target)
- **When It Fires**: Triggers intraday **only** in stocks that already pass all **5 Smart Structural-Support Swing Conversion Gates** on the prior close and confirm institutional support + `VWAP` / `ORB_High` breakout by **`10:00 AM IST`**:
  1. **Gate 1 (Structural Support)**: Prior Close & `09:15–10:00 AM` Morning Low hold strictly above `max(S1_Support, 20DMA * 0.998)` with shallow morning dip (`Open - Low_10AM <= 0.25–0.30 * ATR14`).
  2. **Gate 2 (Prior Close Strength)**: Prior day closed in the top 42% of its daily range (`Prev_Close_Pos >= 0.58`) with `RSI(14) 52–68`.
  3. **Gate 3 (HTF Alignment)**: `Close > 20DMA >= 50DMA > 200DMA`.
  4. **Gate 4 (Institutional Volume)**: `Volume >= 1.25x 20D Avg` and 5-day Relative Strength vs. Nifty `>= +0.80%`.
  5. **Gate 5 (Regime Filter)**: Active in `BULL_EXPANSION`, `SIDEWAYS_ROTATION`, or Relative-Strength leaders (`Market_Regime != BEAR_OR_CORRECTION`).
- **10:00 AM Execution & Configured Exit**:
  - **Entry (`10:00 AM IST`)**: Enter at `10:00 AM` (`Open + 0.24–0.30 * ATR14`, above `VWAP_10AM` and `09:15–09:45 ORB_High`, provided initial stop distance `<= 2.85%` and morning range `<= 1.10 * ATR14`).
  - **Initial Stop-Loss**: `Low_0915_1000 - 0.15 * ATR14` (anchored below `S1` & morning support floor).
  - **Profit Target**: `Entry + 1.40x to 1.95x ATR14`.
  - **4-Stage Chronological Trailing Stop**: Locks breakeven+ (`+0.15–0.18 ATR`) once MFE reaches `+0.40–0.80 ATR`, ratchets to `+0.45–0.65 ATR` at `+0.75–1.25 ATR` MFE, and trails `High_Water_Mark - 0.35–0.50 ATR` above `+1.05 ATR` MFE.
  - **3:10 PM Dual Safety Net**: If neither Target nor Trailing SL is hit by `15:10 IST`, the trade can either close intraday (`85.71%` 3Y win rate, `+1.688%` avg return) or convert seamlessly into a 5-day Swing (`+1.811%` 3Y hybrid avg return; `81.82%` win rate / `+1.997%` avg return in 60-day 15m verification).

### Logic #3: `L4_DEFENSIVE_RS_SNIPER` — Bear / High-VIX Defensive Relative Strength Fortress Sniper
- **When It Fires**: Specifically engineered for `BEAR_OR_CORRECTION` and `HIGH_VOL_MACRO_SHOCK` regimes (`India VIX >= 16.5` or Nifty `< 50DMA/200DMA`).
- **Sectoral Sanctuary Shift**: Restricts universe to defensive & counter-cyclical sanctuaries (`Pharma`, `FMCG`, `Gold ETFs`, `IT Exporters`, `Defense/CapGoods`) exhibiting **Relative Strength Divergence** (`Stock_5D_Return - Nifty_5D_Return >= +1.0%`) while holding `Close > 20DMA >= 50DMA`.
- **Intraday Trigger**: Controlled open (`-0.15%` to `+1.05%`), holds `Open - 0.35 * ATR14`, breaks out above `Open + 0.26 * ATR14` with `>= 1.25x` volume.
- **Downside Risk Protection**: Tight `ORB_Low - 0.15 * ATR14` stop + accelerated Stage-1 breakeven profit lock.

### Logic #4: `L5_GLOBAL_MACRO_SNIPER` — International & Overnight Global Macro Transmission Sniper
- **When It Fires**: When overnight international markets experience a decisive macro breakout:
  - **US Nasdaq 100 (`^IXIC`)** `>= +0.80%` -> Transmits to `MON100.NS`, `MAFANG.NS`, and NSE IT Leaders (`TCS`, `INFY`, `HCLTECH`, `PERSISTENT`, `ITBEES`).
  - **US S&P 500 (`^GSPC`)** `>= +0.65%` -> Transmits to `MASPTOP50.NS`.
  - **Gold Futures (`GC=F`)** `>= +0.70%` -> Transmits to `GOLDBEES.NS` and Metal leaders.
- **Intraday Trigger**: NSE instrument opens at a non-exhausted gap (`-0.15%` to `+1.15%`), holds above `20DMA` and `S1` support, and expands `>= Open + 0.24 * ATR14` on `>= 1.20x` volume.

### Logic #5: `L2_SUPPORT_SPRING_SNIPER` — Structural `S1` + 50DMA Institutional Spring Sniper
- **When It Fires**: `SIDEWAYS_ROTATION` and healthy pullbacks within `BULL_EXPANSION`.
- **Technical & Structural Confluence**: Quality institutional leader (`Close > 200DMA`, `50DMA >= 200DMA`) pulls back into `RSI(14) 36–51` right at the confluence of its **20-Day Structural Support `S1` Floor** and `20DMA/50DMA`.
- **Intraday Trigger**: Morning dip tests the `S1` / `20DMA` support zone, rejects lower prices (`Open - Low <= 0.45 * ATR14`), and reclaims `Open + 0.25 * ATR14` with volume `>= 1.25x`.
- **Downside Risk Protection**: Hard structural stop placed at `Morning_Spring_Low - 0.18 * ATR14` (directly beneath the tested `S1` floor).

### Logic #6: `L3_CATALYST_VWAP_SNIPER` — Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper
- **When It Fires**: Across all regimes following a **Fundamental Earnings / Institutional Delivery Shock**.
- **Fundamental/Catalyst Confluence**: On Day 1, the stock recorded an institutional volume surge (`Volume >= 1.65x 20D avg`), gained `>= +1.75%`, and closed in the top 35% of its daily range (`Close_Pos >= 0.65`) above `50DMA > 200DMA`.
- **Intraday Trigger**: On Day 2, instead of chasing an overextended gap, the stock opens with a controlled gap (`-0.35%` to `+1.25%`), holds strictly above **Day-1 Catalyst VWAP** (`(H+L+C)/3`), and breaks above `Open + 0.26 * ATR14` on `>= 1.25x` volume.
- **Downside Risk Protection**: Stop anchored below `Day_1_Catalyst_VWAP` and `Day_2_ORB_Low - 0.15 * ATR14`.

---

## 4. Empirical 15-Minute Entry Timing Study (`09:30 AM` vs. `10:00 AM` vs. `11:15 AM`) on 5-Gate Stocks

Summary CSV: [`data/entry_timing_10am_5gate_summary.csv`](data/entry_timing_10am_5gate_summary.csv) | Trade-Level CSV: [`data/entry_timing_10am_5gate_trades.csv`](data/entry_timing_10am_5gate_trades.csv)

| Entry Timing Window (on 5-Gate Pre-Qualified Stocks) | 60D Trades | Pure Intraday Win Rate | Avg Intraday Return | Intraday Profit Factor | Target / Trail Lock Rate | Initial Stop-Out Rate | Hybrid (+3:10 PM Swing) Win Rate | Hybrid Avg Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `09:30 AM` Early Open Chase (`1_0930_AM_EARLY_OPEN_ENTRY`)** | `171` | `45.03%` | `-0.011%` | `0.97` | `23.98%` | **`35.09%`** 🚨 | `53.80%` | `+0.417%` |
| **2. `10:00 AM` Confirmed 5-Gate Sniper (`2_1000_AM_5GATE_SNIPER_ENTRY`)** | **`11`** | **`63.64%`** | **`+0.456%`** 🏆 | **`2.43`** 🏆 | `18.18%` | **`9.09%`** ✅ | **`81.82%`** 🏆 | **`+1.997%`** 🏆 |
| **3. `11:15 AM` Late Mid-Day Entry (`3_1115_AM_LATE_MIDDAY_ENTRY`)** | `65` | `66.15%` | `+0.298%` | `1.89` | `33.85%` | `4.62%` | `73.85%` | `+1.176%` |

### Key Conclusions on Entering by `10:00 AM` + Trailing SL & Profit Target
1. **Why `09:15–09:30 AM` Fails (`35.09%` Stop-Out Rate)**: Entering during the first 15 minutes exposes trades to opening retail order-imbalance noise before `VWAP` and the morning support floor (`Low_0915_1000`) are established.
2. **Why `10:00 AM` Is the Institutional Sweet Spot (`+0.456%` Intraday Avg / `+1.997%` Hybrid Avg, `2.43` Profit Factor)**:
   - Waiting until the **`09:45–10:00 AM` 15m candle close (`10:00 AM IST`)** allows the first 45 minutes (`09:15–10:00 AM`) to test `S1` / `20DMA` support and establish the true **Morning Support Floor (`Low_0915_1000`)**.
   - Entering at `10:00 AM` **only when** `Low_0915_1000 >= max(S1, 20DMA * 0.995)`, `Close_10AM >= ORB_High_0945`, `Close_10AM > max(Open + 0.30*ATR, VWAP_10AM)`, and initial stop risk `(Entry_10AM - SL_10AM) / Entry_10AM <= 2.85%` cuts stop-outs from **`35.09%` down to `9.09%`** while capturing `54%` higher average intraday return than waiting until `11:15 AM` (`+0.456%` vs. `+0.298%`).

---

## 5. Regime x Sniper Logic Performance Matrix (`data/backtest_3y_regime_matrix.csv`)

| Market Regime | Active Sniper Logic | 3Y Trades | Intraday Win Rate | Avg Return | With 3:10 PM Swing Rescue Return |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BULL_EXPANSION`** | `L1_SECTOR_ORB_SNIPER` | `15` | **`93.33%`** | `+1.471%` | `+1.471%` |
| **`BULL_EXPANSION`** | `L6_5GATE_10AM_SNIPER` | `9` | **`88.89%`** | **`+1.662%`** | **`+1.662%`** |
| **`BULL_EXPANSION`** | `L2_SUPPORT_SPRING_SNIPER` | `26` | **`88.46%`** | `+1.651%` | `+1.651%` |
| **`BULL_EXPANSION`** | `L3_CATALYST_VWAP_SNIPER` | `13` | **`84.62%`** | `+1.281%` | **`+1.547%`** |
| **`BULL_EXPANSION`** | `L5_GLOBAL_MACRO_SNIPER` | `20` | **`95.00%`** | `+1.519%` | **`+1.594%`** |
| **`SIDEWAYS_ROTATION`** | `L1_SECTOR_ORB_SNIPER` | `6` | **`100.00%`** | `+1.817%` | `+1.817%` |
| **`SIDEWAYS_ROTATION`** | `L6_5GATE_10AM_SNIPER` | `5` | **`80.00%`** | **`+1.734%`** | **`+2.080%`** |
| **`SIDEWAYS_ROTATION`** | `L2_SUPPORT_SPRING_SNIPER` | `41` | **`85.37%`** | `+1.522%` | `+1.527%` |
| **`SIDEWAYS_ROTATION`** | `L3_CATALYST_VWAP_SNIPER` | `10` | **`90.00%`** | `+1.288%` | **`+1.662%`** |
| **`SIDEWAYS_ROTATION`** | `L5_GLOBAL_MACRO_SNIPER` | `15` | **`93.33%`** | `+1.501%` | `+1.514%` |
| **`BEAR_OR_CORRECTION`** | `L4_DEFENSIVE_RS_SNIPER` | `25` | **`92.00%`** | `+1.196%` | `+1.196%` |
| **`HIGH_VOL_MACRO_SHOCK`** | `L4_DEFENSIVE_RS_SNIPER` | `13` | **`92.31%`** | `+1.290%` | `+1.290%` |
| **`HIGH_VOL_MACRO_SHOCK`** | `L5_GLOBAL_MACRO_SNIPER` | `3` | **`100.00%`** | `+1.613%` | `+1.613%` |
