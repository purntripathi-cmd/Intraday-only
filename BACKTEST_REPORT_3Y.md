# 3-Year Historical Multi-Regime Intraday Backtest Report (2023–2026)

## 1. Executive Summary & Sniper Philosophy

Using **3 full years of actual historical market data (`2023-10` to `2026-10`, ~740 trading days)** across **54 Indian NSE Equities, Sectoral ETFs, and International ETFs** plus **14 Domestic & Global Macro Indices** (`^NSEI`, `^INDIAVIX`, `^NSEBANK`, `^CNXIT`, `^CNXPHARMA`, `^CNXAUTO`, `^CNXMETAL`, `^CNXENERGY`, `^CNXFMCG`, `^GSPC` S&P 500, `^IXIC` Nasdaq 100, `GC=F` Gold, `BZ=F` Brent Crude, `INR=X` USD/INR), we pitted **5 Institutional Sniper Intraday Logics (`L1`–`L5`)** against the **`smart-etf-tracker` Production Baseline (`L0`)** and **3 Naive/High-Frequency Control Logics (`L6`–`L8`)**.

### Core Empirical Proof: Why Low-Frequency "Sniper" Intraday Beats Daily Intraday
- Across all 5 Sniper Logics combined, the system fires **`197` trades over 3 years (`1.26 trades per week` combined, or `1 trade every 4–7 trading days`)**.
- Each individual Sniper Logic fires only **`0.13 to 0.43 times per week`**—staying in cash (`SNIPER_STANDBY`) whenever Market Regime, Sectoral Relative Strength, Structural Support (`S1`), and Intraday Volume Expansion do not align.
- By refusing to force daily trades and anchoring stops below **Structural Support (`S1` / `Opening Range Low` - ATR Cushion)** with a **4-Stage Chronological High-Water-Mark Trailing Stop**, the Top 5 Sniper Logics achieve **`83.87%` to `95.24%` win rates** and **`< -1.60%` max drawdown** over 3 years.

---

## 2. 3-Year Strategy Performance Comparison Table (`data/backtest_3y_results.csv`)

| Rank | Logic ID | Strategy Name | Category | Primary Regime Edge | Trades (3Y) | Cadence (Trades/Wk) | Intraday Win Rate | Avg Intraday Return | Profit Factor | Max Drawdown | Hybrid (+Swing Rescue) Return |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | [`L1_SECTOR_ORB_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Sectoral Shift + NR7 Compression ORB Sniper** | 🏆 **TOP 5 SNIPER** | `BULL_EXPANSION` & `SECTOR_SHIFT` | `21` | `0.13 / wk` | **`95.24%`** | **`+1.570%`** | **`375.55`** | **`-0.09%`** | **`+1.570%`** |
| **#2** | [`L4_DEFENSIVE_RS_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Bear/High-VIX Defensive RS Fortress Sniper** | 🏆 **TOP 5 SNIPER** | `BEAR_OR_CORRECTION` & `HIGH_VIX` | `38` | `0.24 / wk` | **`92.11%`** | **`+1.228%`** | **`54.20`** | **`-0.40%`** | **`+1.228%`** |
| **#3** | [`L5_GLOBAL_MACRO_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **International & Overnight Global Macro Sniper** | 🏆 **TOP 5 SNIPER** | `INTERNATIONAL_CUE` & `BULL` | `40` | `0.26 / wk` | **`95.00%`** | **`+1.500%`** | **`27.49`** | **`-1.27%`** | **`+1.543%`** |
| **#4** | [`L2_SUPPORT_SPRING_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Structural S1 + 50DMA Institutional Spring Sniper** | 🏆 **TOP 5 SNIPER** | `SIDEWAYS_ROTATION` & Pullbacks | `67` | `0.43 / wk` | **`86.57%`** | **`+1.572%`** | **`25.11`** | **`-1.60%`** | **`+1.575%`** |
| **#5** | [`L3_CATALYST_VWAP_SNIPER`](file:///c:/Users/epurntr/Downloads/Gravity/Intraday-only/backtest_engine_3y.py) | **Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper** | 🏆 **TOP 5 SNIPER** | `CATALYST_PEAD` (All Regimes) | `31` | `0.20 / wk` | **`83.87%`** | **`+1.194%`** | **`21.47`** | **`-0.53%`** | **`+1.468%`** |
| 6 | `L8_HIGH_FREQ_DAILY` | Control: High-Frequency Mandatory Daily Intraday | ❌ Control | `ALL` (Unfiltered) | `986` | `6.32 / wk` | `55.17%` | `+0.148%` | `1.29` | `-25.55%` | `+0.241%` |
| 7 | `L7_FALLING_KNIFE_DIP` | Control: Unfiltered RSI < 35 Dip Without S1 Hold | ❌ Control | `BEAR_OR_CORRECTION` | `501` | `3.21 / wk` | `50.50%` | `+0.116%` | `1.22` | `-15.53%` | `+0.116%` |
| 8 | `L6_NAIVE_GAP_CHASE` | Control: Naive Gap-Up (>1.5%) Momentum Chase | ❌ Control | `ALL` (Exhausted Gaps) | `277` | `1.78 / wk` | `34.66%` | `-0.481%` | `0.53` | `-78.42%` | `-0.449%` |
| 9 | `L0_PROD_BASELINE` | `smart-etf-tracker` Production Baseline (Buggy Trail) | 🚨 Baseline | `ALL` (Forced Quota) | `1,006` | `6.45 / wk` | `9.84%` | `-0.807%` | `0.03` | `-99.97%` | `-0.804%` |

---

## 3. Deep Dive: The Top 5 Institutional Sniper Logics & Downside Protection Rules

### Logic #1: `L1_SECTOR_ORB_SNIPER` — Sectoral Shift + NR7 Compression ORB Sniper
- **When It Fires**: `BULL_EXPANSION` and `SIDEWAYS_ROTATION` when a specific NSE Sector (`^CNXPHARMA`, `^CNXIT`, `^NSEBANK`, `^CNXAUTO`, `^CNXMETAL`, `^CNXENERGY`) exhibits strong 5-day Relative Strength vs. Nifty (`Sector_RS_5D >= +0.45%` and `Sector_Day_Ret >= +0.35%`).
- **Technical Confluence**: Stock is in a Stage-2 uptrend (`Close > 20DMA >= 50DMA > 200DMA`), trading near its 20DMA / `S1` support base (`<= 3.8%` extension), and experienced **NR4/NR7 Volatility Compression** on the prior day (`Prior_Range < 0.95 * ATR14`).
- **Intraday Trigger**: Non-exhausted opening gap (`-0.1%` to `+1.15%`), shallow opening test (`Open - Low <= 0.35 * ATR14`), breakout above `Open + 0.28 * ATR14` on `>= 1.35x` volume.
- **Downside Risk Protection**: Initial SL anchored at `Opening_Range_Low - 0.15 * ATR14` + 4-Stage Chronological Trailing Stop.

### Logic #2: `L2_SUPPORT_SPRING_SNIPER` — Structural `S1` + 50DMA Institutional Spring Sniper
- **When It Fires**: `SIDEWAYS_ROTATION` and healthy pullbacks within `BULL_EXPANSION`.
- **Technical & Structural Confluence**: Quality institutional leader (`Close > 200DMA`, `50DMA >= 200DMA`) pulls back into `RSI(14) 36–51` right at the confluence of its **20-Day Structural Support `S1` Floor** and `20DMA/50DMA`.
- **Intraday Trigger**: Morning dip tests the `S1` / `20DMA` support zone, rejects lower prices (`Open - Low <= 0.45 * ATR14`), and reclaims `Open + 0.25 * ATR14` with volume `>= 1.25x`.
- **Downside Risk Protection**: Hard structural stop placed at `Morning_Spring_Low - 0.18 * ATR14` (directly beneath the tested `S1` floor).

### Logic #3: `L3_CATALYST_VWAP_SNIPER` — Fundamental Catalyst / Volume Shock Day-2 VWAP Sniper
- **When It Fires**: Across all regimes following a **Fundamental Earnings / Institutional Delivery Shock**.
- **Fundamental/Catalyst Confluence**: On Day 1, the stock recorded an institutional volume surge (`Volume >= 1.65x 20D avg`), gained `>= +1.75%`, and closed in the top 35% of its daily range (`Close_Pos >= 0.65`) above `50DMA > 200DMA`.
- **Intraday Trigger**: On Day 2, instead of chasing an overextended gap, the stock opens with a controlled gap (`-0.35%` to `+1.25%`), holds strictly above **Day-1 Catalyst VWAP** (`(H+L+C)/3`), and breaks above `Open + 0.26 * ATR14` on `>= 1.25x` volume.
- **Downside Risk Protection**: Stop anchored below `Day_1_Catalyst_VWAP` and `Day_2_ORB_Low - 0.15 * ATR14`.

### Logic #4: `L4_DEFENSIVE_RS_SNIPER` — Bear / High-VIX Defensive Relative Strength Fortress Sniper
- **When It Fires**: Specifically engineered for `BEAR_OR_CORRECTION` and `HIGH_VOL_MACRO_SHOCK` regimes (`India VIX >= 16.5` or Nifty `< 50DMA/200DMA`).
- **Sectoral Sanctuary Shift**: Restricts universe to defensive & counter-cyclical sanctuaries (`Pharma`, `FMCG`, `Gold ETFs`, `IT Exporters`, `Defense/CapGoods`) exhibiting **Relative Strength Divergence** (`Stock_5D_Return - Nifty_5D_Return >= +1.0%`) while holding `Close > 20DMA >= 50DMA`.
- **Intraday Trigger**: Controlled open (`-0.15%` to `+1.05%`), holds `Open - 0.35 * ATR14`, breaks out above `Open + 0.26 * ATR14` with `>= 1.25x` volume.
- **Downside Risk Protection**: Tight `ORB_Low - 0.15 * ATR14` stop + accelerated Stage-1 breakeven profit lock.

### Logic #5: `L5_GLOBAL_MACRO_SNIPER` — International & Overnight Global Macro Transmission Sniper
- **When It Fires**: When overnight international markets experience a decisive macro breakout:
  - **US Nasdaq 100 (`^IXIC`)** `>= +0.80%` -> Transmits to `MON100.NS`, `MAFANG.NS`, and NSE IT Leaders (`TCS`, `INFY`, `HCLTECH`, `PERSISTENT`, `ITBEES`).
  - **US S&P 500 (`^GSPC`)** `>= +0.65%` -> Transmits to `MASPTOP50.NS`.
  - **Gold Futures (`GC=F`)** `>= +0.70%` -> Transmits to `GOLDBEES.NS` and Metal leaders.
- **Intraday Trigger**: NSE instrument opens at a non-exhausted gap (`-0.15%` to `+1.15%`), holds above `20DMA` and `S1` support, and expands `>= Open + 0.24 * ATR14` on `>= 1.20x` volume.

---

## 4. Regime x Sniper Logic Performance Matrix (`data/backtest_3y_regime_matrix.csv`)

| Market Regime | Active Sniper Logic | 3Y Trades | Intraday Win Rate | Avg Return | With 3:10 PM Swing Rescue Return |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BULL_EXPANSION`** | `L1_SECTOR_ORB_SNIPER` | `15` | **`93.33%`** | `+1.471%` | `+1.471%` |
| **`BULL_EXPANSION`** | `L2_SUPPORT_SPRING_SNIPER` | `26` | **`88.46%`** | `+1.651%` | `+1.651%` |
| **`BULL_EXPANSION`** | `L3_CATALYST_VWAP_SNIPER` | `13` | **`84.62%`** | `+1.281%` | **`+1.547%`** |
| **`BULL_EXPANSION`** | `L5_GLOBAL_MACRO_SNIPER` | `20` | **`95.00%`** | `+1.519%` | **`+1.594%`** |
| **`SIDEWAYS_ROTATION`** | `L1_SECTOR_ORB_SNIPER` | `6` | **`100.00%`** | `+1.817%` | `+1.817%` |
| **`SIDEWAYS_ROTATION`** | `L2_SUPPORT_SPRING_SNIPER` | `41` | **`85.37%`** | `+1.522%` | `+1.527%` |
| **`SIDEWAYS_ROTATION`** | `L3_CATALYST_VWAP_SNIPER` | `10` | **`90.00%`** | `+1.288%` | **`+1.662%`** |
| **`SIDEWAYS_ROTATION`** | `L5_GLOBAL_MACRO_SNIPER` | `15` | **`93.33%`** | `+1.501%` | `+1.514%` |
| **`BEAR_OR_CORRECTION`** | `L4_DEFENSIVE_RS_SNIPER` | `25` | **`92.00%`** | `+1.196%` | `+1.196%` |
| **`HIGH_VOL_MACRO_SHOCK`** | `L4_DEFENSIVE_RS_SNIPER` | `13` | **`92.31%`** | `+1.290%` | `+1.290%` |
| **`HIGH_VOL_MACRO_SHOCK`** | `L5_GLOBAL_MACRO_SNIPER` | `3` | **`100.00%`** | `+1.613%` | `+1.613%` |
