# Silver Bullet Backtrader Report (PEPPERSTONE_XAUUSD, 5m)

- Total trades fired: **3**
- Wins: **3**
- Losses: **0**
- Win rate: **100.00%**

## Trades fired (HK time)

| # | Side | Entry time (HK) | Entry signal | Entry | Stop loss | Take profit | Result | PnL |
|---:|:-----|:----------------|:-------------|------:|----------:|------------:|:------|----:|
| 1 | LONG | 2026-01-27 09:05:00 HKT | IFVG_Realtime | 5048.56 | 4989.92 | 5165.84 | WIN | 400.00 |
| 2 | LONG | 2026-01-27 09:10:00 HKT | IFVG_Realtime | 5071.60 | 4989.92 | 5234.96 | WIN | 400.00 |
| 3 | LONG | 2026-01-27 10:05:00 HKT | IFVG_Realtime | 5065.41 | 5013.76 | 5168.71 | WIN | 400.00 |

## Blocked entry triggers (filters)
- Total blocked triggers: **205**

### Block count by direction + filter reason
- SHORT | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **96**
- LONG | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **87**
- LONG | Trend Filter (15M/1H bias not aligned): **19**
- LONG | Time Filter (ICT session not active): **2**
- SHORT | Trend Filter (15M/1H bias not aligned): **1**

### Block count by entry signal + filter reason
- IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **130**
- ICT_Setup01 | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **48**
- IFVG_Realtime | Trend Filter (15M/1H bias not aligned): **13**
- ICT_Setup01 | Trend Filter (15M/1H bias not aligned): **7**
- Fib_OTE | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **3**
- ICT_Setup01 + IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **1**
- ICT_Setup01 | Time Filter (ICT session not active): **1**
- IFVG_Realtime | Time Filter (ICT session not active): **1**
- SB_FVG_Retrace | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB): **1**

### First 20 blocked triggers (HK time)

| HK time | Side | Signal | Rejection reason | HTF POI | Trend | Time |
|:--------|:-----|:-------|:-----------------|:-------:|:-----:|:----:|
| 2026-01-22 00:15:00 HKT | SHORT | Fib_OTE | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 00:30:00 HKT | LONG | Fib_OTE | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | True |
| 2026-01-22 00:55:00 HKT | LONG | Fib_OTE | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | True |
| 2026-01-22 03:25:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 04:00:00 HKT | SHORT | ICT_Setup01 | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 04:20:00 HKT | SHORT | ICT_Setup01 | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 04:30:00 HKT | LONG | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | True |
| 2026-01-22 05:10:00 HKT | LONG | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | False |
| 2026-01-22 05:55:00 HKT | LONG | ICT_Setup01 | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | False |
| 2026-01-22 07:00:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | False |
| 2026-01-22 08:20:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | False |
| 2026-01-22 09:00:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 09:15:00 HKT | LONG | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | True |
| 2026-01-22 09:40:00 HKT | LONG | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | True |
| 2026-01-22 10:00:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 10:20:00 HKT | SHORT | ICT_Setup01 | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 10:30:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 16:05:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 16:15:00 HKT | SHORT | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | True | True |
| 2026-01-22 17:00:00 HKT | LONG | IFVG_Realtime | HTF POI Filter (no prior-10-bar touch of any 1H/4H OB) | False | False | True |