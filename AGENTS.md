# AGENTS.md — S1_Silver_Bullet repo context for Codex CLI

Nested git repo inside `STOCKS_Bot/`. Separate origin: `https://github.com/alvinng214/S1_Silver_Bullet.git`

Parent project context: `../AGENTS.md` (read that first for response style + locks).

---

## What lives here

Pine Script translations and source files for ICT / Smart Money Concepts indicators. These are the **canonical reference** for the ICT adapters in `../tools/ict_*.py`.

File types:
- `.py` — Pine → Python translations (consumed by `tools/ict_*.py`)
- `.cs` — cAlgo / cTrader C# ports
- `.pine` / `.txt` — original Pine Script sources
- `.csv` — backtest event logs
- `.png` — chart references

---

## Canonical indicator sources

| Indicator | Source file | Adapter in `../tools/` |
|---|---|---|
| Market Structure MTF Trend | `Market Structure MTF Trend [Pt].py` | `ict_structure.py` |
| Order Blocks & Imbalance MTF | `Order Blocks & Imbalance MTF.py` | `ict_order_blocks.py` (HTF) |
| Smart Money Concepts | `Smart Money Concepts [LuxAlgo].py` | `ict_order_blocks.py` (current-TF) |
| MTF FVG x2 | `MTF FVG x2 [MK].py` | `ict_fvg.py` |
| Liquidity Sweeps UAlgo | `Liquidity Sweeps [UAlgo].py` | `ict_liquidity.py` |
| SMC Target V.35 | `SMC Target Liquidity V.35 ...py` | `ict_liquidity.py` |
| ICT HTF MSS (fadi) | `ICT HTF MSS & Liquidity (fadi).py` | `ict_liquidity.py` |
| Fibonacci OTE Zeiierman | `Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py` | `ict_ote.py` |
| Displacement [MsF] | `Liquidity Engulfing & Displacement [MsF].py` | `ict_displacement.py` |
| BPR TFO mod | `Balanced Price Range - BPR [TFO]mod.py` | `ict_bpr.py` |
| FVG + IFVG TradingFinder | `S1_Silver_Bullet/ICT Balanced Price Range [TradingFinder]  BPR  FVG + IFVG.py` | `ict_bpr.py` |
| CISD cdikici71 | `cd_sweep&cisd_Cx.py` (variant) | `ict_cisd.py` |

---

## Editing rules

- Do NOT modify `../tools/ict_*.py` adapters without cross-checking the source `.py` here
- If a Pine source is ported with parity claims (e.g. BPR TFO mod "100% TV parity"), regression-test before changing
- Backtest events (`.csv` files) are reference artefacts — don't delete

## Git discipline

- This repo is independent. `git status` from `STOCKS_Bot/` does not show changes here.
- After editing any file in `S1_Silver_Bullet/`:
  1. `cd S1_Silver_Bullet && git status`
  2. `git add <files> && git commit -m "..."`
  3. `git push origin main` (auto-approved)
- Always check both repos before saying "pushed" to user
