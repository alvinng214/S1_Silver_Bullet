# cTrader cBot Creation Notes

## What a cBot is
- A **cBot** is an automated trading program in cTrader.
- You can implement cBots in **C#** or **Python** with access to the cTrader trading API.
- Typical lifecycle: **create → edit → save/build → add instance → start**, with optional **backtesting** and **optimisation**.

## Step-by-step: Create a cBot
1. Open **cTrader Algo** and go to the **cBots** tab.
2. Click **New**.
3. Enter a cBot name.
4. Choose language (**C#** or **Python**).
5. Choose creation mode:
   - **From scratch** (minimal template)
   - **Using a template** (prebuilt logic + parameters)
6. Click **Create** to open the editor.
7. Edit code, then **Save** and **Build**.

## Core structure in generated templates
A cBot usually includes lifecycle hooks:
- `OnStart()` / `on_start()` for initialization.
- `OnTick()` / `on_tick()` for per-tick logic (more CPU intensive).
- `OnBar()` / `on_bar_closed()` for bar-based logic (often preferred for lower load).
- `OnStop()` / `on_stop()` for cleanup.

In C#, a class is typically decorated with a `Robot` attribute, e.g. timezone/access rights settings.

## Run a cBot (instances)
- You run a cBot by launching one or more **instances**.
- Each instance can have different settings (account, symbol, timeframe, custom parameters).
- Windows/Mac: support local and cloud instances.
- Web/Mobile: cloud instances only.

## Manage instance parameters
When creating or editing an instance, configure:
- Trading account
- Symbol
- Timeframe
- Strategy-specific custom parameters (defined by cBot author)

Best practice: edit parameters while instance is stopped, then restart.

## Backtesting essentials
- Backtesting simulates the cBot on historical data and does not use real funds.
- Important settings include:
  - Starting capital
  - Commission
  - Data source (tick, M1/H1 server bars, CSV M1 data)
  - Spread model (fixed/random)
- Supports visual and non-visual modes.
- Produces reports: equity, trade stats, positions, orders, logs, etc.

## Optimisation essentials
- Optimisation runs multiple backtests across parameter combinations.
- Select optimisable parameters and criteria (standard or custom via `GetFitness`).
- Useful for finding parameter sets that improve selected performance metrics.

## cBot code samples page highlights
Reference examples exist for:
- Synchronous order operations
- Asynchronous order operations
- Position and pending-order management
- Callback patterns for async methods

## Practical workflow
1. Create from template if new to cTrader.
2. Replace template signal logic with your strategy.
3. Keep heavy logic on bar events unless tick-level reactivity is necessary.
4. Backtest with realistic spreads/commission.
5. Optimise cautiously (avoid overfitting).
6. Run with conservative risk settings first.
