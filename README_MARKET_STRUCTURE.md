# Market Structure MTF Trend - Python/Backtrader Implementation

This is a Python translation of the Pine Script indicator "Market Structure MTF Trend [Pt]" for backtesting with Backtrader.

## Overview

The Market Structure MTF Trend indicator identifies market structure shifts based on pivot points. It detects two key events:

- **CHoCH (Change of Character)**: A trend reversal when price breaks structure in the opposite direction
- **BoS (Break of Structure)**: Trend continuation when price breaks structure in the same direction

## Key Concepts

### Market Structure Detection

The indicator uses **candle CLOSES** (not wicks) to determine structure breaks:

1. **Bullish Structure Break**: When close crosses above the last pivot high
   - If already in bullish trend → BoS (continuation)
   - If in bearish trend → CHoCH (reversal to bullish)

2. **Bearish Structure Break**: When close crosses below the last pivot low
   - If already in bearish trend → BoS (continuation)
   - If in bullish trend → CHoCH (reversal to bearish)

### Pivot Detection

Pivots are detected using a symmetrical lookback/forward window:
- **Pivot High**: A high that is higher than N bars before and N bars after it
- **Pivot Low**: A low that is lower than N bars before and N bars after it

Default pivot strength: 15 bars

## Files Created

```
S1_Silver_Bullet/
├── indicators/
│   └── market_structure_mtf.py       # Market Structure indicator
├── strategies/
│   └── market_structure_strategy.py  # Trading strategies
├── backtest_market_structure.py      # Main backtest script
└── README_MARKET_STRUCTURE.md        # This file
```

## Installation

```bash
pip install backtrader pandas matplotlib
```

## Usage

### Basic Backtest

Run the backtest with default parameters:

```bash
python backtest_market_structure.py
```

### Custom Parameters

Edit `backtest_market_structure.py` to customize:

```python
cerebro, strat = run_backtest(
    data_df,
    strategy_class=SimpleMarketStructureStrategy,
    params={
        'pivot_strength': 15,  # Pivot detection sensitivity
        'print_signals': True  # Show trade signals
    }
)
```

### Using the Indicator in Your Own Strategy

```python
from indicators.market_structure_mtf import MarketStructureIndicator

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.ms = MarketStructureIndicator(
            self.data,
            pivot_strength=15
        )

    def next(self):
        # Access indicator values
        trend = self.ms.lines.trend[0]        # 1=bullish, -1=bearish
        choch = self.ms.lines.choch[0]        # CHoCH signal
        bos = self.ms.lines.bos[0]            # BoS signal
        pivot_high = self.ms.lines.pivot_high[0]
        pivot_low = self.ms.lines.pivot_low[0]

        # Your trading logic here
        if choch > 0:  # Bullish CHoCH
            self.buy()
        elif choch < 0:  # Bearish CHoCH
            self.sell()
```

## Indicator Outputs

### MarketStructureIndicator Lines

| Line | Description | Values |
|------|-------------|--------|
| `trend` | Current market trend | 1 (bullish), -1 (bearish), 0 (undefined) |
| `choch` | Change of Character signal | 1 (bullish), -1 (bearish), 0 (none) |
| `bos` | Break of Structure signal | 1 (bullish), -1 (bearish), 0 (none) |
| `pivot_high` | Last confirmed pivot high | Price level or NaN |
| `pivot_low` | Last confirmed pivot low | Price level or NaN |

## Strategies Included

### 1. SimpleMarketStructureStrategy

A basic strategy that:
- Enters LONG on bullish CHoCH
- Enters SHORT on bearish CHoCH
- Closes opposite positions before entering new ones

### 2. MarketStructureStrategy

An advanced strategy with:
- Risk management (% of capital per trade)
- ATR-based stop losses
- Risk:reward ratio take profits
- Position sizing

## Important Notes

### This is NOT a Complete Trading System

As outlined in the Silver Bullet Strategy guide, market structure detection alone is **insufficient** for profitable trading. The complete S1 strategy requires:

1. ✅ **Market Structure Trend** (Step 1 - THIS IMPLEMENTATION)
2. ❌ HTF Points of Interest (OB/FVG) - Step 2
3. ❌ Draw on Liquidity - Step 3
4. ❌ Session Levels - Step 4
5. ❌ Killzone Window - Step 5
6. ❌ Liquidity Sweep Detection - Step 6
7. ❌ FVG Detection - Step 8
8. ❌ Entry Rules - Step 9-11

### Why the Basic Backtest Shows Losses

The current implementation trades EVERY CHoCH signal without:
- HTF bias filter (only trade with higher timeframe trend)
- Liquidity sweep confirmation
- FVG entry zones
- Session/killzone timing
- Proper risk management

This results in many false signals and losses, which is expected.

### Next Steps

To create a complete S1 Silver Bullet backtesting system, you need to implement:

1. **Higher Timeframe Bias**: Multi-timeframe market structure analysis
2. **Liquidity Detection**: Identify session highs/lows, equal highs/lows
3. **Sweep Detection**: Detect liquidity raids (wick beyond level, close back inside)
4. **FVG Detection**: Identify fair value gaps during MSS displacement
5. **Session Filtering**: Only trade during London/NY Silver Bullet windows
6. **Entry Logic**: Enter at FVG zones after sweep + MSS confirmation

## Customization

### Adjusting Pivot Strength

Smaller values = more sensitive (more signals, more noise):
```python
pivot_strength = 10  # More pivots detected
```

Larger values = less sensitive (fewer signals, stronger pivots):
```python
pivot_strength = 20  # Fewer, stronger pivots
```

### Multi-Timeframe Analysis

To implement true MTF analysis like the original Pine Script:

1. Create multiple data feeds with different timeframes
2. Pass each to separate MarketStructureIndicator instances
3. Use request.security equivalent (resampling in backtrader)

Example:
```python
# Resample to 1H
data_1h = cerebro.resampledata(data_5m, timeframe=bt.TimeFrame.Minutes, compression=60)

# Add indicators for each timeframe
ms_5m = MarketStructureIndicator(data_5m, pivot_strength=15)
ms_1h = MarketStructureIndicator(data_1h, pivot_strength=15)
```

## Performance Metrics Explained

The backtest outputs several metrics:

- **Total Return**: Overall % gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of winning trades
- **Average Win/Loss**: Average profit/loss per trade

## Testing Different Parameters

You can test different pivot strengths to find optimal values:

```python
pivot_strengths = [10, 15, 20, 25]
results = {}

for pivot_strength in pivot_strengths:
    cerebro, strat = run_backtest(
        data_df,
        params={'pivot_strength': pivot_strength}
    )
    results[pivot_strength] = cerebro.broker.getvalue()
```

## Data Format

The script expects CSV with these columns:
- `time`: Datetime in ISO format
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price

Example:
```csv
time,open,high,low,close
2025-12-25T01:20:00+08:00,4483.58,4484.89,4477.41,4478.59
```

## Troubleshooting

### "ValueError: 'time' is not in list"
- Make sure datetime is set as index in load_data()
- Check PandasDataCustom params has `datetime=None`

### "Not enough bars for pivot detection"
- Reduce pivot_strength parameter
- Use more data (need 2*pivot_strength bars minimum)

### Strategy not executing trades
- Check if CHoCH signals are being generated
- Set `print_signals=True` to see signal output
- Verify data has sufficient bars

## References

- Original Pine Script: `Market Structure MTF Trend [Pt].txt`
- Strategy Guide: `SILVER_BULLET_S1_COMPLETE_GUIDE_V2_UPDATED.txt`
- Backtrader Documentation: https://www.backtrader.com/

## License

This is a translation/implementation of the original Pine Script indicator for educational purposes.
