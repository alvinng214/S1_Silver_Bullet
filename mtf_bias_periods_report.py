"""MTF Bias Report - Consolidated into Periods.

Uses the same trend calculation as Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.py
to match Pine Script request.security(..., lookahead_on) behavior exactly.

Groups consecutive bullish/bearish bars into date ranges.
"""

import pandas as pd
from datetime import timedelta

# Import the trend calculation function from the Smart Money Zones module
from Smart_Money_Zones__FVG___OB____MTF_Trend_Panel import _trend_series


def consolidate_to_periods(series: pd.Series, interval_minutes: int):
    """Convert boolean Series to periods (start, end) for consecutive True values.

    Args:
        series: Boolean pandas Series with datetime index (True = bias active)
        interval_minutes: Expected interval between consecutive bars

    Returns:
        List of (start, end) datetime tuples for consecutive True periods
    """
    if series.empty or not series.any():
        return []

    # Get timestamps where the condition is True
    active_times = series[series].index.tolist()
    if not active_times:
        return []

    periods = []
    period_start = active_times[0]
    period_end = active_times[0]

    # Allow some tolerance for gaps (up to 2x the interval to handle weekends/gaps)
    max_gap = timedelta(minutes=interval_minutes * 2)

    for dt in active_times[1:]:
        gap = dt - period_end

        # If this datetime is consecutive (or within acceptable gap for market hours)
        if gap <= max_gap:
            period_end = dt
        else:
            # Save current period and start a new one
            periods.append((period_start, period_end))
            period_start = dt
            period_end = dt

    # Don't forget the last period
    periods.append((period_start, period_end))

    return periods


def main() -> None:
    ma_period = 50

    # Load data
    print("Loading data...")
    data = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    data["datetime"] = pd.to_datetime(data["time"])
    data = data.set_index("datetime").sort_index()

    # Rename columns to match expected format
    data = data.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    })

    print(f"Data range: {data.index.min()} to {data.index.max()}")
    print(f"Total bars: {len(data)}")

    # Calculate MTF trends using the same function as Smart Money Zones
    # This matches Pine Script request.security(..., lookahead_on) behavior
    print("\nCalculating MTF trends (matching Pine Script lookahead_on behavior)...")

    timeframes = {
        "15m": ("15min", 15),
        "1h": ("60min", 60),
    }

    for tf_name, (tf_code, interval) in timeframes.items():
        print(f"\n{'='*60}")
        print(f"{tf_name.upper()} TREND ANALYSIS (close > SMA{ma_period})")
        print(f"{'='*60}")

        # Calculate trend using the corrected function
        trend = _trend_series(data, tf_code, ma_period)

        # Bullish = True, Bearish = False (and close < SMA)
        bullish = trend

        # For bearish, we need close < SMA, not just "not bullish"
        # Recalculate to get the actual bearish condition
        resampled = data.resample(tf_code, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna()
        ma = resampled["close"].rolling(window=ma_period).mean()
        bearish_htf = resampled["close"] < ma

        # Handle gaps
        if not bearish_htf.empty:
            full_index = pd.date_range(
                start=bearish_htf.index.min(),
                end=bearish_htf.index.max(),
                freq=tf_code,
                tz=bearish_htf.index.tz,
            )
            bearish_htf = bearish_htf.reindex(full_index).ffill()

        # Shift forward by 1 period so signal appears AFTER bar closes
        bearish_htf = bearish_htf.shift(1)

        bearish = bearish_htf.reindex(data.index, method="ffill").fillna(False)

        # Bullish periods
        print(f"\n{tf_name.upper()} BULLISH PERIODS")
        print("-" * 40)
        bullish_periods = consolidate_to_periods(bullish, interval)
        print(f"Total: {len(bullish_periods)} periods\n")
        for start, end in bullish_periods:
            if start == end:
                print(f"  {start.isoformat()}")
            else:
                print(f"  {start.isoformat()} to {end.isoformat()}")

        # Bearish periods
        print(f"\n{tf_name.upper()} BEARISH PERIODS")
        print("-" * 40)
        bearish_periods = consolidate_to_periods(bearish, interval)
        print(f"Total: {len(bearish_periods)} periods\n")
        for start, end in bearish_periods:
            if start == end:
                print(f"  {start.isoformat()}")
            else:
                print(f"  {start.isoformat()} to {end.isoformat()}")


if __name__ == "__main__":
    main()
