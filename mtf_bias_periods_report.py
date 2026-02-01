"""MTF Bias Report - Consolidated into Periods.

Groups consecutive bullish/bearish bars into date ranges.
"""

import backtrader as bt
from datetime import timedelta


class MtfBiasStrategy(bt.Strategy):
    params = dict(
        ma_period=50,
    )

    def __init__(self):
        self.data_15m = self.datas[1]
        self.data_1h = self.datas[2]

        self.ma_15m = bt.indicators.SimpleMovingAverage(self.data_15m.close, period=self.p.ma_period)
        self.ma_1h = bt.indicators.SimpleMovingAverage(self.data_1h.close, period=self.p.ma_period)

        self._last_len_15m = 0
        self._last_len_1h = 0

        self.records = {
            "15m": {"bullish": [], "bearish": []},
            "1h": {"bullish": [], "bearish": []},
        }

    def next(self):
        if len(self.data_15m) > self._last_len_15m:
            self._last_len_15m = len(self.data_15m)
            if len(self.data_15m) >= self.p.ma_period:
                dt = self.data_15m.datetime.datetime(0)
                if self.data_15m.close[0] > self.ma_15m[0]:
                    self.records["15m"]["bullish"].append(dt)
                elif self.data_15m.close[0] < self.ma_15m[0]:
                    self.records["15m"]["bearish"].append(dt)

        if len(self.data_1h) > self._last_len_1h:
            self._last_len_1h = len(self.data_1h)
            if len(self.data_1h) >= self.p.ma_period:
                dt = self.data_1h.datetime.datetime(0)
                if self.data_1h.close[0] > self.ma_1h[0]:
                    self.records["1h"]["bullish"].append(dt)
                elif self.data_1h.close[0] < self.ma_1h[0]:
                    self.records["1h"]["bearish"].append(dt)


def consolidate_to_periods(datetimes, interval_minutes):
    """Convert list of datetimes to periods (start, end) for consecutive bars."""
    if not datetimes:
        return []

    datetimes = sorted(datetimes)
    periods = []
    period_start = datetimes[0]
    period_end = datetimes[0]

    # Allow some tolerance for gaps (up to 2x the interval to handle weekends/gaps)
    max_gap = timedelta(minutes=interval_minutes * 2)

    for dt in datetimes[1:]:
        expected_next = period_end + timedelta(minutes=interval_minutes)
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
    cerebro = bt.Cerebro()

    data = bt.feeds.GenericCSVData(
        dataname="PEPPERSTONE_XAUUSD, 5.csv",
        dtformat="%Y-%m-%dT%H:%M:%S%z",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=-1,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
        separator=",",
        headers=True,
    )

    cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=60)
    cerebro.addstrategy(MtfBiasStrategy)

    results = cerebro.run()
    strategy = results[0]

    # Consolidate and print results
    timeframes = {
        "15m": 15,
        "1h": 60,
    }

    for tf, interval in timeframes.items():
        print(f"\n{'='*60}")
        print(f"{tf.upper()} BULLISH PERIODS (close > SMA{strategy.p.ma_period})")
        print(f"{'='*60}")
        periods = consolidate_to_periods(strategy.records[tf]["bullish"], interval)
        print(f"Total: {len(periods)} periods\n")
        for start, end in periods:
            if start == end:
                print(f"  {start.isoformat()}")
            else:
                print(f"  {start.isoformat()} to {end.isoformat()}")

        print(f"\n{'='*60}")
        print(f"{tf.upper()} BEARISH PERIODS (close < SMA{strategy.p.ma_period})")
        print(f"{'='*60}")
        periods = consolidate_to_periods(strategy.records[tf]["bearish"], interval)
        print(f"Total: {len(periods)} periods\n")
        for start, end in periods:
            if start == end:
                print(f"  {start.isoformat()}")
            else:
                print(f"  {start.isoformat()} to {end.isoformat()}")


if __name__ == "__main__":
    main()
