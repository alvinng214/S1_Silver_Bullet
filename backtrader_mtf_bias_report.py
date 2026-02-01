import backtrader as bt


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

    for tf in ("15m", "1h"):
        print(f"\n{tf.upper()} Bullish bias (close > SMA{strategy.p.ma_period}):")
        for dt in strategy.records[tf]["bullish"]:
            print(dt.isoformat())
        print(f"\n{tf.upper()} Bearish bias (close < SMA{strategy.p.ma_period}):")
        for dt in strategy.records[tf]["bearish"]:
            print(dt.isoformat())


if __name__ == "__main__":
    main()
