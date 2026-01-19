"""
Backtest the Silver Bullet S1 strategy using Backtrader.

This script loads OHLC data from a CSV file, runs the SilverBulletStrategy,
and prints summary performance metrics.
"""

from __future__ import annotations

import os
import sys
import warnings

import backtrader as bt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))

from silver_bullet_strategy import SilverBulletStrategy


class PandasDataCustom(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", -1),
        ("openinterest", -1),
    )


def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    df = df[["open", "high", "low", "close"]].dropna()
    return df


def run_backtest(csv_file: str) -> None:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    warnings.filterwarnings("ignore", category=FutureWarning)

    data_df = load_data(csv_file)
    data = PandasDataCustom(dataname=data_df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SilverBulletStrategy, print_signals=False)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print("\n" + "=" * 80)
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print("=" * 80 + "\n")

    results = cerebro.run()
    strat = results[0]

    print("\n" + "=" * 80)
    print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print(f"Total Return: {((cerebro.broker.getvalue() / 10000.0) - 1) * 100:.2f}%")
    print("=" * 80)

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print("\nPerformance Metrics:")
    print("-" * 80)
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    total_trades = trades.get("total", {}).get("total", 0)
    won_trades = trades.get("won", {}).get("total", 0)
    lost_trades = trades.get("lost", {}).get("total", 0)
    print(f"Total Trades: {total_trades}")
    print(f"Won Trades: {won_trades}")
    print(f"Lost Trades: {lost_trades}")
    if total_trades > 0:
        win_rate = (won_trades / total_trades) * 100
        print(f"Win Rate: {win_rate:.2f}%")


def main() -> None:
    csv_file = "PEPPERSTONE_XAUUSD, 5.csv"
    run_backtest(csv_file)


if __name__ == "__main__":
    main()
