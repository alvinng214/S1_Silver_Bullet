"""
Backtest Market Structure Strategy using Backtrader

This script loads OHLC data from CSV and runs the Market Structure trading strategy.
"""

import backtrader as bt
import pandas as pd
import sys
import os
from datetime import datetime

# Add strategies and indicators to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'indicators'))

from market_structure_strategy import MarketStructureStrategy, SimpleMarketStructureStrategy


class PandasDataCustom(bt.feeds.PandasData):
    """
    Custom Pandas Data Feed for Backtrader
    Handles the datetime parsing from the CSV file
    """
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', -1),  # No volume in this CSV
        ('openinterest', -1),
    )


def load_data(csv_file):
    """Load and prepare CSV data for backtrader"""
    print(f"Loading data from {csv_file}...")

    # Read CSV
    df = pd.read_csv(csv_file)

    # Parse datetime
    df['time'] = pd.to_datetime(df['time'])

    # Set datetime as index
    df.set_index('time', inplace=True)

    # Sort by datetime
    df.sort_index(inplace=True)

    # Keep only OHLC columns
    df = df[['open', 'high', 'low', 'close']]

    # Remove any NaN values
    df.dropna(inplace=True)

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def run_backtest(data_df, strategy_class=SimpleMarketStructureStrategy, params=None):
    """
    Run backtest with the given strategy and parameters

    Args:
        data_df: Pandas DataFrame with OHLC data
        strategy_class: Strategy class to use
        params: Dictionary of parameters for the strategy
    """
    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    if params:
        cerebro.addstrategy(strategy_class, **params)
    else:
        cerebro.addstrategy(strategy_class)

    # Add data feed
    data = PandasDataCustom(dataname=data_df)
    cerebro.adddata(data)

    # Set broker parameters
    cerebro.broker.setcash(10000.0)  # Starting capital
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Print starting conditions
    print('\n' + '='*80)
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    print('='*80 + '\n')

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Print ending conditions
    print('\n' + '='*80)
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    print(f'Total Return: {((cerebro.broker.getvalue() / 10000.0) - 1) * 100:.2f}%')
    print('='*80)

    # Print analyzer results
    print('\nPerformance Metrics:')
    print('-'*80)

    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")

    # Trade Analysis
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    lost_trades = trade_analysis.get('lost', {}).get('total', 0)

    print(f"Total Trades: {total_trades}")
    print(f"Won Trades: {won_trades}")
    print(f"Lost Trades: {lost_trades}")

    if total_trades > 0:
        win_rate = (won_trades / total_trades) * 100
        print(f"Win Rate: {win_rate:.2f}%")

        if won_trades > 0:
            avg_win = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0)
            print(f"Average Win: ${avg_win:.2f}")

        if lost_trades > 0:
            avg_loss = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
            print(f"Average Loss: ${avg_loss:.2f}")

    print('='*80 + '\n')

    # Plot the results (optional - requires matplotlib)
    try:
        # cerebro.plot(style='candlestick')
        pass  # Uncomment above line to see plots
    except:
        print("Plotting not available")

    return cerebro, strat


def main():
    """Main execution function"""
    # CSV file path
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'

    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return

    # Load data
    data_df = load_data(csv_file)

    # Run backtest with simple strategy
    print("\n" + "#"*80)
    print("# Running Simple Market Structure Strategy")
    print("#"*80 + "\n")

    cerebro, strat = run_backtest(
        data_df,
        strategy_class=SimpleMarketStructureStrategy,
        params={
            'pivot_strength': 15,
            'print_signals': False  # Set to True to see all signals
        }
    )

    # Optionally run advanced strategy
    # print("\n" + "#"*80)
    # print("# Running Advanced Market Structure Strategy")
    # print("#"*80 + "\n")
    #
    # cerebro2, strat2 = run_backtest(
    #     data_df,
    #     strategy_class=MarketStructureStrategy,
    #     params={
    #         'pivot_strength': 15,
    #         'use_bos': False,
    #         'risk_percent': 1.0,
    #         'stop_loss_atr_mult': 2.0,
    #         'take_profit_rr': 3.0,
    #         'print_signals': False
    #     }
    # )


if __name__ == '__main__':
    main()
