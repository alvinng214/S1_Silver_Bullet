"""
Backtest script for ICT Setup 01 [TradingFinder] FVG + Liquidity Sweeps/Hunt Alerts

This script:
1. Loads OHLC data from CSV with original timezone preserved
2. Runs the ICT Setup 01 logic to detect FVGs and trading signals
3. Displays all bullish/bearish FVGs, long/short signals, and alert events
4. Runs a Backtrader strategy to visualize the signals on price data
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib.machinery import SourceFileLoader

import backtrader as bt
import pandas as pd

# Load ICT Setup 01 module
SETUP01_PATH = os.path.join(
    os.path.dirname(__file__),
    "ICT Setup 01 [TradingFinder] FVG + Liquidity SweepsHunt Alerts, ICT Setup 01 TFlab.py",
)
setup01_module = SourceFileLoader("ict_setup_01", SETUP01_PATH).load_module()
calculate_setup_01 = setup01_module.calculate_setup_01
FVGState = setup01_module.FVGState
SignalState = setup01_module.SignalState
BarState = setup01_module.BarState
AlertEvent = setup01_module.AlertEvent


def load_data_with_timezone(csv_file: str) -> tuple[pd.DataFrame, str]:
    """Load CSV data preserving the original timezone from the file."""
    df = pd.read_csv(csv_file)

    # Parse the first timestamp to extract timezone info
    first_time = df["time"].iloc[0]
    # Example: 2025-12-25T01:20:00+08:00
    tz_info = first_time[-6:]  # Extract +08:00 or similar

    # Parse timestamps and keep original timezone
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close"]].dropna()

    return df, tz_info


class ICTSetup01PandasData(bt.feeds.PandasData):
    """Custom Pandas data feed with ICT Setup 01 signals."""
    lines = (
        "long_signal",
        "short_signal",
        "bull_fvg",
        "bear_fvg",
        "validity_bull",
        "validity_bear",
        "distal_bull",
        "proximal_bull",
        "distal_bear",
        "proximal_bear",
    )
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", -1),
        ("openinterest", -1),
        ("long_signal", "long_signal"),
        ("short_signal", "short_signal"),
        ("bull_fvg", "bull_fvg"),
        ("bear_fvg", "bear_fvg"),
        ("validity_bull", "validity_bull"),
        ("validity_bear", "validity_bear"),
        ("distal_bull", "distal_bull"),
        ("proximal_bull", "proximal_bull"),
        ("distal_bear", "distal_bear"),
        ("proximal_bear", "proximal_bear"),
    )


class ICTSetup01Strategy(bt.Strategy):
    """Strategy to display ICT Setup 01 signals in Backtrader."""

    params = (
        ("print_signals", True),
    )

    def __init__(self):
        self.long_signal = self.data.long_signal
        self.short_signal = self.data.short_signal
        self.bull_fvg = self.data.bull_fvg
        self.bear_fvg = self.data.bear_fvg
        self.signal_count = {"long": 0, "short": 0}
        self.fvg_count = {"bull": 0, "bear": 0}

    def next(self):
        dt = self.data.datetime.datetime(0)

        # Track FVG formations
        if self.bull_fvg[0]:
            self.fvg_count["bull"] += 1
            if self.p.print_signals:
                print(f"[{dt}] BULLISH FVG #{self.fvg_count['bull']} formed | "
                      f"Distal: {self.data.distal_bull[0]:.2f} | "
                      f"Proximal: {self.data.proximal_bull[0]:.2f}")

        if self.bear_fvg[0]:
            self.fvg_count["bear"] += 1
            if self.p.print_signals:
                print(f"[{dt}] BEARISH FVG #{self.fvg_count['bear']} formed | "
                      f"Distal: {self.data.distal_bear[0]:.2f} | "
                      f"Proximal: {self.data.proximal_bear[0]:.2f}")

        # Track trading signals
        if self.long_signal[0]:
            self.signal_count["long"] += 1
            if self.p.print_signals:
                print(f"[{dt}] *** LONG SIGNAL #{self.signal_count['long']} *** | "
                      f"Close: {self.data.close[0]:.2f} | "
                      f"FVG Zone: {self.data.distal_bull[0]:.2f} - {self.data.proximal_bull[0]:.2f}")

        if self.short_signal[0]:
            self.signal_count["short"] += 1
            if self.p.print_signals:
                print(f"[{dt}] *** SHORT SIGNAL #{self.signal_count['short']} *** | "
                      f"Close: {self.data.close[0]:.2f} | "
                      f"FVG Zone: {self.data.distal_bear[0]:.2f} - {self.data.proximal_bear[0]:.2f}")

    def stop(self):
        print("\n" + "=" * 80)
        print("ICT SETUP 01 BACKTEST SUMMARY")
        print("=" * 80)
        print(f"Total Bullish FVGs: {self.fvg_count['bull']}")
        print(f"Total Bearish FVGs: {self.fvg_count['bear']}")
        print(f"Total Long Signals: {self.signal_count['long']}")
        print(f"Total Short Signals: {self.signal_count['short']}")
        print("=" * 80)


def run_ict_setup_01_backtest(
    csv_file: str = "PEPPERSTONE_XAUUSD, 5.csv",
    matr: float = 1.0,
    fvg_validity: int = 15,
    discount_premium: bool = False,
    issue_signal_method: str = "Hunt",
    max_signals: int = 3,
    signal_after_hunts: bool = False,
    hunts_needed: int = 2,
    print_signals: bool = True,
    print_fvg_details: bool = True,
    print_alert_events: bool = True,
) -> dict:
    """
    Run ICT Setup 01 backtest and display all signals.

    Args:
        csv_file: Path to CSV file with OHLC data
        matr: FVG detector ATR multiplier
        fvg_validity: Validity period in bars
        discount_premium: Use discount/premium refinement
        issue_signal_method: "Hunt" or "Sweeps"
        max_signals: Max signals from a zone
        signal_after_hunts: Require hunts before signaling
        hunts_needed: Number of hunts needed to confirm signal
        print_signals: Print signal details during backtest
        print_fvg_details: Print FVG state details
        print_alert_events: Print alert events

    Returns:
        Dict with backtest results
    """
    # Load data with timezone
    csv_path = os.path.join(os.path.dirname(__file__), csv_file)
    if not os.path.exists(csv_path):
        csv_path = csv_file

    print(f"\nLoading data from: {csv_path}")
    df, tz_info = load_data_with_timezone(csv_path)
    print(f"Timezone from CSV: {tz_info}")
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total bars: {len(df)}")

    # Run ICT Setup 01 calculation
    print("\n" + "=" * 80)
    print("RUNNING ICT SETUP 01 [TradingFinder] FVG + Liquidity Sweeps/Hunt Alerts")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  - ATR Multiplier (matr): {matr}")
    print(f"  - FVG Validity: {fvg_validity} bars")
    print(f"  - Discount/Premium: {discount_premium}")
    print(f"  - Signal Method: {issue_signal_method}")
    print(f"  - Max Signals: {max_signals}")
    print(f"  - Signal After Hunts: {signal_after_hunts}")
    print(f"  - Hunts Needed: {hunts_needed}")
    print("=" * 80 + "\n")

    results = calculate_setup_01(
        df,
        matr=matr,
        fvg_validity=fvg_validity,
        discount_premium=discount_premium,
        issue_signal_method=issue_signal_method,
        max_signals=max_signals,
        signal_after_hunts=signal_after_hunts,
        hunts_needed=hunts_needed,
    )

    fvg_states = results["fvg_states"]
    signal_states = results["signals"]
    bar_states = results["bar_states"]
    alert_events = results["alerts"]

    # Print FVG details
    if print_fvg_details:
        print("\n" + "-" * 80)
        print("FVG STATES (Fair Value Gaps Detected)")
        print("-" * 80)
        bull_fvgs = [f for f in fvg_states if f.direction == "bull"]
        bear_fvgs = [f for f in fvg_states if f.direction == "bear"]

        print(f"\nBullish FVGs: {len(bull_fvgs)}")
        for i, fvg in enumerate(bull_fvgs, 1):
            bar_time = df.index[fvg.point] if fvg.point < len(df) else "N/A"
            print(f"  #{i} @ Bar {fvg.point} ({bar_time})")
            print(f"      Distal: {fvg.distal:.2f} | Proximal: {fvg.proximal:.2f}")
            print(f"      Discount: {fvg.discount:.2f} | Premium: {fvg.premium:.2f} | Equilibrium: {fvg.equilibrium:.2f}")

        print(f"\nBearish FVGs: {len(bear_fvgs)}")
        for i, fvg in enumerate(bear_fvgs, 1):
            bar_time = df.index[fvg.point] if fvg.point < len(df) else "N/A"
            print(f"  #{i} @ Bar {fvg.point} ({bar_time})")
            print(f"      Distal: {fvg.distal:.2f} | Proximal: {fvg.proximal:.2f}")
            print(f"      Discount: {fvg.discount:.2f} | Premium: {fvg.premium:.2f} | Equilibrium: {fvg.equilibrium:.2f}")

    # Print Alert Events
    if print_alert_events:
        print("\n" + "-" * 80)
        print("ALERT EVENTS (Trading Signals)")
        print("-" * 80)
        long_alerts = [a for a in alert_events if a.direction == "long"]
        short_alerts = [a for a in alert_events if a.direction == "short"]

        print(f"\nLong Signals: {len(long_alerts)}")
        for i, alert in enumerate(long_alerts, 1):
            bar_time = df.index[alert.index] if alert.index < len(df) else "N/A"
            close_price = df["close"].iloc[alert.index] if alert.index < len(df) else 0
            print(f"  #{i} @ Bar {alert.index} ({bar_time}) | Close: {close_price:.2f}")
            print(f"      {alert.message}")

        print(f"\nShort Signals: {len(short_alerts)}")
        for i, alert in enumerate(short_alerts, 1):
            bar_time = df.index[alert.index] if alert.index < len(df) else "N/A"
            close_price = df["close"].iloc[alert.index] if alert.index < len(df) else 0
            print(f"  #{i} @ Bar {alert.index} ({bar_time}) | Close: {close_price:.2f}")
            print(f"      {alert.message}")

    # Prepare data for Backtrader
    df_bt = df.copy()

    # Add signal columns from bar_states
    df_bt["long_signal"] = 0
    df_bt["short_signal"] = 0
    df_bt["bull_fvg"] = 0
    df_bt["bear_fvg"] = 0
    df_bt["validity_bull"] = 0
    df_bt["validity_bear"] = 0
    df_bt["distal_bull"] = 0.0
    df_bt["proximal_bull"] = 0.0
    df_bt["distal_bear"] = 0.0
    df_bt["proximal_bear"] = 0.0

    for state in bar_states:
        if state.index < len(df_bt):
            df_bt.iloc[state.index, df_bt.columns.get_loc("long_signal")] = int(state.long_signal)
            df_bt.iloc[state.index, df_bt.columns.get_loc("short_signal")] = int(state.short_signal)
            df_bt.iloc[state.index, df_bt.columns.get_loc("bull_fvg")] = int(state.bull_fvg)
            df_bt.iloc[state.index, df_bt.columns.get_loc("bear_fvg")] = int(state.bear_fvg)
            df_bt.iloc[state.index, df_bt.columns.get_loc("validity_bull")] = int(state.validity_bull)
            df_bt.iloc[state.index, df_bt.columns.get_loc("validity_bear")] = int(state.validity_bear)
            df_bt.iloc[state.index, df_bt.columns.get_loc("distal_bull")] = state.distal_bull
            df_bt.iloc[state.index, df_bt.columns.get_loc("proximal_bull")] = state.proximal_bull
            df_bt.iloc[state.index, df_bt.columns.get_loc("distal_bear")] = state.distal_bear
            df_bt.iloc[state.index, df_bt.columns.get_loc("proximal_bear")] = state.proximal_bear

    # Remove timezone for backtrader compatibility
    df_bt.index = df_bt.index.tz_localize(None) if df_bt.index.tz is not None else df_bt.index

    # Run Backtrader
    print("\n" + "=" * 80)
    print("BACKTRADER SIGNAL VISUALIZATION")
    print("=" * 80 + "\n")

    cerebro = bt.Cerebro()
    data_feed = ICTSetup01PandasData(dataname=df_bt)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(ICTSetup01Strategy, print_signals=print_signals)
    cerebro.broker.setcash(10000.0)

    cerebro.run()

    # Summary statistics
    total_long = sum(1 for s in signal_states if s.long_signal)
    total_short = sum(1 for s in signal_states if s.short_signal)
    total_bull_fvg = len([f for f in fvg_states if f.direction == "bull"])
    total_bear_fvg = len([f for f in fvg_states if f.direction == "bear"])

    return {
        "fvg_states": fvg_states,
        "signal_states": signal_states,
        "bar_states": bar_states,
        "alert_events": alert_events,
        "summary": {
            "total_bars": len(df),
            "total_long_signals": total_long,
            "total_short_signals": total_short,
            "total_bull_fvg": total_bull_fvg,
            "total_bear_fvg": total_bear_fvg,
            "timezone": tz_info,
            "data_start": str(df.index.min()),
            "data_end": str(df.index.max()),
        },
        "dataframe": df_bt,
    }


def main():
    """Run the ICT Setup 01 backtest with default parameters."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ICT Setup 01 [TradingFinder] FVG + Liquidity Sweeps/Hunt Alerts Backtest"
    )
    parser.add_argument(
        "--csv-file",
        default="PEPPERSTONE_XAUUSD, 5.csv",
        help="Path to CSV file with OHLC data",
    )
    parser.add_argument(
        "--matr",
        type=float,
        default=1.0,
        help="FVG detector ATR multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--fvg-validity",
        type=int,
        default=15,
        help="FVG validity period in bars (default: 15)",
    )
    parser.add_argument(
        "--discount-premium",
        action="store_true",
        help="Use discount/premium refinement",
    )
    parser.add_argument(
        "--signal-method",
        choices=["Hunt", "Sweeps"],
        default="Hunt",
        help="Signal method: Hunt or Sweeps (default: Hunt)",
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=3,
        help="Max signals from a zone (default: 3)",
    )
    parser.add_argument(
        "--signal-after-hunts",
        action="store_true",
        help="Require hunts before signaling",
    )
    parser.add_argument(
        "--hunts-needed",
        type=int,
        default=2,
        help="Number of hunts needed to confirm signal (default: 2)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed signal output during backtest",
    )

    args = parser.parse_args()

    results = run_ict_setup_01_backtest(
        csv_file=args.csv_file,
        matr=args.matr,
        fvg_validity=args.fvg_validity,
        discount_premium=args.discount_premium,
        issue_signal_method=args.signal_method,
        max_signals=args.max_signals,
        signal_after_hunts=args.signal_after_hunts,
        hunts_needed=args.hunts_needed,
        print_signals=not args.quiet,
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    summary = results["summary"]
    print(f"Data File Timezone: {summary['timezone']}")
    print(f"Data Range: {summary['data_start']} to {summary['data_end']}")
    print(f"Total Bars: {summary['total_bars']}")
    print(f"Bullish FVGs: {summary['total_bull_fvg']}")
    print(f"Bearish FVGs: {summary['total_bear_fvg']}")
    print(f"Long Signals: {summary['total_long_signals']}")
    print(f"Short Signals: {summary['total_short_signals']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
