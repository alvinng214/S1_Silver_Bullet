"""
HTF Market Structure + Smart Money Zones bias strategy.

Combines:
- Market structure trend (CHoCH/BoS) on 4H and 1D resampled feeds
- Smart Money Zones MTF trend panel (MA-based) on 4H and 1D
- HTF Points of Interest (POI) from MirPapa HTF FVG/OB Threeple

The strategy logs the consolidated HTF bias as bullish, bearish, or neutral.
"""

from __future__ import annotations

import os
import sys

import backtrader as bt
import math
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indicators"))
from market_structure_mtf import MarketStructureIndicator


class HTFBiasStrategy(bt.Strategy):
    params = (
        ("pivot_strength", 15),
        ("print_bias", True),
    )

    def __init__(self) -> None:
        if len(self.datas) < 3:
            raise ValueError("HTFBiasStrategy requires base + 4H + 1D data feeds.")

        self.data_4h = self.datas[1]
        self.data_1d = self.datas[2]

        self.ms_4h = MarketStructureIndicator(
            self.data_4h,
            pivot_strength=self.params.pivot_strength,
        )
        self.ms_1d = MarketStructureIndicator(
            self.data_1d,
            pivot_strength=self.params.pivot_strength,
        )

        self.last_bias = None
        self.last_poi_state = None
        self.last_liquidity_state = None
        self.last_killzone_state = None

    def log(self, message: str) -> None:
        if not self.params.print_bias:
            return
        dt = self.data.datetime.datetime(0)
        print(f"{dt.isoformat()}: {message}")

    @staticmethod
    def _resolve_bias(ms_bias: int, smz_bias: int) -> int:
        if ms_bias == 1 and smz_bias == 1:
            return 1
        if ms_bias == -1 and smz_bias == -1:
            return -1
        return 0

    @staticmethod
    def _nan_to_none(value: float) -> Optional[float]:
        if value != value or math.isnan(value):
            return None
        return float(value)

    def next(self) -> None:
        ms_4h = int(self.ms_4h.lines.trend[0])
        ms_1d = int(self.ms_1d.lines.trend[0])

        ms_bias = 0
        if ms_4h == 1 and ms_1d == 1:
            ms_bias = 1
        elif ms_4h == -1 and ms_1d == -1:
            ms_bias = -1

        smz_4h = int(self.data.smz_trend_4h[0])
        smz_1d = int(self.data.smz_trend_1d[0])
        smz_bias = 0
        if smz_4h == 1 and smz_1d == 1:
            smz_bias = 1
        elif smz_4h == -1 and smz_1d == -1:
            smz_bias = -1

        bias = self._resolve_bias(ms_bias, smz_bias)
        if bias != self.last_bias:
            label = "BULLISH" if bias == 1 else "BEARISH" if bias == -1 else "NEUTRAL"
            self.log(
                "HTF Bias -> {label} | MS(4H={ms4h}, 1D={ms1d}) | "
                "SMZ(4H={sm4h}, 1D={sm1d})".format(
                    label=label,
                    ms4h=ms_4h,
                    ms1d=ms_1d,
                    sm4h=smz_4h,
                    sm1d=smz_1d,
                )
            )
            self.last_bias = bias

        poi_state = (
            int(self.data.poi_high_bull[0]),
            int(self.data.poi_high_bear[0]),
            int(self.data.poi_mid_bull[0]),
            int(self.data.poi_mid_bear[0]),
        )
        if poi_state != self.last_poi_state:
            self.log(
                "HTF POI | HighTF Bull={high_bull} Bear={high_bear} | "
                "MidTF Bull={mid_bull} Bear={mid_bear}".format(
                    high_bull=poi_state[0],
                    high_bear=poi_state[1],
                    mid_bull=poi_state[2],
                    mid_bear=poi_state[3],
                )
            )
            self.last_poi_state = poi_state

        liquidity_state = (
            self._nan_to_none(float(self.data.smc_liquidity_high[0])),
            self._nan_to_none(float(self.data.smc_liquidity_low[0])),
            self._nan_to_none(float(self.data.liq_buyside_target[0])),
            self._nan_to_none(float(self.data.liq_sellside_target[0])),
        )
        if liquidity_state != self.last_liquidity_state:
            self.log(
                "External Liquidity | SMC High={smc_high} Low={smc_low} | "
                "Buyside={buyside} Sellside={sellside}".format(
                    smc_high=liquidity_state[0],
                    smc_low=liquidity_state[1],
                    buyside=liquidity_state[2],
                    sellside=liquidity_state[3],
                )
            )
            self.last_liquidity_state = liquidity_state

        killzone_state = (
            int(self.data.sb_sig_ln[0]),
            int(self.data.sb_sig_am[0]),
            int(self.data.sb_sig_pm[0]),
            int(self.data.sb_lux_ln[0]),
            int(self.data.sb_lux_am[0]),
            int(self.data.sb_lux_pm[0]),
            int(self.data.sb_or_range[0]),
            int(self.data.sb_trading_range[0]),
        )
        if killzone_state != self.last_killzone_state:
            self.log(
                "Killzone Window | Signals LN/AM/PM={sig_ln}/{sig_am}/{sig_pm} | "
                "Lux LN/AM/PM={lux_ln}/{lux_am}/{lux_pm} | OR={or_range} Trade={trade_range}".format(
                    sig_ln=killzone_state[0],
                    sig_am=killzone_state[1],
                    sig_pm=killzone_state[2],
                    lux_ln=killzone_state[3],
                    lux_am=killzone_state[4],
                    lux_pm=killzone_state[5],
                    or_range=killzone_state[6],
                    trade_range=killzone_state[7],
                )
            )
            self.last_killzone_state = killzone_state


class SilverBulletStrategy(bt.Strategy):
    params = (
        ("pivot_strength", 15),
        ("risk_per_trade", 0.02),
        ("print_trades", True),
        ("debug_signals", False),
    )

    def __init__(self) -> None:
        self.order = None
        self.signal_stats = {
            # Entry trigger counts
            "trigger_bull": 0,
            "trigger_bear": 0,
            "entry_sb_bull": 0,
            "entry_sb_bear": 0,
            "entry_setup01_bull": 0,
            "entry_setup01_bear": 0,
            "entry_ote_bull": 0,
            "entry_ote_bear": 0,
            # Filter rejection counts
            "filter_time_rejected_bull": 0,
            "filter_time_rejected_bear": 0,
            "filter_trend_rejected_bull": 0,
            "filter_trend_rejected_bear": 0,
            "filter_structure_rejected_bull": 0,
            "filter_structure_rejected_bear": 0,
            # Risk management rejections
            "stop_invalid_long": 0,
            "stop_invalid_short": 0,
            "target_invalid_long": 0,
            "target_invalid_short": 0,
            # Successful orders
            "orders_placed": 0,
        }
        # Track detailed trade information
        self.pending_trade = None
        self.completed_trades = []

    def log(self, message: str) -> None:
        if not self.params.print_trades:
            return
        dt = self.data.datetime.datetime(0)
        print(f"{dt.isoformat()}: {message}")

    @staticmethod
    def _valid_price(value: float | None) -> float | None:
        if value is None:
            return None
        if value != value or math.isnan(value):
            return None
        if float(value) <= 0:
            return None
        return float(value)

    def _resolve_stop_long(self) -> float | None:
        sellside_liquidity = self._valid_price(float(self.data.liq_sellside_target[0]))
        return sellside_liquidity

    def _resolve_stop_short(self) -> float | None:
        buyside_liquidity = self._valid_price(float(self.data.liq_buyside_target[0]))
        return buyside_liquidity

    def _resolve_target_long(self, entry_price: float, risk_per_unit: float) -> float | None:
        return entry_price + risk_per_unit * 2

    def _resolve_target_short(self, entry_price: float, risk_per_unit: float) -> float | None:
        return entry_price - risk_per_unit * 2

    def _get_entry_signal_type(self, is_long: bool) -> str:
        """Determine which specific entry signal triggered the trade."""
        signals = []
        if is_long:
            if int(self.data.entry_sb_bull[0]) == 1:
                signals.append("SB_FVG_Retrace")
            if int(self.data.entry_setup01_bull[0]) == 1:
                signals.append("ICT_Setup01")
            if int(self.data.entry_ote_bull[0]) == 1:
                signals.append("Fib_OTE")
        else:
            if int(self.data.entry_sb_bear[0]) == 1:
                signals.append("SB_FVG_Retrace")
            if int(self.data.entry_setup01_bear[0]) == 1:
                signals.append("ICT_Setup01")
            if int(self.data.entry_ote_bear[0]) == 1:
                signals.append("Fib_OTE")
        return " + ".join(signals) if signals else "Unknown"

    def notify_order(self, order: bt.Order) -> None:
        if order.status in {order.Completed, order.Canceled, order.Margin, order.Rejected}:
            self.order = None
            return

    def notify_trade(self, trade: bt.Trade) -> None:
        if not trade.isclosed:
            return
        pnl = trade.pnl
        pnl_pct = (pnl / (self.broker.getvalue() - pnl)) * 100
        is_win = pnl > 0

        if self.pending_trade:
            trade_record = {
                **self.pending_trade,
                "exit_time": self.data.datetime.datetime(0).isoformat(),
                "exit_price": trade.price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "result": "WIN" if is_win else "LOSS",
            }
            self.completed_trades.append(trade_record)
            self.log(
                "CLOSED {result} | Entry: {entry_time} @ {entry_price:.2f} | "
                "Exit: {exit_time} @ {exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)".format(
                    **trade_record
                )
            )
            self.pending_trade = None

    def _track_signal_counts(self) -> None:
        """Track entry trigger counts for debugging."""
        if not self.params.debug_signals:
            return
        if int(self.data.entry_trigger_bull[0]) == 1:
            self.signal_stats["trigger_bull"] += 1
        if int(self.data.entry_trigger_bear[0]) == 1:
            self.signal_stats["trigger_bear"] += 1
        if int(self.data.entry_sb_bull[0]) == 1:
            self.signal_stats["entry_sb_bull"] += 1
        if int(self.data.entry_sb_bear[0]) == 1:
            self.signal_stats["entry_sb_bear"] += 1
        if int(self.data.entry_setup01_bull[0]) == 1:
            self.signal_stats["entry_setup01_bull"] += 1
        if int(self.data.entry_setup01_bear[0]) == 1:
            self.signal_stats["entry_setup01_bear"] += 1
        if int(self.data.entry_ote_bull[0]) == 1:
            self.signal_stats["entry_ote_bull"] += 1
        if int(self.data.entry_ote_bear[0]) == 1:
            self.signal_stats["entry_ote_bear"] += 1

    def _check_filters(self, is_long: bool) -> tuple[bool, str]:
        """
        Check filters in sequence: Time -> Trend -> Structure.
        Returns (passed, rejection_reason).

        Filter sequence:
        1. Time Filter: ICT session must be active
        2. Trend Filter: 15M and 1H SMZ trends must agree with trade direction
        3. Structure Filter: MSS + FVG gate must be satisfied
        """
        # Filter 1: Time Filter - ICT Session must be active
        session_active = int(self.data.filter_session_active[0]) == 1
        if not session_active:
            return False, "Time Filter (ICT session not active)"

        # Filter 2: Trend Filter - HTF bias must align with trade direction
        if is_long:
            htf_aligned = int(self.data.filter_htf_bias_bull[0]) == 1
        else:
            htf_aligned = int(self.data.filter_htf_bias_bear[0]) == 1
        if not htf_aligned:
            return False, "Trend Filter (15M/1H bias not aligned)"

        # Filter 3: Structure Filter - MSS + FVG gate
        if is_long:
            structure_ok = int(self.data.filter_structure_bull[0]) == 1
        else:
            structure_ok = int(self.data.filter_structure_bear[0]) == 1
        if not structure_ok:
            return False, "Structure Filter (MSS+FVG not confirmed)"

        return True, ""

    def next(self) -> None:
        self._track_signal_counts()
        if self.order:
            return
        if self.position:
            return

        # STEP 1: Check if any entry trigger condition is met FIRST
        long_trigger = int(self.data.entry_trigger_bull[0]) == 1
        short_trigger = int(self.data.entry_trigger_bear[0]) == 1

        if not long_trigger and not short_trigger:
            return

        # STEP 2: Check filters only when triggers are met
        # Process long trigger
        if long_trigger:
            filters_passed, rejection_reason = self._check_filters(is_long=True)
            if not filters_passed:
                if self.params.debug_signals:
                    self.log(f"LONG trigger rejected: {rejection_reason}")
                if "Time" in rejection_reason:
                    self.signal_stats["filter_time_rejected_bull"] += 1
                elif "Trend" in rejection_reason:
                    self.signal_stats["filter_trend_rejected_bull"] += 1
                elif "Structure" in rejection_reason:
                    self.signal_stats["filter_structure_rejected_bull"] += 1
                long_trigger = False

        # Process short trigger
        if short_trigger:
            filters_passed, rejection_reason = self._check_filters(is_long=False)
            if not filters_passed:
                if self.params.debug_signals:
                    self.log(f"SHORT trigger rejected: {rejection_reason}")
                if "Time" in rejection_reason:
                    self.signal_stats["filter_time_rejected_bear"] += 1
                elif "Trend" in rejection_reason:
                    self.signal_stats["filter_trend_rejected_bear"] += 1
                elif "Structure" in rejection_reason:
                    self.signal_stats["filter_structure_rejected_bear"] += 1
                short_trigger = False

        # After filtering, check if any signal remains
        long_signal = long_trigger
        short_signal = short_trigger

        if not long_signal and not short_signal:
            return

        entry_price = float(self.data.close[0])
        risk_cash = self.broker.getvalue() * self.params.risk_per_trade
        max_cash = self.broker.getcash()

        if long_signal:
            stop_price = self._resolve_stop_long()
            if stop_price is None or stop_price >= entry_price:
                self.signal_stats["stop_invalid_long"] += 1
                return
            risk_per_unit = entry_price - stop_price
            size = risk_cash / risk_per_unit
            size = min(size, max_cash / entry_price)
            if size <= 0:
                return
            target_price = self._resolve_target_long(entry_price, risk_per_unit)
            if target_price is None or target_price <= entry_price:
                self.signal_stats["target_invalid_long"] += 1
                return
            signal_type = self._get_entry_signal_type(is_long=True)
            self.pending_trade = {
                "trade_num": len(self.completed_trades) + 1,
                "direction": "LONG",
                "entry_time": self.data.datetime.datetime(0).isoformat(),
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "signal_type": signal_type,
                "size": size,
            }
            self.order = self.buy_bracket(
                size=size,
                stopprice=stop_price,
                limitprice=target_price,
            )
            self.signal_stats["orders_placed"] += 1
            self.log(
                "LONG entry={entry:.2f} stop={stop:.2f} target={target:.2f} size={size:.4f} signal={signal}".format(
                    entry=entry_price,
                    stop=stop_price,
                    target=target_price,
                    size=size,
                    signal=signal_type,
                )
            )

        if short_signal:
            stop_price = self._resolve_stop_short()
            if stop_price is None or stop_price <= entry_price:
                self.signal_stats["stop_invalid_short"] += 1
                return
            risk_per_unit = stop_price - entry_price
            size = risk_cash / risk_per_unit
            size = min(size, max_cash / entry_price)
            if size <= 0:
                return
            target_price = self._resolve_target_short(entry_price, risk_per_unit)
            if target_price is None or target_price >= entry_price:
                self.signal_stats["target_invalid_short"] += 1
                return
            signal_type = self._get_entry_signal_type(is_long=False)
            self.pending_trade = {
                "trade_num": len(self.completed_trades) + 1,
                "direction": "SHORT",
                "entry_time": self.data.datetime.datetime(0).isoformat(),
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "signal_type": signal_type,
                "size": size,
            }
            self.order = self.sell_bracket(
                size=size,
                stopprice=stop_price,
                limitprice=target_price,
            )
            self.signal_stats["orders_placed"] += 1
            self.log(
                "SHORT entry={entry:.2f} stop={stop:.2f} target={target:.2f} size={size:.4f} signal={signal}".format(
                    entry=entry_price,
                    stop=stop_price,
                    target=target_price,
                    size=size,
                    signal=signal_type,
                )
            )

    def stop(self) -> None:
        # Always print the detailed trade report
        self._print_detailed_report()

        if not self.params.debug_signals:
            return

        # Print filter statistics
        print("\n" + "=" * 100)
        print("{:^100}".format("FILTER STATISTICS"))
        print("=" * 100)

        print("\n--- Entry Triggers Detected ---")
        print(f"  LONG triggers:  {self.signal_stats['trigger_bull']}")
        print(f"  SHORT triggers: {self.signal_stats['trigger_bear']}")

        print("\n--- Trigger Breakdown ---")
        print(f"  SB FVG Retrace:  LONG={self.signal_stats['entry_sb_bull']} SHORT={self.signal_stats['entry_sb_bear']}")
        print(f"  ICT Setup 01:    LONG={self.signal_stats['entry_setup01_bull']} SHORT={self.signal_stats['entry_setup01_bear']}")
        print(f"  Fibonacci OTE:   LONG={self.signal_stats['entry_ote_bull']} SHORT={self.signal_stats['entry_ote_bear']}")

        print("\n--- Filter Rejections ---")
        print(f"  Time Filter (ICT Session):")
        print(f"    LONG rejected:  {self.signal_stats['filter_time_rejected_bull']}")
        print(f"    SHORT rejected: {self.signal_stats['filter_time_rejected_bear']}")
        print(f"  Trend Filter (15M/1H Bias):")
        print(f"    LONG rejected:  {self.signal_stats['filter_trend_rejected_bull']}")
        print(f"    SHORT rejected: {self.signal_stats['filter_trend_rejected_bear']}")
        print(f"  Structure Filter (MSS+FVG):")
        print(f"    LONG rejected:  {self.signal_stats['filter_structure_rejected_bull']}")
        print(f"    SHORT rejected: {self.signal_stats['filter_structure_rejected_bear']}")

        print("\n--- Risk Management Rejections ---")
        print(f"  Invalid Stop (LONG):   {self.signal_stats['stop_invalid_long']}")
        print(f"  Invalid Stop (SHORT):  {self.signal_stats['stop_invalid_short']}")
        print(f"  Invalid Target (LONG): {self.signal_stats['target_invalid_long']}")
        print(f"  Invalid Target (SHORT):{self.signal_stats['target_invalid_short']}")

        print("\n--- Orders Placed ---")
        print(f"  Total orders: {self.signal_stats['orders_placed']}")
        print("=" * 100)

    def _print_detailed_report(self) -> None:
        """Print a comprehensive trade-by-trade report."""
        if not self.completed_trades:
            print("\n" + "=" * 100)
            print("NO TRADES COMPLETED")
            print("=" * 100)
            return

        print("\n" + "=" * 100)
        print("DETAILED TRADE REPORT - Silver Bullet XAUUSD Backtest")
        print("=" * 100)

        # Trade-by-trade details
        print("\n{:^100}".format("TRADE-BY-TRADE DETAILS"))
        print("-" * 100)
        print(
            "{:<4} {:<6} {:<20} {:<10} {:<10} {:<10} {:<12} {:<6} {:<25}".format(
                "#", "Dir", "Entry Time", "Entry", "Stop", "Target", "P&L", "Result", "Signal"
            )
        )
        print("-" * 100)

        wins = 0
        losses = 0
        total_pnl = 0
        signal_stats = {}

        for trade in self.completed_trades:
            print(
                "{:<4} {:<6} {:<20} {:<10.2f} {:<10.2f} {:<10.2f} ${:<11.2f} {:<6} {:<25}".format(
                    trade["trade_num"],
                    trade["direction"],
                    trade["entry_time"][:19],
                    trade["entry_price"],
                    trade["stop_price"],
                    trade["target_price"],
                    trade["pnl"],
                    trade["result"],
                    trade["signal_type"][:25],
                )
            )

            if trade["result"] == "WIN":
                wins += 1
            else:
                losses += 1
            total_pnl += trade["pnl"]

            # Track stats by signal type
            sig = trade["signal_type"]
            if sig not in signal_stats:
                signal_stats[sig] = {"wins": 0, "losses": 0, "pnl": 0}
            if trade["result"] == "WIN":
                signal_stats[sig]["wins"] += 1
            else:
                signal_stats[sig]["losses"] += 1
            signal_stats[sig]["pnl"] += trade["pnl"]

        # Summary statistics
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

        print("-" * 100)
        print("\n{:^100}".format("SUMMARY STATISTICS"))
        print("-" * 100)
        print(f"Total Trades:     {total}")
        print(f"Wins:             {wins}")
        print(f"Losses:           {losses}")
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Total P&L:        ${total_pnl:.2f}")
        print(f"Average P&L:      ${total_pnl / total:.2f}" if total > 0 else "N/A")

        # Signal type breakdown
        print("\n{:^100}".format("PERFORMANCE BY SIGNAL TYPE"))
        print("-" * 100)
        print(
            "{:<30} {:<10} {:<10} {:<12} {:<15}".format(
                "Signal Type", "Wins", "Losses", "Win Rate", "Total P&L"
            )
        )
        print("-" * 100)
        for sig, stats in sorted(signal_stats.items()):
            sig_total = stats["wins"] + stats["losses"]
            sig_win_rate = (stats["wins"] / sig_total * 100) if sig_total > 0 else 0
            print(
                "{:<30} {:<10} {:<10} {:<12.2f}% ${:<14.2f}".format(
                    sig[:30], stats["wins"], stats["losses"], sig_win_rate, stats["pnl"]
                )
            )

        print("=" * 100)
