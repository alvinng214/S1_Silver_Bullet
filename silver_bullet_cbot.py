"""
Silver Bullet strategy logic ported from ``Silver_bullet_backtrader.py`` /
``strategies/htf_bias_strategy.py`` into a cBot-style Python event model.

IMPORTANT
---------
- Native cTrader cBots run in C# (cAlgo API). This module focuses on faithfully
  mirroring the *trading logic/state machine* in Python.
- To run this against live cTrader data, wire ``on_bar_closed`` to your bridge
  (for example Open API/websocket + external signal pipeline) and feed all
  required signal fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isnan
from typing import Optional


@dataclass
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class SignalSnapshot:
    # Entry triggers (already computed by upstream indicator stack)
    entry_trigger_bull: int = 0
    entry_trigger_bear: int = 0

    # Trigger attribution fields
    entry_sb_bull: int = 0
    entry_sb_bear: int = 0
    entry_setup01_bull: int = 0
    entry_setup01_bear: int = 0
    entry_ote_bull: int = 0
    entry_ote_bear: int = 0
    entry_ifvg_bull: int = 0
    entry_ifvg_bear: int = 0
    entry_obdet_bull: int = 0
    entry_obdet_bear: int = 0

    # Filters
    filter_session_active: int = 0
    filter_htf_bias_bull: int = 0
    filter_htf_bias_bear: int = 0
    filter_htf_poi_bull: int = 0
    filter_htf_poi_bear: int = 0

    # External liquidity targets used for SL anchor
    liq_buyside_target: Optional[float] = None
    liq_sellside_target: Optional[float] = None


class SilverBulletCbotPython:
    """Python cBot-style mirror of ``SilverBulletStrategy`` trade execution logic."""

    def __init__(
        self,
        *,
        risk_per_trade: float = 0.02,
        leverage: float = 100.0,
        max_concurrent_trades: int = 3,
        debug_signals: bool = False,
        print_trades: bool = True,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.max_concurrent_trades = max_concurrent_trades
        self.debug_signals = debug_signals
        self.print_trades = print_trades

        self.signal_stats: dict[str, int] = {
            "trigger_bull": 0,
            "trigger_bear": 0,
            "entry_sb_bull": 0,
            "entry_sb_bear": 0,
            "entry_setup01_bull": 0,
            "entry_setup01_bear": 0,
            "entry_ote_bull": 0,
            "entry_ote_bear": 0,
            "entry_ifvg_bull": 0,
            "entry_ifvg_bear": 0,
            "entry_obdet_bull": 0,
            "entry_obdet_bear": 0,
            "filter_time_rejected_bull": 0,
            "filter_time_rejected_bear": 0,
            "filter_trend_rejected_bull": 0,
            "filter_trend_rejected_bear": 0,
            "filter_structure_rejected_bull": 0,
            "filter_structure_rejected_bear": 0,
            "stop_invalid_long": 0,
            "stop_invalid_short": 0,
            "target_invalid_long": 0,
            "target_invalid_short": 0,
            "orders_placed": 0,
            "max_trades_rejected_bull": 0,
            "max_trades_rejected_bear": 0,
        }

        self.active_trades: list[dict] = []
        self.completed_trades: list[dict] = []
        self.rejected_triggers: list[dict] = []

        self.trade_counter = 0
        self.equity = 0.0
        self.cash = 0.0

    def on_start(self, *, initial_equity: float, initial_cash: Optional[float] = None) -> None:
        self.equity = float(initial_equity)
        self.cash = float(initial_cash if initial_cash is not None else initial_equity)

    def log(self, when: datetime, message: str) -> None:
        if self.print_trades:
            print(f"{when.isoformat()}: {message}")

    @staticmethod
    def _valid_price(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        v = float(value)
        if isnan(v) or v <= 0:
            return None
        return v

    def _resolve_stop_long(self, signals: SignalSnapshot) -> Optional[float]:
        return self._valid_price(signals.liq_sellside_target)

    def _resolve_stop_short(self, signals: SignalSnapshot) -> Optional[float]:
        return self._valid_price(signals.liq_buyside_target)

    @staticmethod
    def _resolve_target_long(entry_price: float, risk_per_unit: float) -> float:
        return entry_price + risk_per_unit * 2.0

    @staticmethod
    def _resolve_target_short(entry_price: float, risk_per_unit: float) -> float:
        return entry_price - risk_per_unit * 2.0

    def _get_entry_signal_type(self, signals: SignalSnapshot, *, is_long: bool) -> str:
        s: list[str] = []
        if is_long:
            if signals.entry_sb_bull == 1:
                s.append("SB_FVG_Retrace")
            if signals.entry_setup01_bull == 1:
                s.append("ICT_Setup01")
            if signals.entry_ote_bull == 1:
                s.append("Fib_OTE")
            if signals.entry_ifvg_bull == 1:
                s.append("IFVG_Realtime")
            if signals.entry_obdet_bull == 1:
                s.append("OB_Detector")
        else:
            if signals.entry_sb_bear == 1:
                s.append("SB_FVG_Retrace")
            if signals.entry_setup01_bear == 1:
                s.append("ICT_Setup01")
            if signals.entry_ote_bear == 1:
                s.append("Fib_OTE")
            if signals.entry_ifvg_bear == 1:
                s.append("IFVG_Realtime")
            if signals.entry_obdet_bear == 1:
                s.append("OB_Detector")
        return " + ".join(s) if s else "Unknown"

    def _track_signal_counts(self, signals: SignalSnapshot) -> None:
        if not self.debug_signals:
            return
        for k in (
            "entry_trigger_bull",
            "entry_trigger_bear",
            "entry_sb_bull",
            "entry_sb_bear",
            "entry_setup01_bull",
            "entry_setup01_bear",
            "entry_ote_bull",
            "entry_ote_bear",
            "entry_ifvg_bull",
            "entry_ifvg_bear",
            "entry_obdet_bull",
            "entry_obdet_bear",
        ):
            if getattr(signals, k) == 1:
                self.signal_stats[k.replace("entry_", "", 1) if k.startswith("entry_trigger") else k] += 1

    def _check_filters(self, signals: SignalSnapshot, *, is_long: bool) -> tuple[bool, str]:
        htf_poi_ok = signals.filter_htf_poi_bull == 1 if is_long else signals.filter_htf_poi_bear == 1
        if not htf_poi_ok:
            return False, "HTF POI Filter (no prior-10-bar touch of any 1H/4H OB)"

        htf_aligned = signals.filter_htf_bias_bull == 1 if is_long else signals.filter_htf_bias_bear == 1
        if not htf_aligned:
            return False, "Trend Filter (15M/1H bias not aligned)"

        if signals.filter_session_active != 1:
            return False, "Time Filter (ICT session not active)"

        return True, ""

    def _filter_states(self, signals: SignalSnapshot, *, is_long: bool) -> dict[str, bool]:
        return {
            "time": signals.filter_session_active == 1,
            "trend": signals.filter_htf_bias_bull == 1 if is_long else signals.filter_htf_bias_bear == 1,
            "htf_poi": signals.filter_htf_poi_bull == 1 if is_long else signals.filter_htf_poi_bear == 1,
        }

    def _record_rejected_trigger(self, when: datetime, signals: SignalSnapshot, *, is_long: bool, reason: str) -> None:
        states = self._filter_states(signals, is_long=is_long)
        self.rejected_triggers.append(
            {
                "time": when.isoformat(),
                "direction": "LONG" if is_long else "SHORT",
                "signal_type": self._get_entry_signal_type(signals, is_long=is_long),
                "rejection_reason": reason,
                "filter_htf_poi": states["htf_poi"],
                "filter_trend": states["trend"],
                "filter_time": states["time"],
            }
        )

    def _log_filter_sequence(self, when: datetime, signals: SignalSnapshot, *, is_long: bool) -> None:
        direction = "LONG" if is_long else "SHORT"
        states = self._filter_states(signals, is_long=is_long)
        sequence = "HTF POI -> Trend -> Time"
        self.log(
            when,
            "{direction} filter sequence: {sequence} | "
            "Time={time} Trend={trend} HTF_POI={htf_poi}".format(
                direction=direction,
                sequence=sequence,
                time="PASS" if states["time"] else "FAIL",
                trend="PASS" if states["trend"] else "FAIL",
                htf_poi="PASS" if states["htf_poi"] else "FAIL",
            ),
        )

    def _check_stops_and_targets(self, bar: Bar) -> None:
        if not self.active_trades:
            return

        trades_to_close: list[int] = []
        for i, trade in enumerate(self.active_trades):
            exit_price = None
            result = None
            if trade["direction"] == "LONG":
                if bar.low <= trade["stop_price"]:
                    exit_price = trade["stop_price"]
                    result = "LOSS"
                elif bar.high >= trade["target_price"]:
                    exit_price = trade["target_price"]
                    result = "WIN"
            else:
                if bar.high >= trade["stop_price"]:
                    exit_price = trade["stop_price"]
                    result = "LOSS"
                elif bar.low <= trade["target_price"]:
                    exit_price = trade["target_price"]
                    result = "WIN"

            if exit_price is None:
                continue

            pnl = (
                (exit_price - trade["entry_price"]) * trade["size"]
                if trade["direction"] == "LONG"
                else (trade["entry_price"] - exit_price) * trade["size"]
            )
            prev_equity = self.equity
            self.equity += pnl
            self.cash += pnl
            pnl_pct = (pnl / prev_equity) * 100 if prev_equity != 0 else 0.0

            record = {
                **trade,
                "exit_time": bar.time.isoformat(),
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "result": result,
            }
            self.completed_trades.append(record)
            trades_to_close.append(i)
            self.log(
                bar.time,
                "CLOSED {result} | Entry: {entry_time} @ {entry_price:.2f} | Exit: {exit_time} @ {exit_price:.2f} | "
                "P&L: ${pnl:.2f} ({pnl_pct:.2f}%)".format(**record),
            )

        for i in reversed(trades_to_close):
            del self.active_trades[i]

    def on_bar_closed(self, bar: Bar, signals: SignalSnapshot) -> None:
        """Mirror of Backtrader ``next()`` for one closed bar."""
        self._track_signal_counts(signals)
        self._check_stops_and_targets(bar)

        long_trigger = signals.entry_trigger_bull == 1
        short_trigger = signals.entry_trigger_bear == 1
        if not long_trigger and not short_trigger:
            return

        if long_trigger:
            self._log_filter_sequence(bar.time, signals, is_long=True)
            ok, reason = self._check_filters(signals, is_long=True)
            if not ok:
                self._record_rejected_trigger(bar.time, signals, is_long=True, reason=reason)
                if self.debug_signals:
                    self.log(bar.time, f"LONG trigger rejected: {reason}")
                if "Time" in reason:
                    self.signal_stats["filter_time_rejected_bull"] += 1
                elif "Trend" in reason:
                    self.signal_stats["filter_trend_rejected_bull"] += 1
                elif "Structure" in reason:
                    self.signal_stats["filter_structure_rejected_bull"] += 1
                long_trigger = False

        if short_trigger:
            self._log_filter_sequence(bar.time, signals, is_long=False)
            ok, reason = self._check_filters(signals, is_long=False)
            if not ok:
                self._record_rejected_trigger(bar.time, signals, is_long=False, reason=reason)
                if self.debug_signals:
                    self.log(bar.time, f"SHORT trigger rejected: {reason}")
                if "Time" in reason:
                    self.signal_stats["filter_time_rejected_bear"] += 1
                elif "Trend" in reason:
                    self.signal_stats["filter_trend_rejected_bear"] += 1
                elif "Structure" in reason:
                    self.signal_stats["filter_structure_rejected_bear"] += 1
                short_trigger = False

        if not long_trigger and not short_trigger:
            return

        if len(self.active_trades) >= self.max_concurrent_trades:
            if long_trigger:
                self.signal_stats["max_trades_rejected_bull"] += 1
                self._record_rejected_trigger(bar.time, signals, is_long=True, reason="Max concurrent trades reached")
            if short_trigger:
                self.signal_stats["max_trades_rejected_bear"] += 1
                self._record_rejected_trigger(bar.time, signals, is_long=False, reason="Max concurrent trades reached")
            return

        entry_price = float(bar.close)
        risk_cash = self.equity * self.risk_per_trade
        buying_power = self.cash * self.leverage

        if long_trigger:
            stop = self._resolve_stop_long(signals)
            if stop is None or stop >= entry_price:
                self.signal_stats["stop_invalid_long"] += 1
                return
            rpu = entry_price - stop
            size = min(risk_cash / rpu, buying_power / entry_price)
            if size <= 0:
                return
            target = self._resolve_target_long(entry_price, rpu)
            if target <= entry_price:
                self.signal_stats["target_invalid_long"] += 1
                return

            self.trade_counter += 1
            sig = self._get_entry_signal_type(signals, is_long=True)
            self.active_trades.append(
                {
                    "trade_num": self.trade_counter,
                    "direction": "LONG",
                    "entry_time": bar.time.isoformat(),
                    "entry_price": entry_price,
                    "stop_price": stop,
                    "target_price": target,
                    "signal_type": sig,
                    "size": size,
                }
            )
            self.signal_stats["orders_placed"] += 1
            self.log(bar.time, f"LONG entry={entry_price:.2f} stop={stop:.2f} target={target:.2f} size={size:.4f} signal={sig}")

        if short_trigger:
            stop = self._resolve_stop_short(signals)
            if stop is None or stop <= entry_price:
                self.signal_stats["stop_invalid_short"] += 1
                return
            rpu = stop - entry_price
            size = min(risk_cash / rpu, buying_power / entry_price)
            if size <= 0:
                return
            target = self._resolve_target_short(entry_price, rpu)
            if target >= entry_price:
                self.signal_stats["target_invalid_short"] += 1
                return

            self.trade_counter += 1
            sig = self._get_entry_signal_type(signals, is_long=False)
            self.active_trades.append(
                {
                    "trade_num": self.trade_counter,
                    "direction": "SHORT",
                    "entry_time": bar.time.isoformat(),
                    "entry_price": entry_price,
                    "stop_price": stop,
                    "target_price": target,
                    "signal_type": sig,
                    "size": size,
                }
            )
            self.signal_stats["orders_placed"] += 1
            self.log(bar.time, f"SHORT entry={entry_price:.2f} stop={stop:.2f} target={target:.2f} size={size:.4f} signal={sig}")

    def summary(self) -> dict:
        wins = sum(1 for t in self.completed_trades if t["result"] == "WIN")
        losses = len(self.completed_trades) - wins
        total_pnl = sum(t["pnl"] for t in self.completed_trades)
        return {
            "completed": len(self.completed_trades),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": (wins / len(self.completed_trades) * 100) if self.completed_trades else 0.0,
            "total_pnl": total_pnl,
            "equity": self.equity,
            "open_trades": len(self.active_trades),
            "orders_placed": self.signal_stats["orders_placed"],
            "rejected_triggers": len(self.rejected_triggers),
        }

    def print_filter_statistics(self) -> None:
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
        print(f"  IFVG Realtime:   LONG={self.signal_stats['entry_ifvg_bull']} SHORT={self.signal_stats['entry_ifvg_bear']}")
        print(f"  OB Detector:     LONG={self.signal_stats['entry_obdet_bull']} SHORT={self.signal_stats['entry_obdet_bear']}")

        print("\n--- Filter Rejections ---")
        print("  Time Filter (ICT Session):")
        print(f"    LONG rejected:  {self.signal_stats['filter_time_rejected_bull']}")
        print(f"    SHORT rejected: {self.signal_stats['filter_time_rejected_bear']}")
        print("  Trend Filter (15M/1H Bias):")
        print(f"    LONG rejected:  {self.signal_stats['filter_trend_rejected_bull']}")
        print(f"    SHORT rejected: {self.signal_stats['filter_trend_rejected_bear']}")
        print("  Structure Filter (MSS+FVG):")
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
        if not self.completed_trades:
            print("\n" + "=" * 100)
            print("NO TRADES COMPLETED")
            print("=" * 100)
            return

        print("\n" + "=" * 100)
        print("DETAILED TRADE REPORT - Silver Bullet XAUUSD Backtest")
        print("=" * 100)

        print("\n{:^100}".format("TRADE-BY-TRADE DETAILS"))
        print("-" * 100)
        print(
            "{:<4} {:<6} {:<20} {:<10} {:<10} {:<10} {:<12} {:<6} {:<40}".format(
                "#", "Dir", "Entry Time", "Entry", "Stop", "Target", "P&L", "Result", "Signal"
            )
        )
        print("-" * 100)

        wins = 0
        losses = 0
        total_pnl = 0.0
        signal_stats: dict[str, dict[str, int | float]] = {}

        for trade in self.completed_trades:
            print(
                "{:<4} {:<6} {:<20} {:<10.2f} {:<10.2f} {:<10.2f} ${:<11.2f} {:<6} {:<40}".format(
                    trade["trade_num"],
                    trade["direction"],
                    trade["entry_time"][:19],
                    trade["entry_price"],
                    trade["stop_price"],
                    trade["target_price"],
                    trade["pnl"],
                    trade["result"],
                    trade["signal_type"],
                )
            )

            if trade["result"] == "WIN":
                wins += 1
            else:
                losses += 1
            total_pnl += trade["pnl"]

            sig = trade["signal_type"]
            if sig not in signal_stats:
                signal_stats[sig] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if trade["result"] == "WIN":
                signal_stats[sig]["wins"] += 1
            else:
                signal_stats[sig]["losses"] += 1
            signal_stats[sig]["pnl"] += trade["pnl"]

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0

        print("-" * 100)
        print("\n{:^100}".format("SUMMARY STATISTICS"))
        print("-" * 100)
        print(f"Total Trades:     {total}")
        print(f"Wins:             {wins}")
        print(f"Losses:           {losses}")
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Total P&L:        ${total_pnl:.2f}")
        print(f"Average P&L:      ${total_pnl / total:.2f}" if total > 0 else "N/A")

        print("\n{:^100}".format("PERFORMANCE BY SIGNAL TYPE"))
        print("-" * 100)
        print("{:<30} {:<10} {:<10} {:<12} {:<15}".format("Signal Type", "Wins", "Losses", "Win Rate", "Total P&L"))
        print("-" * 100)
        for sig, stats in sorted(signal_stats.items()):
            sig_total = stats["wins"] + stats["losses"]
            sig_win_rate = (stats["wins"] / sig_total * 100) if sig_total > 0 else 0.0
            print(
                "{:<30} {:<10} {:<10} {:<12.2f}% ${:<14.2f}".format(
                    sig[:30], stats["wins"], stats["losses"], sig_win_rate, stats["pnl"]
                )
            )

        print("=" * 100)

    def print_detailed_report(self) -> None:
        self._print_detailed_report()

    def on_stop(self) -> None:
        self._print_detailed_report()
        if self.debug_signals:
            self.print_filter_statistics()


    def stop(self) -> None:
        """Backtrader-compatible alias for end-of-run reporting."""
        self.on_stop()
