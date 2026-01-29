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
            "mss_fvg_bull": 0,
            "mss_fvg_bear": 0,
            "entry_bull": 0,
            "entry_bear": 0,
            "entry_sb_bull": 0,
            "entry_sb_bear": 0,
            "entry_setup01_bull": 0,
            "entry_setup01_bear": 0,
            "entry_ote_bull": 0,
            "entry_ote_bear": 0,
            "stop_invalid_long": 0,
            "stop_invalid_short": 0,
            "target_invalid_long": 0,
            "target_invalid_short": 0,
            "orders_placed": 0,
        }

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

    def notify_order(self, order: bt.Order) -> None:
        if order.status in {order.Completed, order.Canceled, order.Margin, order.Rejected}:
            self.order = None
            return

    def _track_signal_counts(self) -> None:
        if not self.params.debug_signals:
            return
        if int(self.data.mss_fvg_bull[0]) == 1:
            self.signal_stats["mss_fvg_bull"] += 1
        if int(self.data.mss_fvg_bear[0]) == 1:
            self.signal_stats["mss_fvg_bear"] += 1
        if int(self.data.entry_fvg_bull[0]) == 1:
            self.signal_stats["entry_bull"] += 1
        if int(self.data.entry_fvg_bear[0]) == 1:
            self.signal_stats["entry_bear"] += 1
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

    def next(self) -> None:
        self._track_signal_counts()
        if self.order:
            return
        if self.position:
            return

        long_signal = int(self.data.entry_fvg_bull[0]) == 1
        short_signal = int(self.data.entry_fvg_bear[0]) == 1

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
            self.order = self.buy_bracket(
                size=size,
                stopprice=stop_price,
                limitprice=target_price,
            )
            self.signal_stats["orders_placed"] += 1
            self.log(
                "LONG entry={entry:.2f} stop={stop:.2f} target={target:.2f} size={size:.4f}".format(
                    entry=entry_price,
                    stop=stop_price,
                    target=target_price,
                    size=size,
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
            self.order = self.sell_bracket(
                size=size,
                stopprice=stop_price,
                limitprice=target_price,
            )
            self.signal_stats["orders_placed"] += 1
            self.log(
                "SHORT entry={entry:.2f} stop={stop:.2f} target={target:.2f} size={size:.4f}".format(
                    entry=entry_price,
                    stop=stop_price,
                    target=target_price,
                    size=size,
                )
            )

    def stop(self) -> None:
        if not self.params.debug_signals:
            return
        self.log(
            "Signal summary | mss_fvg_bull={mss_fvg_bull} mss_fvg_bear={mss_fvg_bear} "
            "entry_bull={entry_bull} entry_bear={entry_bear} "
            "entry_sb_bull={entry_sb_bull} entry_sb_bear={entry_sb_bear} "
            "entry_setup01_bull={entry_setup01_bull} entry_setup01_bear={entry_setup01_bear} "
            "entry_ote_bull={entry_ote_bull} entry_ote_bear={entry_ote_bear} "
            "orders={orders} stop_invalid_long={stop_long} stop_invalid_short={stop_short} "
            "target_invalid_long={target_long} target_invalid_short={target_short}".format(
                mss_fvg_bull=self.signal_stats["mss_fvg_bull"],
                mss_fvg_bear=self.signal_stats["mss_fvg_bear"],
                entry_bull=self.signal_stats["entry_bull"],
                entry_bear=self.signal_stats["entry_bear"],
                entry_sb_bull=self.signal_stats["entry_sb_bull"],
                entry_sb_bear=self.signal_stats["entry_sb_bear"],
                entry_setup01_bull=self.signal_stats["entry_setup01_bull"],
                entry_setup01_bear=self.signal_stats["entry_setup01_bear"],
                entry_ote_bull=self.signal_stats["entry_ote_bull"],
                entry_ote_bear=self.signal_stats["entry_ote_bear"],
                orders=self.signal_stats["orders_placed"],
                stop_long=self.signal_stats["stop_invalid_long"],
                stop_short=self.signal_stats["stop_invalid_short"],
                target_long=self.signal_stats["target_invalid_long"],
                target_short=self.signal_stats["target_invalid_short"],
            )
        )
