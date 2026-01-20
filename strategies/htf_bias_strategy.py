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
