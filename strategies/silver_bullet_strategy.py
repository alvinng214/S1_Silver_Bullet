"""Silver Bullet Strategy (S1) draft for Backtrader.

This strategy implements the 13-step Silver Bullet pipeline using the translated
Python indicators. It is organized as a strict gating sequence:
1) HTF bias
2) HTF POI proximity
3) Draw-on-liquidity context
4) Session liquidity pools
5) Killzone window
6) Liquidity sweep
7) MSS/CHOCH
8) FVG formed during MSS displacement
9) Entry on FVG retrace
10) Stop beyond sweep extreme
11) Targets at external liquidity
12) SMT divergence (optional)
13) Displacement quality (optional)

The implementation favors clarity and explicit step gates so it is easy to
validate each stage during backtests.
"""

from __future__ import annotations

import os
import importlib.util
import sys
from datetime import time
from typing import Optional, Tuple

import backtrader as bt
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module(module_name: str, relative_path: str):
    module_path = os.path.join(ROOT_DIR, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CANDLE_SWEEPS = _load_module("candela_htf_sweeps", "CandelaCharts - HTF Sweeps.py")
ASIA_LONDON = _load_module("asia_london_levels", "SW's AsiaLondon HL's.py")
MARKET_STRUCTURE = _load_module("market_structure_mtf", "Market Structure MTF Trend [Pt].py")
LIQ_INDUCEMENTS = _load_module("liquidity_inducements", "Liquidity & inducements.py")
LEC_DISP = _load_module("lec_displacement", "Liquidity Engulfing & Displacement [MsF].py")
BPR_IFVG = _load_module("bpr_ifvg", "ICT Balanced Price Range [TradingFinder] BPR FVG + IFVG.py")
SMART_MONEY_ZONES = _load_module(
    "smart_money_zones",
    "Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.py",
)
CD_SWEEP_CISD = _load_module("cd_sweep_cisd", "cd_sweep&cisd_Cx.py")
FIB_O = _load_module("fib_ote", "Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py")
MIRPAPA_FOB = _load_module("mirpapa_fob", "MirPapa-ICT-HTF- FVG OB Threeple (EN).py")
SB_WITH_SIGNALS = _load_module("sb_with_signals", "ICT_Silver_Bullet_with_signals.py")


class SilverBulletStrategy(bt.Strategy):
    """Backtrader strategy implementing the Silver Bullet S1 pipeline."""

    params = (
        ("chart_timeframe", "15"),
        ("risk_percent", 1.0),
        ("take_profit_rr", 2.0),
        ("min_rr", 2.0),
        ("use_ote", True),
        ("use_bpr", True),
        ("use_smt", False),
        ("use_displacement_filter", True),
        ("sweep_timeframes", [("240", 200, True)]),
        ("market_structure_timeframes", ("60", "240", "1D", "1W")),
        ("market_structure_pivots", (15, 15, 15, 15)),
        ("print_signals", True),
        ("sweep_lookback", 5),
    )

    def __init__(self):
        self.order = None
        self.stop_loss = None
        self.take_profit = None

    def log(self, txt: str, dt=None) -> None:
        dt = dt or self.datas[0].datetime.datetime(0)
        if self.params.print_signals:
            print(f"{dt.isoformat()}: {txt}")

    def _build_dataframe(self) -> pd.DataFrame:
        size = len(self.data)
        if size == 0:
            return pd.DataFrame()
        opens = list(self.data.open.get(size=size))
        highs = list(self.data.high.get(size=size))
        lows = list(self.data.low.get(size=size))
        closes = list(self.data.close.get(size=size))
        dt_values = self.data.datetime.get(size=size)
        times = [bt.num2date(v) for v in dt_values]
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes},
            index=pd.DatetimeIndex(times),
        )
        return df.sort_index()

    def notify_order(self, order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self.log(f"{action} EXECUTED @ {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order canceled/margin/rejected")
        self.order = None

    def _in_killzone(self, dt: pd.Timestamp) -> bool:
        ny_dt = dt.tz_convert("America/New_York") if dt.tzinfo else dt.tz_localize("America/New_York")
        time_val = ny_dt.time()
        return (
            time(3, 0) <= time_val < time(4, 0)
            or time(10, 0) <= time_val < time(11, 0)
            or time(14, 0) <= time_val < time(15, 0)
        )

    def _latest_bool(self, series: pd.Series) -> bool:
        if series.empty:
            return False
        return bool(series.iloc[-1])

    def _recent_sweep(self, sweeps: Tuple[object, ...], lookback: int) -> bool:
        for sweep in sweeps[:lookback]:
            if sweep.taken and not sweep.invalidated:
                return True
        return False

    def _sweep_stop(self, sweeps: Tuple[object, ...], direction: int) -> Optional[float]:
        for sweep in sweeps:
            if sweep.taken and not sweep.invalidated and sweep.pivot.type == direction:
                return float(sweep.pivot.price)
        return None

    def next(self) -> None:
        if self.order:
            return
        if len(self.data) < 50:
            return

        df = self._build_dataframe()
        if df.empty:
            return

        # Step 1: HTF bias
        market_structure = MARKET_STRUCTURE.calculate_market_structure_mtf(
            df,
            timeframes=self.params.market_structure_timeframes,
            pivot_strengths=self.params.market_structure_pivots,
        )
        htf_trend = market_structure.tf3.data.trend.iloc[-1]
        daily_trend = market_structure.tf4.data.trend.iloc[-1]
        bias = 1 if htf_trend > 0 and daily_trend > 0 else -1 if htf_trend < 0 and daily_trend < 0 else 0
        if bias == 0:
            return

        # Step 2: HTF POI
        fob = MIRPAPA_FOB.calculate_fvg_ob_threeple(df, chart_timeframe=self.params.chart_timeframe)
        poi_confirmed = bool(fob.high_tf_boxes or fob.mid_tf_boxes or fob.current_tf_boxes)
        if not poi_confirmed:
            zones = SMART_MONEY_ZONES.calculate_smart_money_zones(df)
            poi_confirmed = bool(zones["bull_fvg"]) or bool(zones["bear_fvg"])
        if not poi_confirmed:
            return

        # Step 3 + 4: External liquidity context & session pools
        _ = CANDLE_SWEEPS.calculate_htf_sweeps(df, timeframes=self.params.sweep_timeframes)
        _ = ASIA_LONDON.calculate_asia_london_levels(df)

        # Step 5: Killzone filter
        if not self._in_killzone(df.index[-1]):
            return

        # Step 6: Liquidity sweep
        liquidity = LIQ_INDUCEMENTS.calculate_liquidity_inducements(df)
        sweeps = tuple(liquidity["sweeps_highs"] + liquidity["sweeps_lows"])
        if not self._recent_sweep(sweeps, self.params.sweep_lookback):
            return

        # Step 7: MSS / CHOCH confirmation (use TF2 change-of-character)
        mss_confirmed = self._latest_bool(market_structure.tf2.bullish_choch) or self._latest_bool(
            market_structure.tf2.bearish_choch
        )
        if not mss_confirmed:
            return

        # Step 8: FVG formed during MSS displacement
        sb_signals = SB_WITH_SIGNALS.detect_silver_bullet_signals(df)
        fvg_formed = self._latest_bool(sb_signals["signals"]["bull_fvg_formed"]) or self._latest_bool(
            sb_signals["signals"]["bear_fvg_formed"]
        )
        if not fvg_formed:
            return

        # Step 9: Entry on FVG retrace
        entry_trigger = self._latest_bool(sb_signals["signals"]["bull_fvg_retrace"]) or self._latest_bool(
            sb_signals["signals"]["bear_fvg_retrace"]
        )
        if self.params.use_bpr:
            bpr = BPR_IFVG.calculate_bpr_indicator(df)
            entry_trigger = entry_trigger and bool(bpr["bprs"])
        if self.params.use_ote:
            ote = FIB_O.calculate_fibonacci_ote(df)
            entry_trigger = entry_trigger and bool(ote["states"][-1].pos != 0)
        if not entry_trigger:
            return

        # Step 12: Optional SMT divergence confirmation
        if self.params.use_smt:
            smt = CD_SWEEP_CISD.detect_cd_sweep_cisd(df)
            smt_ok = any(
                signal.idx == len(df) - 1 and (signal.is_low_smt or signal.is_high_smt)
                for signal in smt["smt_signals"]
            )
            entry_trigger = entry_trigger and smt_ok
        if not entry_trigger:
            return

        # Step 13: Optional displacement confirmation
        if self.params.use_displacement_filter:
            displacement = LEC_DISP.calculate_displacement(df)
            if not self._latest_bool(displacement.displacement_bar):
                return

        if not self.position:
            stop_price = self._sweep_stop(sweeps, -1 if bias > 0 else 1)
            if stop_price is None:
                return
            entry_price = float(self.data.close[0])
            stop_distance = abs(entry_price - stop_price)
            if stop_distance <= 0:
                return
            risk_amount = self.broker.getvalue() * (self.params.risk_percent / 100.0)
            size = risk_amount / stop_distance
            if bias > 0:
                self.order = self.buy(size=size)
                self.stop_loss = stop_price
                self.take_profit = entry_price + (stop_distance * self.params.take_profit_rr)
                self.log("ENTER LONG (Silver Bullet)")
            else:
                self.order = self.sell(size=size)
                self.stop_loss = stop_price
                self.take_profit = entry_price - (stop_distance * self.params.take_profit_rr)
                self.log("ENTER SHORT (Silver Bullet)")

        if self.position.size > 0:
            if self.data.close[0] <= self.stop_loss:
                self.log("STOP LOSS HIT (Long)")
                self.close()
            elif self.data.close[0] >= self.take_profit:
                self.log("TAKE PROFIT HIT (Long)")
                self.close()
        elif self.position.size < 0:
            if self.data.close[0] >= self.stop_loss:
                self.log("STOP LOSS HIT (Short)")
                self.close()
            elif self.data.close[0] <= self.take_profit:
                self.log("TAKE PROFIT HIT (Short)")
                self.close()
