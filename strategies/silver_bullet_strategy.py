"""Silver Bullet Strategy Skeleton for Backtrader.

This skeleton wires the translated Pine indicators into a Backtrader Strategy
state machine. It mirrors the logic plan:
- Killzone/session gate
- Weekly Monday Range context + inducement sweeps
- HTF bias alignment
- Liquidity sweep before MSS
- MSS + displacement confirmation
- FVG/OB/BPR POI selection + entry triggers

It is intentionally conservative and uses the translated indicator outputs as
inputs to a staged decision process.
"""

from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from typing import Dict, Optional

import backtrader as bt
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module(module_name: str, relative_path: str):
    module_path = os.path.join(ROOT_DIR, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANDLE_SWEEPS = _load_module("candela_htf_sweeps", "CandelaCharts - HTF Sweeps.py")
MONDAY_RANGE = _load_module("monday_range", "Monday_Range__Lines_.py")
SILVER_BULLET_TF = _load_module(
    "silver_bullet_tradingfinder",
    "Silver_Bullet_ICT_Strategy__TradingFinder__10-11_AM_NY_Time__FVG_TFlab_Silver_Bullet.py",
)
ICT_CUSTOM = _load_module(
    "ict_custom_levels",
    "ICT_Customizable_50__Line___DailyAsiaLondonNew_York_HighLow___True_Day_Open.py",
)
ASIA_LONDON = _load_module("asia_london_levels", "SW's AsiaLondon HL's.py")
MARKET_STRUCTURE = _load_module("market_structure_mtf", "Market Structure MTF Trend [Pt].py")
LIQ_INDUCEMENTS = _load_module("liquidity_inducements", "Liquidity & inducements.py")
LUXALGO_SB = _load_module(
    "luxalgo_silver_bullet",
    "ICT_Silver_Bullet__LuxAlgo___shorttitle__LuxAlgo_-_ICT_Silver_Bullet.py",
)
LEC_DISP = _load_module("lec_displacement", "Liquidity Engulfing & Displacement [MsF].py")
BPR_IFVG = _load_module("bpr_ifvg", "ICT Balanced Price Range [TradingFinder] BPR FVG + IFVG.py")
SMART_MONEY_ZONES = _load_module(
    "smart_money_zones",
    "Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.py",
)
SETUP_01 = _load_module(
    "setup_01",
    "ICT Setup 01 [TradingFinder] FVG + Liquidity SweepsHunt Alerts, ICT Setup 01 TFlab.py",
)
ONE_TRADING_SETUP = _load_module(
    "one_trading_setup",
    "One Trading Setup for Life ICT [TradingFinder] Sweep Session FVG.py",
)
FIB_O = _load_module("fib_ote", "Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py")
MIRPAPA_FOB = _load_module("mirpapa_fob", "MirPapa-ICT-HTF- FVG OB Threeple (EN).py")
SB_WITH_SIGNALS = _load_module("sb_with_signals", "ICT_Silver_Bullet_with_signals.py")


@dataclass
class StrategyState:
    last_bias: int = 0
    last_sweep_idx: Optional[int] = None
    last_mss_idx: Optional[int] = None
    active_poi: Optional[str] = None


class SilverBulletStrategy(bt.Strategy):
    """Backtrader skeleton implementing the Silver Bullet logic pipeline."""

    params = (
        ("chart_timeframe", "15"),
        ("risk_percent", 1.0),
        ("stop_loss_atr_mult", 2.0),
        ("take_profit_rr", 3.0),
        ("min_rr", 2.0),
        ("use_ote", True),
        ("use_bpr", True),
        ("use_setup01", True),
        ("use_one_trading_setup", True),
        ("use_silver_bullet_signals", False),
        ("sweep_timeframes", [("240", 200, True)]),
        ("market_structure_timeframes", ("60", "240", "1D", "1W")),
        ("market_structure_pivots", (15, 15, 15, 15)),
        ("print_signals", True),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.order = None
        self.stop_loss = None
        self.take_profit = None
        self.state = StrategyState()

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

    def next(self) -> None:
        if self.order:
            return
        if len(self.data) < 50:
            return

        df = self._build_dataframe()
        if df.empty:
            return

        session = SILVER_BULLET_TF.calculate_tradingfinder_silver_bullet(df)
        in_killzone = bool(session["session_levels"].trading_range.iloc[-1] == 1)
        if not in_killzone:
            return

        custom_levels = ICT_CUSTOM.calculate_indicator(df)
        asia_levels = ASIA_LONDON.calculate_asia_london_levels(df)
        _ = (custom_levels, asia_levels)

        monday_out = MONDAY_RANGE.calculate_monday_range(
            df,
            chart_timeframe=self.params.chart_timeframe,
        )
        monday_high = None
        monday_low = None
        if monday_out["monday_order"]:
            latest_key = monday_out["monday_order"][0]
            monday = monday_out["mondays"].get(latest_key)
            if monday:
                monday_high = monday.high
                monday_low = monday.low

        monday_sweep_up = False
        monday_sweep_down = False
        if monday_high is not None and len(self.data) > 1:
            monday_sweep_up = self.data.high[0] > monday_high and self.data.high[-1] <= monday_high
        if monday_low is not None and len(self.data) > 1:
            monday_sweep_down = self.data.low[0] < monday_low and self.data.low[-1] >= monday_low

        sweeps = CANDLE_SWEEPS.calculate_htf_sweeps(df, timeframes=self.params.sweep_timeframes)
        sweep_confirmed = monday_sweep_up or monday_sweep_down
        if sweeps:
            latest_tf = next(iter(sweeps.keys()))
            candles = sweeps[latest_tf]
            if candles:
                last_candle = candles[-1]
                sweep_confirmed = sweep_confirmed or last_candle.bull_sweep or last_candle.bear_sweep

        if not sweep_confirmed:
            return

        market_structure = MARKET_STRUCTURE.calculate_market_structure_mtf(
            df,
            timeframes=self.params.market_structure_timeframes,
            pivot_strengths=self.params.market_structure_pivots,
        )
        bias_series = market_structure.tf2.data.trend
        bias = 1 if bias_series.iloc[-1] else -1

        fob = MIRPAPA_FOB.calculate_fvg_ob_threeple(df, chart_timeframe=self.params.chart_timeframe)
        _ = fob

        luxalgo = LUXALGO_SB.calculate_luxalgo_silver_bullet(df)
        mss_confirmed = any(state.trend != 0 for state in luxalgo["bar_states"][-3:])

        displacement = LEC_DISP.calculate_displacement(df)
        displacement_confirmed = bool(displacement.displacement_bar.iloc[-1])

        if not (mss_confirmed and displacement_confirmed):
            return

        poi_confirmed = False
        if self.params.use_bpr:
            bpr = BPR_IFVG.calculate_bpr_indicator(df)
            poi_confirmed = len(bpr["bprs"]) > 0

        zones = SMART_MONEY_ZONES.calculate_smart_money_zones(df)
        poi_confirmed = poi_confirmed or bool(zones["bull_fvg"]) or bool(zones["bear_fvg"])

        if not poi_confirmed:
            return

        trigger = False
        if self.params.use_setup01:
            setup01 = SETUP_01.calculate_setup_01(df)
            if setup01["signals"]:
                trigger = trigger or bool(setup01["signals"][-1].long_signal)

        if self.params.use_one_trading_setup:
            setup = ONE_TRADING_SETUP.calculate_one_trading_setup(df)
            trigger = trigger or bool(setup["cisd"].bull_trigger.iloc[-1])

        if self.params.use_silver_bullet_signals:
            sb_signals = SB_WITH_SIGNALS.detect_silver_bullet_signals(df)
            trigger = trigger or bool(sb_signals["signals"]["fvg_activated"].iloc[-1])

        if self.params.use_ote:
            ote = FIB_O.calculate_fibonacci_ote(df)
            trigger = trigger and bool(ote["states"][-1].pos != 0)

        if not trigger:
            return

        if not self.position:
            stop_distance = float(self.atr[0]) * self.params.stop_loss_atr_mult
            risk_amount = self.broker.getvalue() * (self.params.risk_percent / 100.0)
            size = risk_amount / stop_distance if stop_distance > 0 else 0
            if bias > 0:
                self.order = self.buy(size=size)
                self.stop_loss = self.data.close[0] - stop_distance
                self.take_profit = self.data.close[0] + (stop_distance * self.params.take_profit_rr)
                self.log("ENTER LONG (Silver Bullet skeleton)")
            else:
                self.order = self.sell(size=size)
                self.stop_loss = self.data.close[0] + stop_distance
                self.take_profit = self.data.close[0] - (stop_distance * self.params.take_profit_rr)
                self.log("ENTER SHORT (Silver Bullet skeleton)")

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
