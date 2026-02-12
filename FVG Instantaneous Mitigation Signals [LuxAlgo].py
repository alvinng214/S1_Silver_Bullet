"""
FVG Instantaneous Mitigation Signals [LuxAlgo] - Python translation of the Pine Script.

This module mirrors the original TradingView script logic as closely as possible,
including:
- Bullish/Bearish IMFVG detection rules.
- TP/SL area lifecycle with opposite-signal reset.
- Trailing stop reset/update/reach behavior.
- Average imbalance line extension and level-reached checks.
- Output state per bar for plotting/backtesting.

Original script: "FVG Instantaneous Mitigation Signals [LuxAlgo]"
License: CC BY-NC-SA 4.0 (same as source indicator)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import backtrader as bt


@dataclass
class BoxArea:
    left: int
    right: int
    top: float
    bottom: float


@dataclass
class LineObj:
    x1: int
    x2: int
    y: float


@dataclass
class LabelObj:
    x: int
    y: float
    text: str


@dataclass
class TpslState:
    tp_area: Optional[BoxArea] = None
    sl_area: Optional[BoxArea] = None
    reached: bool = False


@dataclass
class TrailingStopState:
    ts: Optional[float] = None
    reached: bool = False


class FVGInstantaneousMitigationSignalsLuxAlgo(bt.Indicator):
    """Backtrader indicator port of LuxAlgo's Pine script.

    Mirrors the Pine logic bar-by-bar for use in Backtrader strategies.
    """

    lines = (
        "bull",
        "bear",
        "os",
        "bull_reached",
        "bear_reached",
        "ts",
        "ts_raw",
        "ts_reached",
        "bull_line",
        "bear_line",
        "bull_lvl_reached",
        "bear_lvl_reached",
    )

    params = (
        ("filter_width", 0.0),
        ("show_tp", False),
        ("tp_mult", 4.0),
        ("show_sl", False),
        ("sl_mult", 2.0),
        ("ts_reset", "Every Signals"),  # ['Every Signals', 'Inverse Signals']
        ("ts_mult", 3.0),
        ("show_bull", True),
        ("bull_avg", True),
        ("show_bear", True),
        ("bear_avg", True),
    )

    def __init__(self):
        self._atr = bt.ind.ATR(self.data, period=200)
        self._os = 0
        self._prev_os: Optional[int] = None
        self._bull_line: Optional[LineObj] = None
        self._bear_line: Optional[LineObj] = None
        self._bull_lvl_reached: Optional[bool] = None
        self._bear_lvl_reached: Optional[bool] = None
        self._bull_tpsl = TpslState()
        self._bear_tpsl = TpslState()
        self._ts_state = TrailingStopState()

    def _tpsl(self, state: TpslState, condition: bool, opposite_condition: bool, level: float, is_long: bool, n: int) -> bool:
        atr_i = _nz(float(self._atr[0]))
        if condition:
            if is_long:
                if self.p.show_tp:
                    state.tp_area = BoxArea(left=n, right=n, top=level + atr_i * self.p.tp_mult, bottom=level)
                if self.p.show_sl:
                    state.sl_area = BoxArea(left=n, right=n, top=level, bottom=level - atr_i * self.p.sl_mult)
            else:
                if self.p.show_tp:
                    state.tp_area = BoxArea(left=n, right=n, top=level, bottom=level - atr_i * self.p.tp_mult)
                if self.p.show_sl:
                    state.sl_area = BoxArea(left=n, right=n, top=level + atr_i * self.p.sl_mult, bottom=level)

            state.reached = False
        elif opposite_condition:
            state.reached = True

        if not state.reached:
            if state.tp_area is not None:
                state.tp_area.right = n
            if state.sl_area is not None:
                state.sl_area.right = n

            hi = float(self.data.high[0])
            lo = float(self.data.low[0])
            if is_long:
                tp_hit = state.tp_area is not None and hi > state.tp_area.top
                sl_hit = state.sl_area is not None and lo < state.sl_area.bottom
                if tp_hit or sl_hit:
                    state.reached = True
            else:
                tp_hit = state.tp_area is not None and lo < state.tp_area.bottom
                sl_hit = state.sl_area is not None and hi > state.sl_area.top
                if tp_hit or sl_hit:
                    state.reached = True
        return state.reached

    def next(self):
        n = len(self.data) - 1
        atr_i = _nz(float(self._atr[0]))

        # Pine offsets
        low_1 = float(self.data.low[-1]) if len(self.data) >= 2 else np.nan
        low_3 = float(self.data.low[-3]) if len(self.data) >= 4 else np.nan
        high_1 = float(self.data.high[-1]) if len(self.data) >= 2 else np.nan
        high_3 = float(self.data.high[-3]) if len(self.data) >= 4 else np.nan
        close_2 = float(self.data.close[-2]) if len(self.data) >= 3 else np.nan

        def _filter(a: float, b: float) -> bool:
            if np.isnan(a) or np.isnan(b):
                return False
            return (a - b) > (atr_i * self.p.filter_width)

        close0 = float(self.data.close[0])

        bull = (
            (not np.isnan(low_3))
            and (not np.isnan(high_1))
            and (not np.isnan(close_2))
            and (low_3 > high_1)
            and (close_2 < low_3)
            and (close0 > low_3)
            and _filter(low_3, high_1)
            and self.p.show_bull
        )

        bear = (
            (not np.isnan(low_1))
            and (not np.isnan(high_3))
            and (not np.isnan(close_2))
            and (low_1 > high_3)
            and (close_2 > high_3)
            and (close0 < high_3)
            and _filter(low_1, high_3)
            and self.p.show_bear
        )

        if bull:
            if self.p.bull_avg:
                self._bull_line = LineObj(x1=n, x2=n, y=(low_3 + high_1) / 2.0)
            self._os = 1
            self._bull_lvl_reached = False

        if bear:
            if self.p.bear_avg:
                self._bear_line = LineObj(x1=n, x2=n, y=(low_1 + high_3) / 2.0)
            self._os = 0
            self._bear_lvl_reached = False

        if self._bull_lvl_reached is False and self._bull_line is not None:
            self._bull_line.x2 = n
        if self._bear_lvl_reached is False and self._bear_line is not None:
            self._bear_line.x2 = n

        if self._bull_line is not None and close0 < self._bull_line.y:
            self._bull_lvl_reached = True
        if self._bear_line is not None and close0 > self._bear_line.y:
            self._bear_lvl_reached = True

        bull_level = (low_3 + high_1) / 2.0 if not (np.isnan(low_3) or np.isnan(high_1)) else np.nan
        bear_level = (low_1 + high_3) / 2.0 if not (np.isnan(low_1) or np.isnan(high_3)) else np.nan
        bull_reached = self._tpsl(self._bull_tpsl, bull, bear, _nz(bull_level), True, n)
        bear_reached = self._tpsl(self._bear_tpsl, bear, bull, _nz(bear_level), False, n)

        if self.p.ts_reset == "Every Signals":
            ts_trigger = bull or bear
        else:
            ts_trigger = (self._prev_os is not None) and (self._os != self._prev_os)

        if ts_trigger:
            self._ts_state.ts = close0 - atr_i * self.p.ts_mult if self._os == 1 else close0 + atr_i * self.p.ts_mult
            self._ts_state.reached = False
        else:
            if self._ts_state.ts is not None:
                if self._os == 1:
                    if close0 - self._ts_state.ts > atr_i * self.p.ts_mult:
                        self._ts_state.ts = close0 - atr_i * self.p.ts_mult
                else:
                    if self._ts_state.ts - close0 > atr_i * self.p.ts_mult:
                        self._ts_state.ts = close0 + atr_i * self.p.ts_mult

                if close0 < self._ts_state.ts and self._os == 1:
                    self._ts_state.reached = True
                elif close0 > self._ts_state.ts and self._os == 0:
                    self._ts_state.reached = True

        self.lines.bull[0] = 1.0 if bull else 0.0
        self.lines.bear[0] = 1.0 if bear else 0.0
        self.lines.os[0] = float(self._os)
        self.lines.bull_reached[0] = 1.0 if bull_reached else 0.0
        self.lines.bear_reached[0] = 1.0 if bear_reached else 0.0
        self.lines.ts_reached[0] = 1.0 if self._ts_state.reached else 0.0
        self.lines.ts_raw[0] = np.nan if self._ts_state.ts is None else self._ts_state.ts
        self.lines.ts[0] = np.nan if (self._ts_state.reached or bull or bear) else self.lines.ts_raw[0]
        self.lines.bull_line[0] = np.nan if self._bull_line is None else self._bull_line.y
        self.lines.bear_line[0] = np.nan if self._bear_line is None else self._bear_line.y
        self.lines.bull_lvl_reached[0] = np.nan if self._bull_lvl_reached is None else (1.0 if self._bull_lvl_reached else 0.0)
        self.lines.bear_lvl_reached[0] = np.nan if self._bear_lvl_reached is None else (1.0 if self._bear_lvl_reached else 0.0)

        self._prev_os = self._os

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _pine_rma(values: pd.Series, length: int) -> pd.Series:
    """Match Pine ta.rma/ta.atr behavior (SMA seed, then Wilder smoothing)."""
    out = np.full(len(values), np.nan, dtype=float)
    arr = values.to_numpy(dtype=float)

    if len(arr) < length:
        return pd.Series(out, index=values.index)

    seed_idx = length - 1
    seed = np.nanmean(arr[:length])
    out[seed_idx] = seed

    alpha = 1.0 / float(length)
    for i in range(seed_idx + 1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]

    return pd.Series(out, index=values.index)


def _nz(value: float, replacement: float = 0.0) -> float:
    if value is None:
        return replacement
    if isinstance(value, float) and np.isnan(value):
        return replacement
    return float(value)


def _series_at(series: np.ndarray, i: int, bars_back: int) -> float:
    j = i - bars_back
    if j < 0:
        return np.nan
    return float(series[j])


def calculate_fvg_instantaneous_mitigation_signals(
    df: pd.DataFrame,
    *,
    filter_width: float = 0.0,
    show_tp: bool = False,
    tp_mult: float = 4.0,
    show_sl: bool = False,
    sl_mult: float = 2.0,
    ts_reset: str = "Every Signals",  # ['Every Signals', 'Inverse Signals']
    ts_mult: float = 3.0,
    show_bull: bool = True,
    bull_avg: bool = True,
    show_bear: bool = True,
    bear_avg: bool = True,
) -> Dict[str, object]:
    """Strict Python mirror of the LuxAlgo Pine indicator logic.

    Required columns: open, high, low, close.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(c.lower() for c in df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    # Case-insensitive column normalization
    cols = {c.lower(): c for c in df.columns}
    high_s = df[cols["high"]].astype(float)
    low_s = df[cols["low"]].astype(float)
    close_s = df[cols["close"]].astype(float)

    # Pine: atr = nz(ta.atr(200))
    tr = _true_range(high_s, low_s, close_s)
    atr_raw = _pine_rma(tr, 200)
    atr = atr_raw.fillna(0.0).to_numpy(dtype=float)

    high = high_s.to_numpy(dtype=float)
    low = low_s.to_numpy(dtype=float)
    close = close_s.to_numpy(dtype=float)

    # Persistent state (Pine var)
    bull_line: Optional[LineObj] = None
    bull_lvl_reached: Optional[bool] = None
    bear_line: Optional[LineObj] = None
    bear_lvl_reached: Optional[bool] = None
    os = 0
    prev_os: Optional[int] = None

    bull_tpsl = TpslState()
    bear_tpsl = TpslState()
    ts_state = TrailingStopState()

    imbalance_boxes: List[Dict[str, object]] = []
    labels: List[LabelObj] = []

    out_rows: List[Dict[str, object]] = []

    for i in range(len(df)):
        n = i
        atr_i = float(atr[i])

        low_1 = _series_at(low, i, 1)
        low_3 = _series_at(low, i, 3)
        high_1 = _series_at(high, i, 1)
        high_3 = _series_at(high, i, 3)
        close_2 = _series_at(close, i, 2)

        def _filter(a: float, b: float) -> bool:
            if np.isnan(a) or np.isnan(b):
                return False
            return (a - b) > (atr_i * filter_width)

        # Bullish Signals
        bull = (
            (not np.isnan(low_3))
            and (not np.isnan(high_1))
            and (not np.isnan(close_2))
            and (low_3 > high_1)
            and (close_2 < low_3)
            and (close[i] > low_3)
            and _filter(low_3, high_1)
            and show_bull
        )

        if bull:
            imbalance_boxes.append(
                {
                    "type": "bull",
                    "left": n - 3,
                    "right": n,
                    "top": low_3,
                    "bottom": high_1,
                }
            )
            avg = (low_3 + high_1) / 2.0

            if bull_avg:
                bull_line = LineObj(x1=n, x2=n, y=avg)

            labels.append(LabelObj(x=n, y=float(low[i]), text="▲"))

            os = 1
            bull_lvl_reached = False

        # Bearish Signals
        bear = (
            (not np.isnan(low_1))
            and (not np.isnan(high_3))
            and (not np.isnan(close_2))
            and (low_1 > high_3)
            and (close_2 > high_3)
            and (close[i] < high_3)
            and _filter(low_1, high_3)
            and show_bear
        )

        if bear:
            imbalance_boxes.append(
                {
                    "type": "bear",
                    "left": n - 3,
                    "right": n,
                    "top": low_1,
                    "bottom": high_3,
                }
            )
            avg = (low_1 + high_3) / 2.0

            if bear_avg:
                bear_line = LineObj(x1=n, x2=n, y=avg)

            labels.append(LabelObj(x=n, y=float(high[i]), text="▼"))

            os = 0
            bear_lvl_reached = False

        # Extend average imbalance areas
        if bull_lvl_reached is False and bull_line is not None:
            bull_line.x2 = n
        if bear_lvl_reached is False and bear_line is not None:
            bear_line.x2 = n

        # Test if reached
        if bull_line is not None and close[i] < bull_line.y:
            bull_lvl_reached = True

        if bear_line is not None and close[i] > bear_line.y:
            bear_lvl_reached = True

        # tpsl() mirror, separated by call-site state (bull call and bear call)
        def _tpsl(
            state: TpslState,
            condition: bool,
            opposite_condition: bool,
            level: float,
            is_long: bool,
        ) -> bool:
            if condition:
                if is_long:
                    if show_tp:
                        state.tp_area = BoxArea(left=n, right=n, top=level + atr_i * tp_mult, bottom=level)
                    if show_sl:
                        state.sl_area = BoxArea(left=n, right=n, top=level, bottom=level - atr_i * sl_mult)
                else:
                    if show_tp:
                        state.tp_area = BoxArea(left=n, right=n, top=level, bottom=level - atr_i * tp_mult)
                    if show_sl:
                        state.sl_area = BoxArea(left=n, right=n, top=level + atr_i * sl_mult, bottom=level)

                state.reached = False

            elif opposite_condition:
                state.reached = True

            if not state.reached:
                if state.tp_area is not None:
                    state.tp_area.right = n
                if state.sl_area is not None:
                    state.sl_area.right = n

                if is_long:
                    tp_hit = state.tp_area is not None and high[i] > state.tp_area.top
                    sl_hit = state.sl_area is not None and low[i] < state.sl_area.bottom
                    if tp_hit or sl_hit:
                        state.reached = True
                else:
                    tp_hit = state.tp_area is not None and low[i] < state.tp_area.bottom
                    sl_hit = state.sl_area is not None and high[i] > state.sl_area.top
                    if tp_hit or sl_hit:
                        state.reached = True

            return state.reached

        bull_level = (low_3 + high_1) / 2.0 if not (np.isnan(low_3) or np.isnan(high_1)) else np.nan
        bear_level = (low_1 + high_3) / 2.0 if not (np.isnan(low_1) or np.isnan(high_3)) else np.nan
        bull_reached = _tpsl(bull_tpsl, bull, bear, _nz(bull_level), True)
        bear_reached = _tpsl(bear_tpsl, bear, bull, _nz(bear_level), False)

        # Trailing Stop
        if ts_reset == "Every Signals":
            ts_trigger = bull or bear
        else:
            ts_trigger = (prev_os is not None) and (os != prev_os)

        if ts_trigger:
            if os == 1:
                ts_state.ts = close[i] - atr_i * ts_mult
            else:
                ts_state.ts = close[i] + atr_i * ts_mult
            ts_state.reached = False
        else:
            if ts_state.ts is not None:
                if os == 1:
                    if close[i] - ts_state.ts > atr_i * ts_mult:
                        ts_state.ts = close[i] - atr_i * ts_mult
                else:
                    if ts_state.ts - close[i] > atr_i * ts_mult:
                        ts_state.ts = close[i] + atr_i * ts_mult

                if close[i] < ts_state.ts and os == 1:
                    ts_state.reached = True
                elif close[i] > ts_state.ts and os == 0:
                    ts_state.reached = True

        # Plot/barcolor-state equivalents
        if ts_state.reached:
            bar_color = None
        elif os == 1 and (not bull_reached):
            bar_color = "bull"
        elif os == 0 and (not bear_reached):
            bar_color = "bear"
        else:
            bar_color = None

        if ts_state.reached or bull or bear:
            ts_plot = np.nan
        else:
            ts_plot = ts_state.ts if ts_state.ts is not None else np.nan

        out_rows.append(
            {
                "bull": bull,
                "bear": bear,
                "os": os,
                "bull_reached": bull_reached,
                "bear_reached": bear_reached,
                "ts": ts_plot,
                "ts_raw": ts_state.ts if ts_state.ts is not None else np.nan,
                "ts_reached": ts_state.reached,
                "bar_color_state": bar_color,
                "bull_line": np.nan if bull_line is None else bull_line.y,
                "bear_line": np.nan if bear_line is None else bear_line.y,
                "bull_lvl_reached": bull_lvl_reached,
                "bear_lvl_reached": bear_lvl_reached,
            }
        )

        prev_os = os

    states = pd.DataFrame(out_rows, index=df.index)

    return {
        "states": states,
        "imbalance_boxes": imbalance_boxes,
        "labels": labels,
        "bull_tp_area": bull_tpsl.tp_area,
        "bull_sl_area": bull_tpsl.sl_area,
        "bear_tp_area": bear_tpsl.tp_area,
        "bear_sl_area": bear_tpsl.sl_area,
    }


if __name__ == "__main__":
    # Minimal runnable example
    sample = pd.DataFrame(
        {
            "open": [1, 2, 3, 2, 4, 5, 4, 6],
            "high": [2, 3, 4, 3, 5, 6, 5, 7],
            "low": [0.5, 1.5, 2.5, 1.8, 3.5, 4.5, 3.8, 5.8],
            "close": [1.5, 2.5, 3.5, 2.2, 4.5, 5.5, 4.2, 6.5],
        }
    )
    result = calculate_fvg_instantaneous_mitigation_signals(sample)
    print(result["states"].tail())
