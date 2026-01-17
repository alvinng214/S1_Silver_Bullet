"""ICT Silver Bullet with Signals.

Translated from Pine Script by fikira.

This module mirrors the Pine Script logic by:
- Tracking Silver Bullet sessions (LN/AM/PM) plus pre-sessions in NY time.
- Building swing pivots and MSS trend state from pivot sequences.
- Detecting FVGs inside sessions with optional trend filter and HTF handling.
- Managing FVG lifecycle (active/current/broken) with optional removal on session end.
- Building target lines from swing, daily, and weekly pivot levels.
- Emitting per-bar signal series for FVG formation, retrace, cancel, and target hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    index: int
    price: float
    kind: str  # "high", "low", or "any"
    source: str  # "swing", "daily", "weekly"


@dataclass
class TargetLine:
    price: float
    source_idx: int
    source_type: str  # 'pivot_high', 'pivot_low', 'weekly', 'daily'
    active: bool = True
    reached: bool = False


@dataclass
class FVG:
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_bullish: bool
    formed_session: str
    active: bool = False
    broken: bool = False
    current: bool = True
    targets: List[TargetLine] = field(default_factory=list)
    target_count: int = 0


@dataclass
class SilverBulletSession:
    name: str
    start_time: time
    end_time: time
    start_idx: int
    end_idx: int
    fvgs: List[FVG]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    if "time" in df.columns:
        df.index = pd.to_datetime(df["time"])
    else:
        df.index = pd.to_datetime(df.index)
    return df


def _ny_time_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("America/New_York")
    return index.tz_convert("America/New_York")


def _in_session(dt: pd.Timestamp, start: str, end: str) -> bool:
    start_h, start_m = int(start[:2]), int(start[2:])
    end_h, end_m = int(end[:2]), int(end[2:])
    start_t = time(start_h, start_m)
    end_t = time(end_h, end_m)
    return start_t <= dt.time() < end_t


def _pivot_series(df: pd.DataFrame, left: int, right: int) -> Tuple[np.ndarray, np.ndarray]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    ph = np.full(len(df), np.nan)
    pl = np.full(len(df), np.nan)
    for i in range(left, len(df) - right):
        pivot_high = True
        pivot_low = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                pivot_high = False
            if lows[j] <= lows[i]:
                pivot_low = False
            if not pivot_high and not pivot_low:
                break
        if pivot_high:
            ph[i + right] = highs[i]
        if pivot_low:
            pl[i + right] = lows[i]
    return ph, pl


def _infer_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    deltas = index.to_series().diff().dropna()
    median = deltas.median()
    if pd.isna(median):
        return 1.0
    return max(median.total_seconds() / 60.0, 1.0)


def _update_zz(a_d: List[int], a_x: List[int], a_y: List[float], direction: int, x1: int, y1: float, x2: int, y2: float) -> None:
    a_d.insert(0, direction)
    a_x.insert(0, x2)
    a_y.insert(0, y2)
    a_d.pop()
    a_x.pop()
    a_y.pop()


def _minimum_trade_framework(symbol_type: str, min_tick: float) -> float:
    if symbol_type == "forex":
        return min_tick * 15 * 10
    if symbol_type in {"index", "futures"}:
        return min_tick * 40
    return 0.0


def _create_target_lines(
    pivots: List[Pivot],
    fvg: FVG,
    current_idx: int,
    df: pd.DataFrame,
    diff_level: float,
    extend_left: bool,
) -> List[TargetLine]:
    targets: List[TargetLine] = []
    for pivot in pivots:
        if pivot.index >= fvg.start_idx:
            continue
        if fvg.is_bullish:
            if pivot.source == "swing" and pivot.kind != "high":
                continue
            if pivot.price <= diff_level:
                continue
            if current_idx - pivot.index >= 4500:
                continue
            broken = False
            for i in range(pivot.index, current_idx):
                body_high = max(df["open"].iloc[i], df["close"].iloc[i])
                body_low = min(df["open"].iloc[i], df["close"].iloc[i])
                if body_high > pivot.price and body_low < pivot.price:
                    broken = True
                    break
            if not broken:
                targets.append(
                    TargetLine(
                        price=pivot.price,
                        source_idx=pivot.index if extend_left else current_idx,
                        source_type=f"pivot_{pivot.kind}" if pivot.source == "swing" else pivot.source,
                    )
                )
        else:
            if pivot.source == "swing" and pivot.kind != "low":
                continue
            if pivot.price >= diff_level:
                continue
            if current_idx - pivot.index >= 4500:
                continue
            broken = False
            for i in range(pivot.index, current_idx):
                body_high = max(df["open"].iloc[i], df["close"].iloc[i])
                body_low = min(df["open"].iloc[i], df["close"].iloc[i])
                if body_high > pivot.price and body_low < pivot.price:
                    broken = True
                    break
            if not broken:
                targets.append(
                    TargetLine(
                        price=pivot.price,
                        source_idx=pivot.index if extend_left else current_idx,
                        source_type=f"pivot_{pivot.kind}" if pivot.source == "swing" else pivot.source,
                    )
                )
    return targets


def detect_silver_bullet_signals(
    df: pd.DataFrame,
    *,
    left: int = 10,
    right: int = 1,
    htf_minutes: int = 15,
    filter_by_trend: bool = True,
    remove_broken_fvg: bool = True,
    extend_fvg: bool = True,
    extend_left: bool = False,
    symbol_type: str = "forex",
    min_tick: float = 0.0001,
    last_bars_only: bool = False,
    last_bars_count: int = 3000,
) -> Dict[str, object]:
    df = _ensure_datetime_index(df)
    df = df.sort_index()
    ny_index = _ny_time_index(df.index)
    bar_minutes = _infer_minutes(df.index)
    is_htf = htf_minutes > bar_minutes
    max_size = 100
    last_index = len(df) - 1

    ph_series, pl_series = _pivot_series(df, left=left, right=right)

    a_d = [0 for _ in range(max_size)]
    a_x = [0 for _ in range(max_size)]
    initial_high = float(df["high"].iloc[0]) if len(df) else 0.0
    a_y = [initial_high for _ in range(max_size)]
    a_trend = [0]
    trend_values = []

    piv_h: List[Pivot] = []
    piv_l: List[Pivot] = []
    piv_w: List[Pivot] = []
    piv_d: List[Pivot] = []

    b_fvg_bull: List[FVG] = []
    b_fvg_bear: List[FVG] = []

    bp_h: List[Tuple[int, float]] = [(0, np.nan)]
    bp_l: List[Tuple[int, float]] = [(0, np.nan)]

    fri_close: Optional[float] = None
    fri_index: Optional[int] = None

    signals = {
        "bull_fvg_formed": pd.Series(False, index=df.index),
        "bull_fvg_cancel": pd.Series(False, index=df.index),
        "bull_fvg_retrace": pd.Series(False, index=df.index),
        "bull_target_hit": pd.Series(False, index=df.index),
        "bear_fvg_formed": pd.Series(False, index=df.index),
        "bear_fvg_cancel": pd.Series(False, index=df.index),
        "bear_fvg_retrace": pd.Series(False, index=df.index),
        "bear_target_hit": pd.Series(False, index=df.index),
    }

    min_tf = _minimum_trade_framework(symbol_type, min_tick)

    prev_is_in_sb = False
    prev_is_pre_sb = False
    prev_end_sb = False
    sessions: List[SilverBulletSession] = []
    current_session: Optional[SilverBulletSession] = None

    for i in range(len(df)):
        dt = ny_index[i]
        last_bars_ok = not last_bars_only or (last_index - i < last_bars_count)

        sb_ln = _in_session(dt, "0300", "0400")
        sb_am = _in_session(dt, "1000", "1100")
        sb_pm = _in_session(dt, "1400", "1500")
        pre_ln = _in_session(dt, "0230", "0300")
        pre_am = _in_session(dt, "0930", "1000")
        pre_pm = _in_session(dt, "1330", "1400")

        is_in_sb = sb_ln or sb_am or sb_pm
        is_pre_sb = pre_ln or pre_am or pre_pm
        str_sb_pre = is_pre_sb and not prev_is_pre_sb
        str_sb = is_in_sb and not prev_is_in_sb
        end_sb = not is_in_sb and prev_is_in_sb

        if is_in_sb:
            name = "LN" if sb_ln else "AM" if sb_am else "PM"
            if current_session is None or current_session.name != name:
                if current_session is not None:
                    current_session.end_idx = i - 1
                    sessions.append(current_session)
                current_session = SilverBulletSession(
                    name=name,
                    start_time=dt.time(),
                    end_time=dt.time(),
                    start_idx=i,
                    end_idx=i,
                    fvgs=[],
                )
            else:
                current_session.end_idx = i
                current_session.end_time = dt.time()
        else:
            if current_session is not None:
                current_session.end_idx = i - 1
                sessions.append(current_session)
                current_session = None

        ph = ph_series[i]
        pl = pl_series[i]
        x2 = i - right
        if not np.isnan(ph):
            piv_h.insert(0, Pivot(index=x2, price=float(ph), kind="high", source="swing"))
            if len(piv_h) > max_size:
                piv_h.pop()
            direction = a_d[0]
            x1 = a_x[0]
            y1 = a_y[0]
            y2 = float(ph)
            if y2 > y1:
                if direction < 1:
                    _update_zz(a_d, a_x, a_y, 1, x1, y1, x2, y2)
                elif direction == 1:
                    a_x[0] = x2
                    a_y[0] = y2

        if not np.isnan(pl):
            piv_l.insert(0, Pivot(index=x2, price=float(pl), kind="low", source="swing"))
            if len(piv_l) > max_size:
                piv_l.pop()
            direction = a_d[0]
            x1 = a_x[0]
            y1 = a_y[0]
            y2 = float(pl)
            if y2 < y1:
                if direction > -1:
                    _update_zz(a_d, a_x, a_y, -1, x1, y1, x2, y2)
                elif direction == -1:
                    a_x[0] = x2
                    a_y[0] = y2

        i_h = 2 if a_d[2] == 1 else 1
        i_l = 2 if a_d[2] == -1 else 1
        close = float(df["close"].iloc[i])
        if close > a_y[i_h] and a_d[i_h] == 1 and a_trend[0] < 1:
            a_trend[0] = 1
        if close < a_y[i_l] and a_d[i_l] == -1 and a_trend[0] > -1:
            a_trend[0] = -1

        dow = dt.dayofweek
        if dow == 4:
            fri_close = close
            fri_index = i

        if i > 0 and ny_index[i - 1].date() != dt.date():
            if dow == 0 and fri_close is not None and fri_index is not None:
                piv_w.insert(0, Pivot(index=i, price=float(df["open"].iloc[i]), kind="any", source="weekly"))
                piv_w.insert(0, Pivot(index=fri_index, price=float(fri_close), kind="any", source="weekly"))
            piv_d.insert(0, Pivot(index=i, price=float(df["open"].iloc[i]), kind="any", source="daily"))
            piv_d.insert(0, Pivot(index=i - 1, price=float(df["close"].iloc[i - 1]), kind="any", source="daily"))
            while len(piv_w) > 5:
                piv_w.pop()
            while len(piv_d) > 5:
                piv_d.pop()

        if str_sb_pre and is_htf:
            bp_h.insert(0, (i, float(df["high"].iloc[i])))
            bp_l.insert(0, (i, float(df["low"].iloc[i])))

        if (is_pre_sb or is_in_sb) and last_bars_ok and is_htf:
            idx, current_high = bp_h[0]
            idx_l, current_low = bp_l[0]
            bp_h[0] = (idx, max(current_high, float(df["high"].iloc[i])))
            bp_l[0] = (idx_l, min(current_low, float(df["low"].iloc[i])))

        if is_in_sb and last_bars_ok:
            trend = a_trend[0]
            allow_fvg = True
            if is_htf:
                allow_fvg = len(bp_h) > 2 and dt.minute % htf_minutes == 0
            if allow_fvg:
                if is_htf:
                    hi = bp_h[0][1]
                    lo = bp_l[0][1]
                    hi2 = bp_h[2][1]
                    lo2 = bp_l[2][1]
                    ix = bp_h[0][0]
                    ix2 = bp_h[2][0]
                else:
                    hi = float(df["high"].iloc[i])
                    lo = float(df["low"].iloc[i])
                    hi2 = float(df["high"].iloc[i - 2]) if i >= 2 else hi
                    lo2 = float(df["low"].iloc[i - 2]) if i >= 2 else lo
                    ix = i
                    ix2 = max(i - 2, 0)

                if hi < lo2 and (not filter_by_trend or trend == -1):
                    fvg = FVG(start_idx=ix2, end_idx=ix, top=lo2, bottom=hi, is_bullish=False, formed_session="SB")
                    b_fvg_bear.insert(0, fvg)
                    if current_session is not None:
                        current_session.fvgs.append(fvg)
                    signals["bear_fvg_formed"].iloc[i] = True

                if lo > hi2 and (not filter_by_trend or trend == 1):
                    fvg = FVG(start_idx=ix2, end_idx=ix, top=hi2, bottom=lo, is_bullish=True, formed_session="SB")
                    b_fvg_bull.insert(0, fvg)
                    if current_session is not None:
                        current_session.fvgs.append(fvg)
                    signals["bull_fvg_formed"].iloc[i] = True

            for idx_fvg in range(len(b_fvg_bull) - 1, -1, -1):
                fvg = b_fvg_bull[idx_fvg]
                if not fvg.current:
                    continue
                fvg.end_idx = i
                if close < fvg.bottom:
                    fvg.broken = True
                    signals["bull_fvg_cancel"].iloc[i] = True
                    if remove_broken_fvg:
                        b_fvg_bull.pop(idx_fvg)
                elif not fvg.active and float(df["low"].iloc[i]) < fvg.top:
                    fvg.active = True
                    signals["bull_fvg_retrace"].iloc[i] = True
                    diff = close + min_tf
                    for pivots in (piv_h, piv_w, piv_d):
                        fvg.targets.extend(_create_target_lines(pivots, fvg, i, df, diff, extend_left))
                    fvg.target_count = len(fvg.targets)

            for idx_fvg in range(len(b_fvg_bear) - 1, -1, -1):
                fvg = b_fvg_bear[idx_fvg]
                if not fvg.current:
                    continue
                fvg.end_idx = i
                if close > fvg.top:
                    fvg.broken = True
                    signals["bear_fvg_cancel"].iloc[i] = True
                    if remove_broken_fvg:
                        b_fvg_bear.pop(idx_fvg)
                elif not fvg.active and float(df["high"].iloc[i]) > fvg.bottom:
                    fvg.active = True
                    signals["bear_fvg_retrace"].iloc[i] = True
                    diff = close - min_tf
                    for pivots in (piv_l, piv_w, piv_d):
                        fvg.targets.extend(_create_target_lines(pivots, fvg, i, df, diff, extend_left))
                    fvg.target_count = len(fvg.targets)

        for fvg in b_fvg_bull:
            if fvg.target_count == 0:
                continue
            for target in fvg.targets:
                if not target.active:
                    continue
                if float(df["high"].iloc[i]) > target.price:
                    target.active = False
                    target.reached = True
                    fvg.target_count -= 1
                    signals["bull_target_hit"].iloc[i] = True

        for fvg in b_fvg_bear:
            if fvg.target_count == 0:
                continue
            for target in fvg.targets:
                if not target.active:
                    continue
                if float(df["low"].iloc[i]) < target.price:
                    target.active = False
                    target.reached = True
                    fvg.target_count -= 1
                    signals["bear_target_hit"].iloc[i] = True

        if (is_pre_sb or is_in_sb) and last_bars_ok and is_htf:
            if dt.minute % htf_minutes == 0:
                bp_h.insert(0, (i, float(df["high"].iloc[i])))
                bp_l.insert(0, (i, float(df["low"].iloc[i])))

        if end_sb and last_bars_ok:
            for idx_fvg in range(len(b_fvg_bull) - 1, -1, -1):
                fvg = b_fvg_bull[idx_fvg]
                if not fvg.current:
                    continue
                if (not fvg.active) or (remove_broken_fvg and fvg.broken) or close < fvg.bottom:
                    b_fvg_bull.pop(idx_fvg)
                    signals["bull_fvg_cancel"].iloc[i] = True
            for idx_fvg in range(len(b_fvg_bear) - 1, -1, -1):
                fvg = b_fvg_bear[idx_fvg]
                if not fvg.current:
                    continue
                if (not fvg.active) or (remove_broken_fvg and fvg.broken) or close > fvg.top:
                    b_fvg_bear.pop(idx_fvg)
                    signals["bear_fvg_cancel"].iloc[i] = True
            bp_h = [(i, np.nan)]
            bp_l = [(i, np.nan)]

        if prev_end_sb and last_bars_ok:
            for fvg in b_fvg_bull:
                fvg.active = False
                fvg.current = False
            for fvg in b_fvg_bear:
                fvg.active = False
                fvg.current = False

        trend_values.append(a_trend[0])
        prev_end_sb = end_sb
        prev_is_in_sb = is_in_sb
        prev_is_pre_sb = is_pre_sb

    if current_session is not None:
        sessions.append(current_session)

    return {
        "sessions": sessions,
        "pivot_highs": piv_h,
        "pivot_lows": piv_l,
        "trend": pd.Series(trend_values, index=df.index),
        "fvgs_bull": b_fvg_bull,
        "fvgs_bear": b_fvg_bear,
        "signals": signals,
    }
