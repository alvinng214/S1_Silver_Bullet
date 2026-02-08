"""Python translation of the Pine Script `MTF FVG x2 [MK]`.

This module mirrors the Pine logic by:
- Detecting multi-timeframe Fair Value Gaps (FVGs) with configurable timeframes.
- Maintaining FVG boxes with mitigation, intrusion, and optional color changes.
- Building a secondary "price overlay" FVG set on a single HTF with fill logic.

The original script uses TradingView drawing primitives (boxes/labels/lines).
Here we expose data structures that represent those objects and their updates
so the same logic can be used programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FVGBox:
    direction: str  # "bull" or "bear"
    timeframe: str
    left_index: int
    right_index: int
    top: float
    bottom: float
    label: Optional[str] = None
    label_time: Optional[pd.Timestamp] = None
    mitigated: bool = False
    intrusion: bool = False
    entered: bool = False


@dataclass
class OverlayBox:
    direction: str  # "up" or "down"
    box_type: str  # "Imbalance", "Gap", "Wick"
    left_time: pd.Timestamp
    right_time: pd.Timestamp
    top: float
    bottom: float
    label: Optional[str] = None
    filled: bool = False


@dataclass
class MTFSettings:
    mtf_imb: bool = True
    mtf_price_overlay: bool = True
    only_market_hours: bool = False
    show_labels: bool = True
    show_time_on_labels: bool = False
    enable_current_timeframe: bool = False
    enable_5min: bool = False
    enable_10min: bool = False
    enable_15min: bool = True
    enable_30min: bool = False
    enable_1hr: bool = True
    enable_4hr: bool = True
    enable_8hr: bool = False
    enable_12hr: bool = False
    enable_daily: bool = True
    enable_weekly: bool = True
    enable_monthly: bool = True
    label_shift: int = 5
    label_shiftr: int = 15
    incursion_pct: int = 20
    entry_change_color: bool = True
    use_body_for_mitigation: bool = True
    mitigation_mode: str = "Normal"  # Normal, Dynamic, None
    session_hours: str = "0930-1600"
    display_ranges: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "15": (1, 4),
            "60": (5, 5),
            "240": (15, 15),
            "D": (60, 60),
            "W": (240, 240),
            "M": (1440, 1440),
        }
    )
    max_per_tf: Dict[str, int] = field(
        default_factory=lambda: {
            "chart": 8,
            "5": 8,
            "10": 8,
            "15": 8,
            "30": 8,
            "60": 8,
            "240": 8,
            "480": 8,
            "720": 8,
            "D": 8,
            "W": 8,
            "M": 8,
        }
    )


@dataclass
class OverlaySettings:
    timeframe: str = "60"
    boxtype: str = "Imbalance"
    showmiddleline: bool = True
    showbottomline: bool = False
    showtopline: bool = False
    showup: bool = True
    showdown: bool = True
    extendtilfilled: bool = True
    filledtype: str = "Full Fill"
    lookback: bool = True
    days_back: float = 5
    hidefilled: bool = True
    showboxes: bool = True
    conditiontype: str = "None"  # None, ATR, Percentage
    atrlength: int = 30
    atrmult: float = 1.0
    pctcond: float = 0.30
    pctmult: float = 1.0
    showboxtext: bool = True
    text_type: str = "Labels + Timeframe"
    max_boxes: int = 499


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def _tf_to_rule(tf: str) -> str:
    if tf in {"5", "10", "15", "30", "60", "240", "480", "720"}:
        return f"{tf}min"
    if tf == "D":
        return "1D"
    if tf == "W":
        return "1W"
    if tf == "M":
        return "1M"
    raise ValueError(f"Unsupported timeframe: {tf}")


def _timeframe_label(tf: str) -> str:
    mapping = {
        "5": "5m",
        "10": "10m",
        "15": "15m",
        "30": "30m",
        "60": "1 Hr",
        "240": "4 Hr",
        "480": "8 Hr",
        "720": "12 Hr",
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly",
    }
    return mapping.get(tf, tf)


def _align_htf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _tf_to_rule(tf)
    htf = _resample_ohlc(df, rule)
    return htf.reindex(df.index, method="ffill")


def _calc_intrusion_threshold(top: float, bottom: float, intrusion_pct: float) -> float:
    return top - (intrusion_pct / 100.0) * (top - bottom)


def _calc_bear_intrusion_threshold(top: float, bottom: float, intrusion_pct: float) -> float:
    return bottom + (intrusion_pct / 100.0) * (top - bottom)


def _detect_bull_fvg(fvgmethod: bool, low: float, high2: float) -> bool:
    # In Pine, body/wick branch both use high2 < low with the current config.
    return high2 < low if fvgmethod else high2 < low


def _detect_bear_fvg(fvgmethod: bool, low2: float, high: float) -> bool:
    return low2 > high if fvgmethod else low2 > high


def _mitigation_mode(mode: str) -> int:
    return {"Normal": 1, "Dynamic": 2, "None": 3}.get(mode, 4)


def compute_mtf_fvg_x2(
    df: pd.DataFrame,
    *,
    mtf_settings: Optional[MTFSettings] = None,
    overlay_settings: Optional[OverlaySettings] = None,
    current_timeframe: Optional[str] = None,
    timeframe_minutes: Optional[int] = None,
    is_dwm: bool = False,
) -> Dict[str, List]:
    """Compute MTF FVG x2 outputs.

    Args:
        df: DataFrame with columns open/high/low/close/volume indexed by datetime.
        mtf_settings: Settings for the MTF FVG section.
        overlay_settings: Settings for the overlay FVG section.

    Returns:
        Dict with keys:
            - "mtf_boxes": List[FVGBox]
            - "overlay_boxes": List[OverlayBox]
    """
    if mtf_settings is None:
        mtf_settings = MTFSettings()
    if overlay_settings is None:
        overlay_settings = OverlaySettings()

    fvgmethod = True
    mitigationaction = _mitigation_mode(mtf_settings.mitigation_mode)
    intrusion_percentage = mtf_settings.incursion_pct / 100.0

    mtf_boxes: List[FVGBox] = []
    total_allowed = sum(mtf_settings.max_per_tf.values())
    error = total_allowed > 500

    def _current_tf_enabled() -> bool:
        if not mtf_settings.enable_current_timeframe or current_timeframe is None:
            return False
        enabled_map = {
            "5": mtf_settings.enable_5min,
            "10": mtf_settings.enable_10min,
            "15": mtf_settings.enable_15min,
            "30": mtf_settings.enable_30min,
            "60": mtf_settings.enable_1hr,
            "240": mtf_settings.enable_4hr,
            "480": mtf_settings.enable_8hr,
            "720": mtf_settings.enable_12hr,
            "D": mtf_settings.enable_daily,
            "W": mtf_settings.enable_weekly,
            "M": mtf_settings.enable_monthly,
        }
        return not enabled_map.get(current_timeframe, False)

    def _display_enabled(tf: str) -> bool:
        if timeframe_minutes is None:
            return True
        if tf not in mtf_settings.display_ranges:
            return True
        min_tf, max_tf = mtf_settings.display_ranges[tf]
        if is_dwm:
            return True
        return min_tf <= timeframe_minutes <= max_tf

    def _in_session(ts: pd.Timestamp) -> bool:
        if not mtf_settings.only_market_hours:
            return True
        start, end = mtf_settings.session_hours.split("-")
        start_h, start_m = int(start[:2]), int(start[2:])
        end_h, end_m = int(end[:2]), int(end[2:])
        return (start_h, start_m) <= (ts.hour, ts.minute) <= (end_h, end_m)

    def process_tf(tf: str, enabled: bool) -> None:
        if not enabled:
            return
        if not _display_enabled(tf):
            return
        aligned = _align_htf(df, tf)

        high = aligned["high"].shift(1)
        high2 = aligned["high"].shift(3)
        low = aligned["low"].shift(1)
        low2 = aligned["low"].shift(3)
        close1 = aligned["close"].shift(2)
        open1 = aligned["open"].shift(2)

        for idx in range(len(df)):
            if idx < 3:
                continue
            if not _in_session(df.index[idx]):
                continue
            h = float(high.iloc[idx])
            h2 = float(high2.iloc[idx])
            l = float(low.iloc[idx])
            l2 = float(low2.iloc[idx])

            if np.isnan(h) or np.isnan(h2) or np.isnan(l) or np.isnan(l2):
                continue

            is_bull = _detect_bull_fvg(fvgmethod, l, h2)
            is_bear = _detect_bear_fvg(fvgmethod, l2, h)

            if is_bull:
                mtf_boxes.append(
                    FVGBox(
                        direction="bull",
                        timeframe=tf,
                        left_index=idx,
                        right_index=idx,
                        top=l,
                        bottom=h2,
                        label=_timeframe_label(tf) if mtf_settings.show_labels else None,
                        label_time=df.index[idx] if mtf_settings.show_time_on_labels else None,
                    )
                )
            if is_bear:
                mtf_boxes.append(
                    FVGBox(
                        direction="bear",
                        timeframe=tf,
                        left_index=idx,
                        right_index=idx,
                        top=l2,
                        bottom=h,
                        label=_timeframe_label(tf) if mtf_settings.show_labels else None,
                        label_time=df.index[idx] if mtf_settings.show_time_on_labels else None,
                    )
                )

        # Update/mitigation pass (mirrors bar-by-bar updates)
        last_low = df["low"].shift(1)
        last_high = df["high"].shift(1)
        for idx in range(len(df)):
            low_now = float(df["low"].iloc[idx])
            high_now = float(df["high"].iloc[idx])
            close_now = float(df["close"].iloc[idx])
            last_low_now = float(last_low.iloc[idx]) if idx > 0 else low_now
            last_high_now = float(last_high.iloc[idx]) if idx > 0 else high_now

            for box in mtf_boxes:
                if box.timeframe != tf:
                    continue
                if box.right_index < idx:
                    box.right_index = idx

                if box.direction == "bull":
                    threshold = _calc_intrusion_threshold(box.top, box.bottom, mtf_settings.incursion_pct)
                    box.intrusion = low_now < threshold and last_low_now > threshold
                    box.entered = low_now < box.top
                    if mitigationaction == 2:
                        if mtf_settings.use_body_for_mitigation and close_now < box.top:
                            box.top = close_now
                        elif low_now < box.top:
                            box.top = low_now
                    if mitigationaction == 3 and mtf_settings.use_body_for_mitigation and close_now < box.bottom:
                        box.mitigated = True
                    elif mitigationaction == 3 and low_now < box.bottom:
                        box.mitigated = True
                    elif mitigationaction != 3 and mtf_settings.use_body_for_mitigation and close_now < box.bottom:
                        box.mitigated = True
                    elif mitigationaction != 3 and low_now < box.bottom:
                        box.mitigated = True
                else:
                    threshold = _calc_bear_intrusion_threshold(box.top, box.bottom, mtf_settings.incursion_pct)
                    box.intrusion = high_now > threshold and last_high_now < threshold
                    box.entered = high_now > box.bottom
                    if mitigationaction == 2:
                        if mtf_settings.use_body_for_mitigation and close_now > box.bottom:
                            box.bottom = close_now
                        elif high_now > box.bottom:
                            box.bottom = high_now
                    if mitigationaction == 3 and mtf_settings.use_body_for_mitigation and close_now > box.top:
                        box.mitigated = True
                    elif mitigationaction == 3 and high_now > box.top:
                        box.mitigated = True
                    elif mitigationaction != 3 and mtf_settings.use_body_for_mitigation and close_now > box.top:
                        box.mitigated = True
                    elif mitigationaction != 3 and high_now > box.top:
                        box.mitigated = True

        # Enforce max per timeframe
        max_allowed = mtf_settings.max_per_tf.get(tf, 8)
        if max_allowed > 0:
            tf_boxes = [b for b in mtf_boxes if b.timeframe == tf]
            if len(tf_boxes) > max_allowed:
                remove = len(tf_boxes) - max_allowed
                kept = tf_boxes[remove:]
                mtf_boxes[:] = [b for b in mtf_boxes if b.timeframe != tf] + kept

    if mtf_settings.mtf_imb:
        process_tf("5", mtf_settings.enable_5min)
        process_tf("10", mtf_settings.enable_10min)
        process_tf("15", mtf_settings.enable_15min)
        process_tf("30", mtf_settings.enable_30min)
        process_tf("60", mtf_settings.enable_1hr)
        process_tf("240", mtf_settings.enable_4hr)
        process_tf("480", mtf_settings.enable_8hr)
        process_tf("720", mtf_settings.enable_12hr)
        process_tf("D", mtf_settings.enable_daily)
        process_tf("W", mtf_settings.enable_weekly)
        process_tf("M", mtf_settings.enable_monthly)
        if _current_tf_enabled():
            process_tf(current_timeframe, True)

    # Overlay (price overlay) section
    overlay_boxes: List[OverlayBox] = []
    if not mtf_settings.mtf_price_overlay:
        return {"mtf_boxes": mtf_boxes, "overlay_boxes": overlay_boxes, "error": error}

    htf = _align_htf(df, overlay_settings.timeframe)
    htf_time = htf.index
    new_htf_bar = htf_time.to_series().diff().dt.total_seconds().fillna(0) != 0

    atr = htf["high"].sub(htf["low"]).rolling(overlay_settings.atrlength).mean()

    last_bar_date = pd.Timestamp.now().normalize()
    days_left = (last_bar_date - df.index).days
    in_range = days_left < overlay_settings.days_back if overlay_settings.lookback else True

    for idx in range(len(df)):
        if not new_htf_bar.iloc[idx]:
            continue

        o0 = float(htf["open"].iloc[idx])
        h0 = float(htf["high"].iloc[idx])
        l0 = float(htf["low"].iloc[idx])
        c0 = float(htf["close"].iloc[idx])
        if idx < 2:
            continue
        o1 = float(htf["open"].iloc[idx - 1])
        h1 = float(htf["high"].iloc[idx - 1])
        l1 = float(htf["low"].iloc[idx - 1])
        c1 = float(htf["close"].iloc[idx - 1])
        o2 = float(htf["open"].iloc[idx - 2])
        h2 = float(htf["high"].iloc[idx - 2])
        l2 = float(htf["low"].iloc[idx - 2])
        c2 = float(htf["close"].iloc[idx - 2])

        upimbdist = (l0 - h2) / h2 * 100
        downimbdist = (l2 - h0) / h0 * 100
        upgapdist = (l0 - h1) / h1 * 100
        downgapdist = (l1 - h0) / h0 * 100
        bodysize = abs(o1 - c1)
        upper_wick = h1 - max(o1, c1)
        lower_wick = min(o1, c1) - l1

        if overlay_settings.conditiontype == "Percentage":
            cond_imb_up = upimbdist > (overlay_settings.pctcond * overlay_settings.pctmult)
            cond_imb_dn = downimbdist > (overlay_settings.pctcond * overlay_settings.pctmult)
            cond_gap_up = upgapdist > (overlay_settings.pctcond * overlay_settings.pctmult)
            cond_gap_dn = downgapdist > (overlay_settings.pctcond * overlay_settings.pctmult)
            cond_wick_up = (upper_wick / max(o1, c1) * 100) > (overlay_settings.pctcond * overlay_settings.pctmult)
            cond_wick_dn = (lower_wick / l1 * 100) > (overlay_settings.pctcond * overlay_settings.pctmult)
        elif overlay_settings.conditiontype == "ATR":
            atr_val = float(atr.iloc[idx]) if not np.isnan(atr.iloc[idx]) else 0
            cond_imb_up = (l0 - h2) > overlay_settings.atrmult * atr_val
            cond_imb_dn = (l2 - h0) > overlay_settings.atrmult * atr_val
            cond_gap_up = (l0 - h1) > overlay_settings.atrmult * atr_val
            cond_gap_dn = (l1 - h0) > overlay_settings.atrmult * atr_val
            cond_wick_up = upper_wick > overlay_settings.atrmult * atr_val
            cond_wick_dn = lower_wick > overlay_settings.atrmult * atr_val
        else:
            cond_imb_up = cond_imb_dn = True
            cond_gap_up = cond_gap_dn = True
            cond_wick_up = cond_wick_dn = True

        if not in_range[idx]:
            continue

        time_now = df.index[idx]
        if overlay_settings.boxtype == "Imbalance":
            if l0 > h2 and overlay_settings.showup and cond_imb_up:
                overlay_boxes.append(
                    OverlayBox(
                        direction="up",
                        box_type="Imbalance",
                        left_time=df.index[idx - 2],
                        right_time=time_now,
                        top=l0,
                        bottom=h2,
                        label=_timeframe_label(overlay_settings.timeframe),
                    )
                )
            if h0 < l2 and overlay_settings.showdown and cond_imb_dn:
                overlay_boxes.append(
                    OverlayBox(
                        direction="down",
                        box_type="Imbalance",
                        left_time=df.index[idx - 2],
                        right_time=time_now,
                        top=l2,
                        bottom=h0,
                        label=_timeframe_label(overlay_settings.timeframe),
                    )
                )

        if overlay_settings.boxtype == "Gap":
            if l0 > h1 and overlay_settings.showup and cond_gap_up:
                overlay_boxes.append(
                    OverlayBox(
                        direction="up",
                        box_type="Gap",
                        left_time=df.index[idx - 1],
                        right_time=time_now,
                        top=l0,
                        bottom=h1,
                        label=_timeframe_label(overlay_settings.timeframe),
                    )
                )
            if h0 < l1 and overlay_settings.showdown and cond_gap_dn:
                overlay_boxes.append(
                    OverlayBox(
                        direction="down",
                        box_type="Gap",
                        left_time=df.index[idx - 1],
                        right_time=time_now,
                        top=l1,
                        bottom=h0,
                        label=_timeframe_label(overlay_settings.timeframe),
                    )
                )

        if overlay_settings.boxtype == "Wick":
            if upper_wick > (bodysize / 6) and overlay_settings.showup and cond_wick_up:
                overlay_boxes.append(
                    OverlayBox(
                        direction="up",
                        box_type="Wick",
                        left_time=df.index[idx - 1],
                        right_time=time_now,
                        top=h1,
                        bottom=max(o1, c1),
                        label=_timeframe_label(overlay_settings.timeframe),
                    )
                )
            if lower_wick > (bodysize / 6) and overlay_settings.showdown and cond_wick_dn:
                overlay_boxes.append(
                    OverlayBox(
                        direction="down",
                        box_type="Wick",
                        left_time=df.index[idx - 1],
                        right_time=time_now,
                        top=min(o1, c1),
                        bottom=l1,
                        label=_timeframe_label(overlay_settings.timeframe),
                    )
                )

    # Fill/extend logic
    for idx in range(len(df)):
        high_now = float(df["high"].iloc[idx])
        low_now = float(df["low"].iloc[idx])
        for box in list(overlay_boxes):
            level = box.bottom
            level2 = box.top
            level3 = (level2 + level) / 2
            if overlay_settings.filledtype == "Touch":
                filled = (high_now > level and low_now < level) or (high_now > level2 and low_now < level2)
            elif overlay_settings.filledtype == "Half Fill":
                filled = high_now > level3 and low_now < level3
            else:
                filled = (high_now > level2 and low_now < level2) if box.direction == "up" else (
                    high_now > level and low_now < level
                )
            if filled and overlay_settings.hidefilled:
                overlay_boxes.remove(box)
                continue
            if filled and overlay_settings.extendtilfilled:
                overlay_boxes.remove(box)
                continue
            if not filled and not overlay_settings.extendtilfilled:
                overlay_boxes.remove(box)
                continue
            box.right_time = df.index[idx]

    while len(overlay_boxes) >= overlay_settings.max_boxes:
        overlay_boxes.pop(0)

    return {"mtf_boxes": mtf_boxes, "overlay_boxes": overlay_boxes, "error": error}
