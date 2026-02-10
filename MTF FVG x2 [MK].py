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
    color_state: str = "default"


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
    visible: bool = True


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
    incursion_alerts: bool = True
    # Pine uses mitig_type = "Wicks" by default.
    use_body_for_mitigation: bool = False
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
    actionbool: bool = False
    actiondel: str = "Stop Zone"
    nmbars: int = 3500
    max_boxes: int = 499


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def _tf_to_rule(tf: str) -> str:
    # Pine `request.security` accepts arbitrary minute strings for intraday timeframes.
    if tf.isdigit():
        return f"{int(tf)}min"
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
        "60": "1hr",
        "240": "4hr",
        "480": "8hr",
        "720": "12hr",
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


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def compute_mtf_fvg_x2(
    df: pd.DataFrame,
    *,
    mtf_settings: Optional[MTFSettings] = None,
    overlay_settings: Optional[OverlaySettings] = None,
    current_timeframe: Optional[str] = None,
    timeframe_minutes: Optional[int] = None,
    is_dwm: bool = False,
) -> Dict[str, List]:
    """Compute MTF FVG x2 outputs with Pine-like bar-by-bar ordering."""
    if mtf_settings is None:
        mtf_settings = MTFSettings()
    if overlay_settings is None:
        overlay_settings = OverlaySettings()

    df = _prepare_ohlcv(df)

    fvgmethod = True
    mitigationaction = _mitigation_mode(mtf_settings.mitigation_mode)

    total_allowed = (
        mtf_settings.max_per_tf.get("chart", 0)
        + mtf_settings.max_per_tf.get("5", 0)
        + mtf_settings.max_per_tf.get("10", 0)
        + mtf_settings.max_per_tf.get("15", 0)
        + mtf_settings.max_per_tf.get("60", 0)
        + mtf_settings.max_per_tf.get("240", 0)
        + mtf_settings.max_per_tf.get("480", 0)
        + mtf_settings.max_per_tf.get("D", 0)
        + mtf_settings.max_per_tf.get("W", 0)
        + mtf_settings.max_per_tf.get("M", 0)
    )
    error = total_allowed > 500
    alerts: List[str] = []
    bull_fvg_creation_alert = False
    bear_fvg_creation_alert = False

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
        start_hhmm, end_hhmm = mtf_settings.session_hours.split("-")
        start_h, start_m = int(start_hhmm[:2]), int(start_hhmm[2:])
        end_h, end_m = int(end_hhmm[:2]), int(end_hhmm[2:])
        return (start_h, start_m) <= (ts.hour, ts.minute) <= (end_h, end_m)

    enabled_tfs: List[str] = []
    if mtf_settings.mtf_imb:
        tf_flags = [
            ("5", mtf_settings.enable_5min),
            ("10", mtf_settings.enable_10min),
            ("15", mtf_settings.enable_15min),
            ("30", mtf_settings.enable_30min),
            ("60", mtf_settings.enable_1hr),
            ("240", mtf_settings.enable_4hr),
            ("480", mtf_settings.enable_8hr),
            ("720", mtf_settings.enable_12hr),
            ("D", mtf_settings.enable_daily),
            ("W", mtf_settings.enable_weekly),
            ("M", mtf_settings.enable_monthly),
        ]
        for tf, enabled in tf_flags:
            if enabled and _display_enabled(tf):
                enabled_tfs.append(tf)
        if _current_tf_enabled() and current_timeframe is not None and _display_enabled(current_timeframe):
            enabled_tfs.append(current_timeframe)

    tf_data: Dict[str, Dict[str, pd.Series]] = {}
    for tf in enabled_tfs:
        aligned = _align_htf(df, tf)
        tf_data[tf] = {
            "high": aligned["high"].shift(1),
            "high2": aligned["high"].shift(3),
            "low": aligned["low"].shift(1),
            "low2": aligned["low"].shift(3),
        }

    bull_boxes: Dict[str, List[FVGBox]] = {tf: [] for tf in enabled_tfs}
    bear_boxes: Dict[str, List[FVGBox]] = {tf: [] for tf in enabled_tfs}
    last_bar_index = len(df) - 1

    # Pine oddity at line ~648: 12hr call uses the 8hr bear array pointer.
    # Mirror this unconditionally when 12hr is enabled (8hr array exists in Pine even if disabled).
    bear_store_tf: Dict[str, str] = {tf: tf for tf in enabled_tfs}
    if "720" in enabled_tfs:
        bear_store_tf["720"] = "480"
        if "480" not in bear_boxes:
            bear_boxes["480"] = []

    for idx in range(len(df)):
        ts = df.index[idx]
        low_now = float(df["low"].iloc[idx])
        high_now = float(df["high"].iloc[idx])
        close_now = float(df["close"].iloc[idx])
        prev_low = float(df["low"].iloc[idx - 1]) if idx > 0 else low_now
        prev_high = float(df["high"].iloc[idx - 1]) if idx > 0 else high_now

        for tf in enabled_tfs:
            series = tf_data[tf]
            h = float(series["high"].iloc[idx]) if idx < len(series["high"]) else np.nan
            h2 = float(series["high2"].iloc[idx]) if idx < len(series["high2"]) else np.nan
            l = float(series["low"].iloc[idx]) if idx < len(series["low"]) else np.nan
            l2 = float(series["low2"].iloc[idx]) if idx < len(series["low2"]) else np.nan

            new_bull = False
            new_bear = False
            if idx >= 3 and (not mtf_settings.only_market_hours or _in_session(ts)):
                if not (np.isnan(h) or np.isnan(h2) or np.isnan(l) or np.isnan(l2)):
                    new_bull = _detect_bull_fvg(fvgmethod, l, h2)
                    new_bear = _detect_bear_fvg(fvgmethod, l2, h)

            max_allowed = mtf_settings.max_per_tf.get(tf, 8)
            tf_bulls = bull_boxes[tf]
            tf_bears = bear_boxes[bear_store_tf[tf]]

            if new_bull:
                if len(tf_bulls) > max_allowed and tf_bulls:
                    tf_bulls.pop(0)
                prev_h2 = float(series["high2"].iloc[idx - 1]) if idx > 0 else np.nan
                prev_l = float(series["low"].iloc[idx - 1]) if idx > 0 else np.nan
                if not (np.isnan(prev_h2) or np.isnan(prev_l)) and (h2 != prev_h2 and l != prev_l):
                    tf_bulls.append(
                        FVGBox(
                            direction="bull",
                            timeframe=tf,
                            left_index=last_bar_index + 5,
                            right_index=last_bar_index + 15,
                            top=l,
                            bottom=h2,
                            label=_timeframe_label(tf) if mtf_settings.show_labels else None,
                            label_time=ts if mtf_settings.show_time_on_labels else None,
                        )
                    )
                    alerts.append(f"Bull FVG Creation {tf}")
                    bull_fvg_creation_alert = True

            if new_bear:
                if len(tf_bears) > max_allowed and tf_bears:
                    tf_bears.pop(0)
                prev_l2 = float(series["low2"].iloc[idx - 1]) if idx > 0 else np.nan
                prev_h = float(series["high"].iloc[idx - 1]) if idx > 0 else np.nan
                if not (np.isnan(prev_l2) or np.isnan(prev_h)) and (l2 != prev_l2 and h != prev_h):
                    tf_bears.append(
                        FVGBox(
                            direction="bear",
                            timeframe=tf,
                            left_index=last_bar_index + 5,
                            right_index=last_bar_index + 15,
                            top=l2,
                            bottom=h,
                            label=_timeframe_label(tf) if mtf_settings.show_labels else None,
                            label_time=ts if mtf_settings.show_time_on_labels else None,
                        )
                    )
                    alerts.append(f"Bear FVG Creation {tf}")
                    bear_fvg_creation_alert = True

            for bi in range(len(tf_bulls) - 1, -1, -1):
                box = tf_bulls[bi]
                if mtf_settings.show_labels:
                    box.left_index = last_bar_index + mtf_settings.label_shift
                    box.right_index = last_bar_index + mtf_settings.label_shiftr

                threshold = _calc_intrusion_threshold(box.top, box.bottom, mtf_settings.incursion_pct)
                low_under_top = low_now < box.top
                low_under_bottom = low_now < box.bottom
                close_under_top = close_now < box.top
                close_under_bottom = close_now < box.bottom
                low_under_mid = low_now < ((box.top + box.bottom) / 2)
                close_under_mid = close_now < ((box.top + box.bottom) / 2)
                box.intrusion = low_now < threshold and prev_low > threshold
                box.entered = low_under_top
                if (mitigationaction in {1, 3}) and box.intrusion and mtf_settings.incursion_alerts:
                    alerts.append(f"Bull FVG Incursion {_timeframe_label(tf)}")
                if mtf_settings.entry_change_color:
                    box.color_state = "entry_bull" if low_under_top else "default"

                if mitigationaction == 2 and mtf_settings.use_body_for_mitigation and close_under_top:
                    box.top = close_now
                elif mitigationaction == 2 and low_under_top:
                    box.top = low_now

                if mitigationaction == 3 and ((mtf_settings.use_body_for_mitigation and close_under_bottom) or low_under_bottom):
                    box.mitigated = True
                    box.color_state = "mitigated_none"

                if mitigationaction in {1, 2} and ((mtf_settings.use_body_for_mitigation and close_under_bottom) or low_under_bottom):
                    tf_bulls.pop(bi)
                    continue

                if mitigationaction == 4 and ((mtf_settings.use_body_for_mitigation and close_under_mid) or low_under_mid):
                    tf_bulls.pop(bi)
                    continue

            for bi in range(len(tf_bears) - 1, -1, -1):
                box = tf_bears[bi]
                if mtf_settings.show_labels:
                    box.left_index = last_bar_index + mtf_settings.label_shift
                    box.right_index = last_bar_index + mtf_settings.label_shiftr

                threshold = _calc_bear_intrusion_threshold(box.top, box.bottom, mtf_settings.incursion_pct)
                high_over_top = high_now > box.top
                high_over_bottom = high_now > box.bottom
                close_over_top = close_now > box.top
                close_over_bottom = close_now > box.bottom
                high_over_mid = high_now > ((box.top + box.bottom) / 2)
                close_over_mid = close_now > ((box.top + box.bottom) / 2)
                box.intrusion = high_now > threshold and prev_high < threshold
                box.entered = high_over_bottom
                if (mitigationaction in {1, 3}) and box.intrusion and mtf_settings.incursion_alerts:
                    alerts.append(f"Bear FVG Incursion {_timeframe_label(tf)}")
                if mtf_settings.entry_change_color:
                    box.color_state = "entry_bear" if high_over_bottom else "default"

                if mitigationaction == 2 and mtf_settings.use_body_for_mitigation and close_over_bottom:
                    box.bottom = close_now
                elif mitigationaction == 2 and high_over_bottom:
                    box.bottom = high_now

                if mitigationaction == 3 and ((mtf_settings.use_body_for_mitigation and close_over_top) or high_over_top):
                    box.mitigated = True
                    box.color_state = "mitigated_none"

                if mitigationaction in {1, 2} and ((mtf_settings.use_body_for_mitigation and close_over_top) or high_over_top):
                    tf_bears.pop(bi)
                    continue

                if mitigationaction == 4 and ((mtf_settings.use_body_for_mitigation and close_over_mid) or high_over_mid):
                    tf_bears.pop(bi)
                    continue

    mtf_boxes: List[FVGBox] = []
    for tf in enabled_tfs:
        mtf_boxes.extend(bull_boxes[tf])

    # Include bear arrays by their actual storage pointer keys (Pine 12hr->8hr quirk).
    bear_output_keys = list(dict.fromkeys(bear_store_tf.get(tf, tf) for tf in enabled_tfs))
    for tf in bear_output_keys:
        mtf_boxes.extend(bear_boxes.get(tf, []))

    overlay_boxes: List[OverlayBox] = []

    htf_rule = _tf_to_rule(overlay_settings.timeframe)
    htf_raw = _resample_ohlc(df, htf_rule)
    htf = htf_raw.reindex(df.index, method="ffill")
    htf_anchor = df.index.floor(htf_rule)
    new_htf_bar = htf_anchor.to_series().diff().ne(pd.Timedelta(0)).fillna(False)

    # Closer to request.security(ta.atr()) by computing ATR on raw HTF then ffill to chart bars.
    prev_close_htf = htf_raw["close"].shift(1)
    tr_raw = pd.concat(
        [
            htf_raw["high"] - htf_raw["low"],
            (htf_raw["high"] - prev_close_htf).abs(),
            (htf_raw["low"] - prev_close_htf).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_raw = tr_raw.rolling(overlay_settings.atrlength).mean()
    atr = atr_raw.reindex(df.index, method="ffill")

    now_tz = df.index.tz if hasattr(df.index, "tz") else None
    last_bar_date = pd.Timestamp.now(tz=now_tz)
    days_left = np.abs(np.floor((last_bar_date - df.index).total_seconds() / (24 * 60 * 60)))
    in_range = days_left < overlay_settings.days_back if overlay_settings.lookback else np.ones(len(df), dtype=bool)

    tf_same_as_chart = current_timeframe is not None and overlay_settings.timeframe == current_timeframe

    def _overlay_text(box_type: str, idx: int) -> Optional[str]:
        if not overlay_settings.showboxtext:
            return None
        volume_series = htf["volume"] if "volume" in htf.columns else pd.Series(0, index=df.index)
        if box_type in {"Imbalance", "Wick"}:
            vol_idx = idx - 1 if tf_same_as_chart else idx - 2
        else:
            vol_idx = idx if tf_same_as_chart else idx - 1
        vol_val = float(volume_series.iloc[vol_idx]) if vol_idx >= 0 else 0.0
        imbtext, gaptext, wicktext = "", "GAP", "WICK"
        base = {"Imbalance": imbtext, "Gap": gaptext, "Wick": wicktext}[box_type]
        tf_lbl = _timeframe_label(overlay_settings.timeframe)
        if overlay_settings.text_type == "Labels":
            return base
        if overlay_settings.text_type == "Volume":
            return f"{vol_val:.0f}"
        if overlay_settings.text_type == "Labels + Timeframe + Volume":
            return f"{base} • {tf_lbl} • {vol_val:.0f}"
        return f"{base} • {tf_lbl}"

    for idx in range(len(df)):
        if not bool(new_htf_bar.iloc[idx]):
            continue
        if idx < 2:
            continue

        o0 = float(htf["open"].iloc[idx])
        h0 = float(htf["high"].iloc[idx])
        l0 = float(htf["low"].iloc[idx])
        c0 = float(htf["close"].iloc[idx])
        o1 = float(htf["open"].iloc[idx - 1])
        h1 = float(htf["high"].iloc[idx - 1])
        l1 = float(htf["low"].iloc[idx - 1])
        c1 = float(htf["close"].iloc[idx - 1])
        o2 = float(htf["open"].iloc[idx - 2])
        h2 = float(htf["high"].iloc[idx - 2])
        l2 = float(htf["low"].iloc[idx - 2])
        c2 = float(htf["close"].iloc[idx - 2])
        _ = (o0, c0, o2, c2)

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
            atr_val = float(atr.iloc[idx]) if not np.isnan(atr.iloc[idx]) else 0.0
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

        if not bool(in_range[idx]):
            continue

        time_now = df.index[idx]
        if overlay_settings.boxtype == "Imbalance":
            if l0 > h2 and overlay_settings.showup and cond_imb_up and mtf_settings.mtf_price_overlay:
                overlay_boxes.append(
                    OverlayBox(
                        direction="up",
                        box_type="Imbalance",
                        left_time=df.index[idx - 2],
                        right_time=time_now,
                        top=l0,
                        bottom=h2,
                        label=_overlay_text("Imbalance", idx),
                        visible=overlay_settings.showboxes,
                    )
                )
            if h0 < l2 and overlay_settings.showdown and cond_imb_dn and mtf_settings.mtf_price_overlay:
                overlay_boxes.append(
                    OverlayBox(
                        direction="down",
                        box_type="Imbalance",
                        left_time=df.index[idx - 2],
                        right_time=time_now,
                        top=l2,
                        bottom=h0,
                        label=_overlay_text("Imbalance", idx),
                        visible=overlay_settings.showboxes,
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
                        label=_overlay_text("Gap", idx),
                        visible=overlay_settings.showboxes,
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
                        label=_overlay_text("Gap", idx),
                        visible=overlay_settings.showboxes,
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
                        label=_overlay_text("Wick", idx),
                        visible=overlay_settings.showboxes,
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
                        label=_overlay_text("Wick", idx),
                        visible=overlay_settings.showboxes,
                    )
                )

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
                filled = high_now > level and low_now < level

            if filled and overlay_settings.hidefilled:
                overlay_boxes.remove(box)
                continue
            if filled and not overlay_settings.hidefilled:
                box.filled = True
            if filled and overlay_settings.extendtilfilled:
                overlay_boxes.remove(box)
                continue

            box.right_time = df.index[idx]

            if (not filled) and (not overlay_settings.extendtilfilled):
                overlay_boxes.remove(box)
                continue

            if (not filled) and overlay_settings.actionbool:
                span_bars = idx - df.index.get_loc(box.left_time) if box.left_time in df.index else 0
                if overlay_settings.actiondel == "Delete Zone" and span_bars > overlay_settings.nmbars:
                    overlay_boxes.remove(box)
                    continue
                if overlay_settings.actiondel == "Stop Zone" and span_bars > overlay_settings.nmbars:
                    overlay_boxes.remove(box)
                    continue

    while len(overlay_boxes) >= overlay_settings.max_boxes:
        overlay_boxes.pop(0)

    return {
        "mtf_boxes": mtf_boxes,
        "overlay_boxes": overlay_boxes,
        "error": error,
        "alerts": alerts,
        "bull_fvg_creation_alert": bull_fvg_creation_alert,
        "bear_fvg_creation_alert": bear_fvg_creation_alert,
        "both_fvg_creation_alert": bull_fvg_creation_alert or bear_fvg_creation_alert,
    }

