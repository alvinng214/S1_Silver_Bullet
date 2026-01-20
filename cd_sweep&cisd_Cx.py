"""
cd_sweep&cisd_Cx - Python Translation

Detects Higher Timeframe (HTF) sweeps and CISD (Change in State of Delivery)
signals. Mirrors the Pine Script logic by cdikici71, including:
- HTF candle tracking (current + previous, sweep flags)
- CISD level detection with pattern lookbacks
- CISD cross triggers gated by sweep conditions
- Bias calculation on a separate HTF
- SMT divergence checks against a correlated symbol
- Key level tracking on 3 configurable timeframes
- Asset screener series (xbull/xbear per symbol)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HTFCandle:
    start_idx: int
    end_idx: int
    open: float
    high: float
    low: float
    close: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


@dataclass
class SweepBox:
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_high_sweep: bool


@dataclass
class CISDLevel:
    idx: int
    price: float
    is_bullish: bool


@dataclass
class CISDSignal:
    idx: int
    cisd_start_idx: int
    cisd_price: float
    is_bullish: bool
    is_sweep_confirmed: bool


@dataclass
class SMTSignal:
    idx: int
    is_low_smt: bool
    is_high_smt: bool


@dataclass
class BiasState:
    idx: int
    bias: int


@dataclass
class KeyLevelState:
    idx: int
    tf_minutes: int
    prev_high: float
    prev_low: float
    prev_open: float
    curr_high: float
    curr_low: float
    curr_open: float
    swept_high: bool
    swept_low: bool


@dataclass
class RemainingTimeState:
    idx: int
    tf_minutes: int
    remaining_seconds: Optional[int]


@dataclass
class AssetSignalState:
    symbol: str
    xbull: List[bool]
    xbear: List[bool]


def _to_minutes(tf: Optional[str | int]) -> Optional[int]:
    if tf is None:
        return None
    if isinstance(tf, int):
        return tf
    if isinstance(tf, float):
        return int(tf)
    tf = tf.strip()
    if tf.upper() in {"D", "1D"}:
        return 1440
    if tf.upper() in {"W", "1W"}:
        return 10080
    if tf.upper() in {"M", "1M"}:
        return 43200
    if tf.upper() == "3M":
        return 129600
    if tf.isdigit():
        return int(tf)
    raise ValueError(f"Unsupported timeframe: {tf}")


def _tf_text(tf_minutes: int) -> str:
    if tf_minutes >= 43200:
        return "M"
    if tf_minutes >= 10080:
        return "W"
    if tf_minutes >= 1440:
        return "D"
    if tf_minutes >= 60:
        return f"h{tf_minutes // 60}"
    return f"{tf_minutes}m"


def _auto_htf(ltf_minutes: int) -> int:
    if ltf_minutes == 1:
        return 15
    if ltf_minutes == 2:
        return 60
    if ltf_minutes == 3:
        return 30
    if ltf_minutes == 5:
        return 60
    if ltf_minutes == 15:
        return 240
    if ltf_minutes == 30:
        return 720
    if ltf_minutes == 60:
        return 1440
    if ltf_minutes == 240:
        return 10080
    if ltf_minutes == 1440:
        return 43200
    if ltf_minutes == 10080:
        return 129600
    return ltf_minutes


def _build_period_map(index: pd.DatetimeIndex, tf_minutes: int) -> Tuple[List[dict], np.ndarray]:
    freq = f"{tf_minutes}min"
    temp = pd.DataFrame({"idx": np.arange(len(index))}, index=index)
    period_map: List[dict] = []
    period_idx = np.full(len(index), -1, dtype=int)
    for pid, (start_time, group) in enumerate(temp.resample(freq, label="left", closed="left")):
        if group.empty:
            continue
        start = int(group["idx"].iloc[0])
        end = int(group["idx"].iloc[-1])
        period_map.append(
            {
                "start": start,
                "end": end,
                "start_time": start_time,
                "end_time": index[end],
            }
        )
        period_idx[start : end + 1] = pid
    return period_map, period_idx


def _remaining_seconds_for_bar(
    index: pd.DatetimeIndex, tf_minutes: int, bar_idx: int, period_map: List[dict]
) -> Optional[int]:
    if bar_idx < 0 or bar_idx >= len(index):
        return None
    period_idx = next((i for i, p in enumerate(period_map) if p["start"] <= bar_idx <= p["end"]), None)
    if period_idx is None:
        return None
    period = period_map[period_idx]
    period_start = period["start_time"]
    period_end = period_start + pd.Timedelta(minutes=tf_minutes)
    return int((period_end - index[bar_idx]).total_seconds())


def _compute_series(
    df: pd.DataFrame,
    htf_minutes: int,
    capture_details: bool = True,
) -> Dict[str, object]:
    n = len(df)
    htf_map, htf_idx = _build_period_map(df.index, htf_minutes)

    htf_candles: List[HTFCandle] = []
    sweep_boxes: List[SweepBox] = []
    cisd_levels: List[CISDLevel] = []
    cisd_signals: List[CISDSignal] = []

    h_swept_series = [False] * n
    l_swept_series = [False] * n
    xbull_series = [False] * n
    xbear_series = [False] * n
    weak_bull_cross = [False] * n
    weak_bear_cross = [False] * n

    o0 = h0 = l0 = c0 = None
    o1 = h1 = l1 = c1 = None
    t0_idx = t1_idx = None
    h_swept1 = l_swept1 = False

    bull_level = np.inf
    bear_level = -np.inf
    bull_index = -1
    bear_index = -1
    xcisd = False
    ycisd = False

    h0_series = [np.nan] * n
    l0_series = [np.nan] * n
    h1_series = [np.nan] * n
    l1_series = [np.nan] * n

    last_htf_idx = None

    for i in range(n):
        row = df.iloc[i]
        current_htf_idx = htf_idx[i]
        new_htf = current_htf_idx != last_htf_idx and current_htf_idx >= 0

        if new_htf:
            if t0_idx is not None:
                o1, h1, l1, c1 = o0, h0, l0, c0
                t1_idx = t0_idx
                if capture_details and last_htf_idx is not None:
                    period = htf_map[last_htf_idx]
                    htf_candles.append(
                        HTFCandle(
                            start_idx=period["start"],
                            end_idx=period["end"],
                            open=o1,
                            high=h1,
                            low=l1,
                            close=c1,
                            start_time=df.index[period["start"]],
                            end_time=df.index[period["end"]],
                        )
                    )
                if i > 0:
                    h_swept1 = h_swept_series[i - 1]
                    l_swept1 = l_swept_series[i - 1]

            t0_idx = i
            o0 = row["open"]
            h0 = row["high"]
            l0 = row["low"]
            c0 = row["close"]
        else:
            if h0 is None:
                o0 = row["open"]
                h0 = row["high"]
                l0 = row["low"]
                c0 = row["close"]
            if row["high"] >= h0:
                h0 = row["high"]
            if row["low"] <= l0:
                l0 = row["low"]
            c0 = row["close"]

        h0_series[i] = h0
        l0_series[i] = l0
        h1_series[i] = h1 if h1 is not None else np.nan
        l1_series[i] = l1 if l1 is not None else np.nan

        h_swept = False
        l_swept = False
        if h1 is not None and l1 is not None:
            h_swept = h0 > h1 and max(o0, c0) < h1
            l_swept = l0 < l1 and min(o0, c0) > l1

        h_swept_series[i] = h_swept
        l_swept_series[i] = l_swept

        if capture_details and t0_idx is not None:
            if h_swept and row["open"] < h1:
                sweep_boxes.append(
                    SweepBox(
                        start_idx=t0_idx,
                        end_idx=i,
                        top=h0,
                        bottom=h1,
                        is_high_sweep=True,
                    )
                )
            if l_swept and row["open"] > l1:
                sweep_boxes.append(
                    SweepBox(
                        start_idx=t0_idx,
                        end_idx=i,
                        top=l1,
                        bottom=l0,
                        is_high_sweep=False,
                    )
                )

        if h1 is not None and l1 is not None:
            up = row["close"] > row["open"]
            dw = row["close"] < row["open"]
            eq = row["close"] == row["open"]

            if row["low"] == l0 and row["low"] < l1:
                if i > 0:
                    prev = df.iloc[i - 1]
                    prev_up = prev["close"] > prev["open"]
                    prev_eq = prev["close"] == prev["open"]
                    if (dw or eq) and (prev_up or prev_eq) and not (eq and prev_eq):
                        bull_level = row["open"]
                        bull_index = i
                        if capture_details:
                            cisd_levels.append(CISDLevel(idx=i, price=bull_level, is_bullish=True))
                    else:
                        for lookback in range(2, min(11, i + 1)):
                            if df.iloc[i - lookback]["low"] < row["low"]:
                                break
                            prev_j = df.iloc[i - lookback]
                            prev_j1 = df.iloc[i - lookback + 1]
                            up_j = prev_j["close"] > prev_j["open"]
                            eq_j = prev_j["close"] == prev_j["open"]
                            dw_j1 = prev_j1["close"] < prev_j1["open"]
                            if (up_j or eq_j) and dw_j1:
                                bar = lookback - 1
                                bull_level = df.iloc[i - bar]["open"]
                                bull_index = i - bar
                                for k in range(bar, -1, -1):
                                    candle_k = df.iloc[i - k]
                                    dw_k = candle_k["close"] < candle_k["open"]
                                    if candle_k["open"] > bull_level and dw_k:
                                        bull_level = candle_k["open"]
                                        bull_index = i - k
                                if bull_level < row["open"] and not up:
                                    bull_level = row["open"]
                                    bull_index = i
                                if bull_level < row["open"] and up:
                                    bull_level = row["high"]
                                    bull_index = i
                                if capture_details:
                                    cisd_levels.append(
                                        CISDLevel(idx=bull_index, price=bull_level, is_bullish=True)
                                    )
                                break

            if row["high"] == h0 and row["high"] > h1:
                if i > 0:
                    prev = df.iloc[i - 1]
                    prev_dw = prev["close"] < prev["open"]
                    prev_eq = prev["close"] == prev["open"]
                    if (up or eq) and (prev_dw or prev_eq) and not (eq and prev_eq):
                        bear_level = row["open"]
                        bear_index = i
                        if capture_details:
                            cisd_levels.append(CISDLevel(idx=i, price=bear_level, is_bullish=False))
                    else:
                        for lookback in range(2, min(11, i + 1)):
                            if df.iloc[i - lookback]["high"] > row["high"]:
                                break
                            prev_j = df.iloc[i - lookback]
                            prev_j1 = df.iloc[i - lookback + 1]
                            dw_j = prev_j["close"] < prev_j["open"]
                            eq_j = prev_j["close"] == prev_j["open"]
                            up_j1 = prev_j1["close"] > prev_j1["open"]
                            if (dw_j or eq_j) and up_j1:
                                bar = lookback - 1
                                bear_level = df.iloc[i - bar]["open"]
                                bear_index = i - bar
                                for k in range(bar, -1, -1):
                                    candle_k = df.iloc[i - k]
                                    up_k = candle_k["close"] > candle_k["open"]
                                    if candle_k["open"] < bear_level and up_k:
                                        bear_level = candle_k["open"]
                                        bear_index = i - k
                                if bear_level > row["open"] and not dw:
                                    bear_level = row["open"]
                                    bear_index = i
                                if bear_level > row["open"] and dw:
                                    bear_level = row["low"]
                                    bear_index = i
                                if capture_details:
                                    cisd_levels.append(
                                        CISDLevel(idx=bear_index, price=bear_level, is_bullish=False)
                                    )
                                break

            if i > 0:
                if row["high"] >= h0_series[i - 1]:
                    ycisd = False
                if row["low"] <= l0_series[i - 1]:
                    xcisd = False

            if i > 0:
                prev_close = df.iloc[i - 1]["close"]
                bull_sweep_ok = l_swept_series[i - 1] or (l1 <= l0 and l_swept1)
                bear_sweep_ok = h_swept_series[i - 1] or (h1 >= h0 and h_swept1)

                if (
                    prev_close > bull_level
                    and bull_sweep_ok
                    and not xcisd
                    and i - 1 >= bull_index
                    and bull_level < np.inf
                ):
                    xbull_series[i] = True
                    if capture_details:
                        cisd_signals.append(
                            CISDSignal(
                                idx=i - 1,
                                cisd_start_idx=bull_index,
                                cisd_price=bull_level,
                                is_bullish=True,
                                is_sweep_confirmed=True,
                            )
                        )
                    bull_level = np.inf
                    xcisd = True
                elif (
                    prev_close > bull_level
                    and not bull_sweep_ok
                    and not xcisd
                    and i - 1 >= bull_index
                    and bull_level < np.inf
                ):
                    weak_bull_cross[i] = True
                    bull_level = np.inf

                if (
                    prev_close < bear_level
                    and bear_sweep_ok
                    and not ycisd
                    and i - 1 >= bear_index
                    and bear_level > -np.inf
                ):
                    xbear_series[i] = True
                    if capture_details:
                        cisd_signals.append(
                            CISDSignal(
                                idx=i - 1,
                                cisd_start_idx=bear_index,
                                cisd_price=bear_level,
                                is_bullish=False,
                                is_sweep_confirmed=True,
                            )
                        )
                    bear_level = -np.inf
                    ycisd = True
                elif (
                    prev_close < bear_level
                    and not bear_sweep_ok
                    and not ycisd
                    and i - 1 >= bear_index
                    and bear_level > -np.inf
                ):
                    weak_bear_cross[i] = True
                    bear_level = -np.inf

        last_htf_idx = current_htf_idx

    return {
        "htf_candles": htf_candles,
        "sweep_boxes": sweep_boxes,
        "cisd_levels": cisd_levels,
        "cisd_signals": cisd_signals,
        "h_swept_series": h_swept_series,
        "l_swept_series": l_swept_series,
        "xbull_series": xbull_series,
        "xbear_series": xbear_series,
        "weak_bull_cross": weak_bull_cross,
        "weak_bear_cross": weak_bear_cross,
        "htf_map": htf_map,
        "h0_series": h0_series,
        "l0_series": l0_series,
        "h1_series": h1_series,
        "l1_series": l1_series,
    }


def detect_cd_sweep_cisd(
    df: pd.DataFrame,
    htf_minutes: Optional[int | str] = None,
    htf_bias_minutes: Optional[int | str] = None,
    key_tfs: Tuple[int | str, int | str, int | str] = (43200, 10080, 1440),
    smt_reference: Optional[pd.DataFrame] = None,
    assets: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, object]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    if len(df) < 2:
        raise ValueError("Dataframe must contain at least 2 bars.")

    ltf_minutes = int((df.index[1] - df.index[0]).total_seconds() / 60)
    htf_minutes = _to_minutes(htf_minutes) or _auto_htf(ltf_minutes)
    htf_bias_minutes = _to_minutes(htf_bias_minutes) or htf_minutes

    core = _compute_series(df, htf_minutes, capture_details=True)

    key_tf_minutes = [_to_minutes(tf) or htf_minutes for tf in key_tfs]
    key_levels: List[KeyLevelState] = []
    for tf_minutes in key_tf_minutes:
        tf_map, tf_idx = _build_period_map(df.index, tf_minutes)
        for i in range(len(df)):
            pid = tf_idx[i]
            if pid < 0:
                continue
            period = tf_map[pid]
            prev_period = tf_map[pid - 1] if pid - 1 >= 0 else None
            if prev_period is None:
                continue
            prev_high = df.iloc[prev_period["start"] : prev_period["end"] + 1]["high"].max()
            prev_low = df.iloc[prev_period["start"] : prev_period["end"] + 1]["low"].min()
            prev_open = df.iloc[prev_period["start"]]["open"]
            curr_high = df.iloc[period["start"] : i + 1]["high"].max()
            curr_low = df.iloc[period["start"] : i + 1]["low"].min()
            curr_open = df.iloc[period["start"]]["open"]
            key_levels.append(
                KeyLevelState(
                    idx=i,
                    tf_minutes=tf_minutes,
                    prev_high=prev_high,
                    prev_low=prev_low,
                    prev_open=prev_open,
                    curr_high=curr_high,
                    curr_low=curr_low,
                    curr_open=curr_open,
                    swept_high=curr_high >= prev_high,
                    swept_low=curr_low <= prev_low,
                )
            )

    bias_states: List[BiasState] = []
    bias_map, bias_idx = _build_period_map(df.index, htf_bias_minutes)
    bo0 = bh0 = bl0 = bc0 = None
    bo1 = bh1 = bl1 = bc1 = None
    bo2 = bh2 = bl2 = bc2 = None
    current_bias = 0
    last_bias_idx = None
    for i in range(len(df)):
        row = df.iloc[i]
        current_idx = bias_idx[i]
        new_bias_tf = current_idx != last_bias_idx and current_idx >= 0
        if new_bias_tf:
            if bo0 is not None:
                bo2, bh2, bl2, bc2 = bo1, bh1, bl1, bc1
                bo1, bh1, bl1, bc1 = bo0, bh0, bl0, bc0
            bo0, bh0, bl0, bc0 = row["open"], row["high"], row["low"], row["close"]
            if bo1 is not None and bo2 is not None:
                current_bias = 0
                if bc1 > bh2:
                    current_bias = 1
                if bc1 < bl2:
                    current_bias = -1
                if bc1 < bh2 and bc1 > bl2 and bh1 > bh2 and bl1 > bl2:
                    current_bias = -1
                if bc1 > bl2 and bc1 < bh2 and bh1 < bh2 and bl1 < bl2:
                    current_bias = 1
                if bh1 <= bh2 and bl1 >= bl2:
                    current_bias = 1 if bc2 > bo2 else -1
            bias_states.append(BiasState(idx=i, bias=current_bias))
        else:
            if bh0 is None:
                bo0, bh0, bl0, bc0 = row["open"], row["high"], row["low"], row["close"]
            bh0 = max(bh0, row["high"])
            bl0 = min(bl0, row["low"])
            bc0 = row["close"]
        last_bias_idx = current_idx

    smt_signals: List[SMTSignal] = []
    if smt_reference is not None and not smt_reference.empty:
        ref = smt_reference.copy()
        if not isinstance(ref.index, pd.DatetimeIndex):
            ref.index = pd.to_datetime(ref.index)
        ref_core = _compute_series(ref, htf_minutes, capture_details=False)
        ref_h0 = ref_core["h0_series"]
        ref_l0 = ref_core["l0_series"]
        ref_h1 = ref_core["h1_series"]
        ref_l1 = ref_core["l1_series"]
        for i in range(len(df)):
            if i >= len(ref_h0):
                break
            l0 = core["l0_series"][i]
            l1 = core["l1_series"][i]
            h0 = core["h0_series"][i]
            h1 = core["h1_series"][i]
            c_l = ref_l0[i] < ref_l1[i] if not np.isnan(ref_l1[i]) else False
            c_h = ref_h0[i] > ref_h1[i] if not np.isnan(ref_h1[i]) else False
            low_smt = (l0 < l1 and not c_l) or (l0 >= l1 and c_l)
            high_smt = (h0 > h1 and not c_h) or (h0 <= h1 and c_h)
            smt_signals.append(SMTSignal(idx=i, is_low_smt=low_smt, is_high_smt=high_smt))

    asset_signals: Dict[str, AssetSignalState] = {}
    if assets:
        for symbol, asset_df in assets.items():
            if not isinstance(asset_df.index, pd.DatetimeIndex):
                asset_df = asset_df.copy()
                asset_df.index = pd.to_datetime(asset_df.index)
            asset_core = _compute_series(asset_df, htf_minutes, capture_details=False)
            asset_signals[symbol] = AssetSignalState(
                symbol=symbol,
                xbull=asset_core["xbull_series"],
                xbear=asset_core["xbear_series"],
            )

    remaining_times: List[RemainingTimeState] = []
    ltf_map, _ = _build_period_map(df.index, ltf_minutes)
    for i in range(len(df)):
        remaining_times.append(
            RemainingTimeState(
                idx=i,
                tf_minutes=ltf_minutes,
                remaining_seconds=_remaining_seconds_for_bar(df.index, ltf_minutes, i, ltf_map),
            )
        )

    return {
        "htf_minutes": htf_minutes,
        "htf_candles": core["htf_candles"],
        "sweep_boxes": core["sweep_boxes"],
        "cisd_levels": core["cisd_levels"],
        "cisd_signals": core["cisd_signals"],
        "h_swept_series": core["h_swept_series"],
        "l_swept_series": core["l_swept_series"],
        "xbull_series": core["xbull_series"],
        "xbear_series": core["xbear_series"],
        "weak_bull_cross": core["weak_bull_cross"],
        "weak_bear_cross": core["weak_bear_cross"],
        "bias_states": bias_states,
        "smt_signals": smt_signals,
        "key_levels": key_levels,
        "asset_signals": asset_signals,
        "remaining_times": remaining_times,
    }


if __name__ == "__main__":
    print("cd_sweep&cisd_Cx - Python Translation")
    print("=" * 70)
    print("Detects HTF sweeps, CISD signals, SMT divergence, bias, and key levels.")
    print("=" * 70)
