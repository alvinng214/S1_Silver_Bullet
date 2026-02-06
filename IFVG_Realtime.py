"""IFVG Signals Realtime (No Wait) - Python translation.

This module mirrors the Pine Script logic in `S1_Silver_Bullet/IFVG_Realtime.txt` by:
- Scanning for bullish/bearish FVGs within a configurable lookback window.
- Validating minimum FVG size and checking for prior breaks in confirmed bars.
- Emitting realtime buy/sell signals when price crosses the FVG boundary.
- Applying an EMA/SMA slope + price filter at the time of signal.

The implementation iterates bar-by-bar to reflect the Pine behavior where
conditions are evaluated using the current close without waiting for bar close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class IFVGZone:
    direction: str  # "buy" or "sell"
    left_index: int
    right_index: int
    top: float
    bottom: float


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _detect_fvg(
    df: pd.DataFrame,
    current_index: int,
    offset: int,
    eps_val: float,
) -> int:
    """Return 1 for bull gap, -1 for bear gap, 0 for none."""
    lookback_index = current_index - (offset + 2)
    if lookback_index < 0:
        return 0

    h2 = float(df["high"].iloc[lookback_index])
    l2 = float(df["low"].iloc[lookback_index])
    lt = float(df["low"].iloc[current_index - offset])
    ht = float(df["high"].iloc[current_index - offset])

    if lt > h2 - eps_val:
        return 1
    if ht < l2 + eps_val:
        return -1
    return 0


def compute_ifvg_realtime(
    df: pd.DataFrame,
    *,
    mintick: float,
    pip_size_multiplier: float = 1.0,
    ifvg_gap_bars: int = 15,
    min_fvg_pips: float = 0.0,
    fvg_eps_points: float = 0.0,
    show_zones: bool = True,
    ma_period: int = 21,
    ma_kind: str = "EMA",
) -> Tuple[pd.DataFrame, List[IFVGZone]]:
    """Compute IFVG realtime signals.

    Args:
        df: DataFrame with columns: open, high, low, close.
        mintick: Minimum tick size (syminfo.mintick in Pine).
        pip_size_multiplier: Multiplier for tick size (kept for parity).
        ifvg_gap_bars: Lookback bars for FVG search.
        min_fvg_pips: Minimum FVG size in pips/points.
        fvg_eps_points: Epsilon tolerance in price units.
        show_zones: Whether to collect IFVG zones.
        ma_period: Moving average period.
        ma_kind: "EMA" or "SMA".

    Returns:
        Tuple of (DataFrame with signal columns, list of IFVGZone entries).
    """
    if ma_kind not in {"EMA", "SMA"}:
        raise ValueError("ma_kind must be 'EMA' or 'SMA'.")

    price = df["close"].astype(float)
    ma_val = _ema(price, ma_period) if ma_kind == "EMA" else _sma(price, ma_period)

    _ = mintick * pip_size_multiplier  # retained for parity with Pine settings

    min_size_val = min_fvg_pips * (mintick * 10)

    signal_dir: List[int] = []
    zones: List[IFVGZone] = []

    for idx in range(len(df)):
        current_signal = 0

        for i in range(1, ifvg_gap_bars + 1):
            if idx - (i + 2) < 0:
                continue

            fvg_type = _detect_fvg(df, idx, i, fvg_eps_points)

            if fvg_type == 1:
                gap_low = float(df["high"].iloc[idx - (i + 2)])
                gap_high = float(df["low"].iloc[idx - i])

                if (gap_high - gap_low) >= min_size_val:
                    already_broken = False
                    if i > 1:
                        for k in range(i - 1, 0, -1):
                            if float(df["close"].iloc[idx - k]) < gap_low:
                                already_broken = True
                                break

                    if not already_broken and float(df["close"].iloc[idx]) < gap_low:
                        ma_condition = False
                        if idx - 1 >= 0 and pd.notna(ma_val.iloc[idx]) and pd.notna(ma_val.iloc[idx - 1]):
                            ma_condition = ma_val.iloc[idx] < ma_val.iloc[idx - 1] and float(
                                df["close"].iloc[idx]
                            ) < float(ma_val.iloc[idx])

                        if ma_condition:
                            current_signal = -1
                            if show_zones:
                                zones.append(
                                    IFVGZone(
                                        direction="sell",
                                        left_index=idx - (i + 2),
                                        right_index=idx,
                                        top=gap_high,
                                        bottom=gap_low,
                                    )
                                )
                            break

            if fvg_type == -1:
                gap_low2 = float(df["high"].iloc[idx - i])
                gap_high2 = float(df["low"].iloc[idx - (i + 2)])

                if (gap_high2 - gap_low2) >= min_size_val:
                    already_broken2 = False
                    if i > 1:
                        for k in range(i - 1, 0, -1):
                            if float(df["close"].iloc[idx - k]) > gap_high2:
                                already_broken2 = True
                                break

                    if not already_broken2 and float(df["close"].iloc[idx]) > gap_high2:
                        ma_condition2 = False
                        if idx - 1 >= 0 and pd.notna(ma_val.iloc[idx]) and pd.notna(ma_val.iloc[idx - 1]):
                            ma_condition2 = ma_val.iloc[idx] > ma_val.iloc[idx - 1] and float(
                                df["close"].iloc[idx]
                            ) > float(ma_val.iloc[idx])

                        if ma_condition2:
                            current_signal = 1
                            if show_zones:
                                zones.append(
                                    IFVGZone(
                                        direction="buy",
                                        left_index=idx - (i + 2),
                                        right_index=idx,
                                        top=gap_high2,
                                        bottom=gap_low2,
                                    )
                                )
                            break

        signal_dir.append(current_signal)

    result = df.copy()
    result["signal_dir"] = signal_dir
    result["buy_signal"] = result["signal_dir"] == 1
    result["sell_signal"] = result["signal_dir"] == -1
    result["ma"] = ma_val

    return result, zones
