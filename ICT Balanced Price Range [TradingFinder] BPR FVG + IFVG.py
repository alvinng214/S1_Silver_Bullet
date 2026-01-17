"""ICT Balanced Price Range (BPR) | FVG + IFVG.

Python translation of the TradingFinder Pine Script.

This module mirrors the Pine logic by:
- Detecting bullish and bearish FVGs with optional ATR width filtering.
- Tracking IFVG (inverted FVG) when price closes through a FVG.
- Computing BPR zones from overlapping FVG/IFVG pairs.
- Emitting mitigation alerts when price reacts at configured mitigation levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FVGZone:
    direction: str  # "bullish" or "bearish"
    distal: float
    proximal: float
    index: int
    valid_until: int
    mitigated: bool = False


@dataclass
class IFVGZone:
    direction: str  # "bullish" or "bearish"
    distal: float
    proximal: float
    index: int


@dataclass
class BPRZone:
    direction: str  # "bullish" or "bearish"
    distal: float
    proximal: float
    index: int
    valid_until: int
    mitigated: bool = False


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    return tr.rolling(length).mean()


def _mitigation_level(distal: float, proximal: float, mode: str) -> float:
    if mode == "Distal":
        return distal
    if mode == "50 % OB":
        return (distal + proximal) / 2.0
    return proximal


def detect_fvgs(
    df: pd.DataFrame,
    *,
    filter_on: bool = True,
    filter_type: str = "Defensive",
    validity: int = 500,
) -> List[FVGZone]:
    """Detect FVGs matching FVGDetectorLibrary semantics."""
    atr = _atr(df) if filter_on else None
    multipliers = {
        "Very Aggressive": 0.0,
        "Aggressive": 0.5,
        "Defensive": 0.7,
        "Very Defensive": 1.0,
    }
    mult = multipliers.get(filter_type, 0.7)

    zones: List[FVGZone] = []

    for i in range(2, len(df)):
        high_2 = float(df["high"].iloc[i - 2])
        low_2 = float(df["low"].iloc[i - 2])
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])

        if low > high_2:
            width = low - high_2
            if filter_on and atr is not None:
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * mult:
                    continue
            zones.append(
                FVGZone(
                    direction="bullish",
                    distal=high_2,
                    proximal=low,
                    index=i,
                    valid_until=i + validity,
                )
            )
        elif high < low_2:
            width = low_2 - high
            if filter_on and atr is not None:
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * mult:
                    continue
            zones.append(
                FVGZone(
                    direction="bearish",
                    distal=low_2,
                    proximal=high,
                    index=i,
                    valid_until=i + validity,
                )
            )

    return zones


def detect_ifvgs(df: pd.DataFrame, fvgs: List[FVGZone]) -> List[IFVGZone]:
    """Detect IFVGs (inversion) when price closes through a FVG."""
    ifvgs: List[IFVGZone] = []
    for zone in fvgs:
        for i in range(zone.index + 1, min(zone.valid_until + 1, len(df))):
            close = float(df["close"].iloc[i])
            if zone.direction == "bullish" and close < zone.proximal:
                ifvgs.append(
                    IFVGZone(
                        direction="bearish",
                        distal=zone.distal,
                        proximal=zone.proximal,
                        index=i,
                    )
                )
                break
            if zone.direction == "bearish" and close > zone.proximal:
                ifvgs.append(
                    IFVGZone(
                        direction="bullish",
                        distal=zone.distal,
                        proximal=zone.proximal,
                        index=i,
                    )
                )
                break
    return ifvgs


def _overlap(a_distal: float, a_prox: float, b_distal: float, b_prox: float) -> Optional[tuple]:
    a_low, a_high = sorted((a_distal, a_prox))
    b_low, b_high = sorted((b_distal, b_prox))
    low = max(a_low, b_low)
    high = min(a_high, b_high)
    if low <= high:
        return low, high
    return None


def build_bprs(
    fvgs: List[FVGZone],
    ifvgs: List[IFVGZone],
    validity: int,
) -> List[BPRZone]:
    """Build BPRs from overlapping FVG/IFVG pairs."""
    zones: List[BPRZone] = []
    for fvg in fvgs:
        for ifvg in ifvgs:
            if fvg.direction == "bullish" and ifvg.direction == "bearish":
                overlap = _overlap(fvg.distal, fvg.proximal, ifvg.distal, ifvg.proximal)
                if overlap:
                    low, high = overlap
                    zones.append(
                        BPRZone(
                            direction="bullish",
                            distal=low,
                            proximal=high,
                            index=max(fvg.index, ifvg.index),
                            valid_until=max(fvg.index, ifvg.index) + validity,
                        )
                    )
            if fvg.direction == "bearish" and ifvg.direction == "bullish":
                overlap = _overlap(fvg.distal, fvg.proximal, ifvg.distal, ifvg.proximal)
                if overlap:
                    low, high = overlap
                    zones.append(
                        BPRZone(
                            direction="bearish",
                            distal=low,
                            proximal=high,
                            index=max(fvg.index, ifvg.index),
                            valid_until=max(fvg.index, ifvg.index) + validity,
                        )
                    )
    return zones


def detect_bpr_mitigations(
    df: pd.DataFrame,
    bprs: List[BPRZone],
    mitigation_mode: str,
) -> List[int]:
    """Return indices where BPR mitigations occur."""
    alerts: List[int] = []
    for zone in bprs:
        level = _mitigation_level(zone.distal, zone.proximal, mitigation_mode)
        for i in range(zone.index, min(zone.valid_until + 1, len(df))):
            if zone.direction == "bullish" and df["low"].iloc[i] <= level:
                zone.mitigated = True
                alerts.append(i)
                break
            if zone.direction == "bearish" and df["high"].iloc[i] >= level:
                zone.mitigated = True
                alerts.append(i)
                break
    return alerts


def calculate_bpr_indicator(
    df: pd.DataFrame,
    *,
    show_all_ifvg: bool = True,
    fvg_validity: int = 500,
    fvg_filter_on: bool = True,
    fvg_filter_type: str = "Defensive",
    mitigation_bpr: str = "Proximal",
    mitigation_fvg: str = "Proximal",
) -> Dict[str, object]:
    """End-to-end BPR/IFVG calculation mirroring the Pine script."""
    fvgs = detect_fvgs(df, filter_on=fvg_filter_on, filter_type=fvg_filter_type, validity=fvg_validity)
    ifvgs = detect_ifvgs(df, fvgs)
    if not show_all_ifvg:
        ifvgs = []
    bprs = build_bprs(fvgs, ifvgs, fvg_validity)

    bpr_alerts = detect_bpr_mitigations(df, bprs, mitigation_bpr)

    fvg_alerts = []
    for zone in fvgs:
        level = _mitigation_level(zone.distal, zone.proximal, mitigation_fvg)
        for i in range(zone.index, min(zone.valid_until + 1, len(df))):
            if zone.direction == "bullish" and df["low"].iloc[i] <= level:
                fvg_alerts.append(i)
                break
            if zone.direction == "bearish" and df["high"].iloc[i] >= level:
                fvg_alerts.append(i)
                break

    return {
        "fvgs": fvgs,
        "ifvgs": ifvgs,
        "bprs": bprs,
        "bpr_alerts": bpr_alerts,
        "fvg_alerts": fvg_alerts,
    }
