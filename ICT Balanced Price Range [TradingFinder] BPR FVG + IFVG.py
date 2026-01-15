"""
ICT Balanced Price Range (BPR) | FVG + IFVG
Python translation of the TradingFinder Pine Script.

This module mirrors the Pine logic at a functional level by:
- Detecting FVGs with optional width filtering.
- Tracking invalidated FVGs as inversion FVGs (IFVGs).
- Building Balanced Price Ranges (BPRs) where bullish/bearish zones overlap.
- Emitting mitigation signals when price re-enters a BPR.

Inputs are expressed as function arguments to keep the logic reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FVGZone:
    direction: str  # "bullish" or "bearish"
    proximal: float
    distal: float
    start_idx: int
    end_idx: int
    mitigated: bool = False

    def contains(self, price: float) -> bool:
        lower = min(self.proximal, self.distal)
        upper = max(self.proximal, self.distal)
        return lower <= price <= upper


@dataclass
class IFVGZone:
    direction: str  # "bullish" or "bearish"
    proximal: float
    distal: float
    created_at: int
    origin_start_idx: int

    def overlaps(self, other: FVGZone) -> bool:
        lower_a, upper_a = sorted((self.proximal, self.distal))
        lower_b, upper_b = sorted((other.proximal, other.distal))
        return max(lower_a, lower_b) <= min(upper_a, upper_b)


@dataclass
class BPRZone:
    direction: str  # "bullish" or "bearish"
    proximal: float
    distal: float
    start_idx: int
    end_idx: int
    mitigated: bool = False

    def contains(self, price: float) -> bool:
        lower = min(self.proximal, self.distal)
        upper = max(self.proximal, self.distal)
        return lower <= price <= upper


def _calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    return tr.rolling(length).mean()


def detect_fvgs(
    df: pd.DataFrame,
    filter_on: bool = True,
    filter_type: str = "Defensive",
) -> List[FVGZone]:
    """
    Detect Fair Value Gaps using the same gap rules as the Pine library:
    - Bullish FVG: low[i] > high[i-2]
    - Bearish FVG: high[i] < low[i-2]

    Filter types map to ATR multipliers in the same order as Pine.
    """
    fvgs: List[FVGZone] = []
    atr = _calculate_atr(df) if filter_on else None
    multipliers = {
        "Very Aggressive": 0.0,
        "Aggressive": 0.5,
        "Defensive": 0.7,
        "Very Defensive": 1.0,
    }
    multiplier = multipliers.get(filter_type, 0.7)

    for i in range(2, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            proximal = float(df["low"].iloc[i])
            distal = float(df["high"].iloc[i - 2])
            if filter_on and atr is not None:
                width = proximal - distal
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * multiplier:
                    continue
            fvgs.append(
                FVGZone(
                    direction="bullish",
                    proximal=proximal,
                    distal=distal,
                    start_idx=int(df.index[i - 1]),
                    end_idx=int(df.index[i]),
                )
            )
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            proximal = float(df["high"].iloc[i])
            distal = float(df["low"].iloc[i - 2])
            if filter_on and atr is not None:
                width = distal - proximal
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * multiplier:
                    continue
            fvgs.append(
                FVGZone(
                    direction="bearish",
                    proximal=proximal,
                    distal=distal,
                    start_idx=int(df.index[i - 1]),
                    end_idx=int(df.index[i]),
                )
            )

    return fvgs


def detect_ifvgs(
    df: pd.DataFrame,
    fvgs: Iterable[FVGZone],
) -> List[IFVGZone]:
    """
    Convert invalidated FVGs into IFVGs.

    - Bullish FVG becomes bearish IFVG if price closes below its distal.
    - Bearish FVG becomes bullish IFVG if price closes above its distal.
    """
    ifvgs: List[IFVGZone] = []

    for fvg in fvgs:
        start = fvg.end_idx + 1
        for i in range(start, len(df)):
            close = float(df["close"].iloc[i])
            if fvg.direction == "bullish" and close < fvg.distal:
                ifvgs.append(
                    IFVGZone(
                        direction="bearish",
                        proximal=fvg.proximal,
                        distal=fvg.distal,
                        created_at=int(df.index[i]),
                        origin_start_idx=fvg.start_idx,
                    )
                )
                break
            if fvg.direction == "bearish" and close > fvg.distal:
                ifvgs.append(
                    IFVGZone(
                        direction="bullish",
                        proximal=fvg.proximal,
                        distal=fvg.distal,
                        created_at=int(df.index[i]),
                        origin_start_idx=fvg.start_idx,
                    )
                )
                break

    return ifvgs


def compute_bprs(
    df: pd.DataFrame,
    fvgs: Iterable[FVGZone],
    ifvgs: Iterable[IFVGZone],
    validity_bars: int = 500,
) -> List[BPRZone]:
    """
    Build Balanced Price Ranges from overlapping FVG and IFVG zones.

    - Bullish BPR: bullish FVG overlapping bearish IFVG.
    - Bearish BPR: bearish FVG overlapping bullish IFVG.
    """
    bprs: List[BPRZone] = []

    for fvg in fvgs:
        for ifvg in ifvgs:
            if fvg.direction == "bullish" and ifvg.direction == "bearish":
                if not ifvg.overlaps(fvg):
                    continue
                overlap_low = max(min(fvg.proximal, fvg.distal), min(ifvg.proximal, ifvg.distal))
                overlap_high = min(max(fvg.proximal, fvg.distal), max(ifvg.proximal, ifvg.distal))
                start_idx = max(fvg.start_idx, ifvg.created_at)
                end_idx = start_idx + validity_bars
                bprs.append(
                    BPRZone(
                        direction="bullish",
                        proximal=overlap_high,
                        distal=overlap_low,
                        start_idx=start_idx,
                        end_idx=end_idx,
                    )
                )
            elif fvg.direction == "bearish" and ifvg.direction == "bullish":
                if not ifvg.overlaps(fvg):
                    continue
                overlap_low = max(min(fvg.proximal, fvg.distal), min(ifvg.proximal, ifvg.distal))
                overlap_high = min(max(fvg.proximal, fvg.distal), max(ifvg.proximal, ifvg.distal))
                start_idx = max(fvg.start_idx, ifvg.created_at)
                end_idx = start_idx + validity_bars
                bprs.append(
                    BPRZone(
                        direction="bearish",
                        proximal=overlap_low,
                        distal=overlap_high,
                        start_idx=start_idx,
                        end_idx=end_idx,
                    )
                )

    return bprs


def detect_bpr_mitigations(
    df: pd.DataFrame,
    bprs: Iterable[BPRZone],
) -> List[Tuple[int, BPRZone]]:
    """
    Track mitigation events: first entry of price into a BPR zone.
    """
    mitigations: List[Tuple[int, BPRZone]] = []

    for bpr in bprs:
        for i in range(bpr.start_idx, min(bpr.end_idx + 1, len(df))):
            price = float(df["close"].iloc[i])
            if bpr.contains(price):
                bpr.mitigated = True
                mitigations.append((int(df.index[i]), bpr))
                break

    return mitigations


def calculate_bpr_indicator(
    df: pd.DataFrame,
    *,
    show_all_ifvg: bool = True,
    fvg_validity: int = 500,
    fvg_filter_on: bool = True,
    fvg_filter_type: str = "Defensive",
) -> dict:
    """
    End-to-end calculation mirroring the Pine Script output.

    Returns a dict with FVGs, IFVGs, BPRs, and mitigation alerts.
    """
    fvgs = detect_fvgs(df, filter_on=fvg_filter_on, filter_type=fvg_filter_type)
    ifvgs = detect_ifvgs(df, fvgs)
    if not show_all_ifvg:
        ifvgs = []
    bprs = compute_bprs(df, fvgs, ifvgs, validity_bars=fvg_validity)
    mitigations = detect_bpr_mitigations(df, bprs)

    return {
        "fvgs": fvgs,
        "ifvgs": ifvgs,
        "bprs": bprs,
        "mitigations": mitigations,
    }
