"""ICT Setup 01 [TradingFinder] FVG + Liquidity Sweeps/Hunt Alerts

Python translation of the TradingFinder Pine Script (ICT Setup 01 TFlab).

This module mirrors the Pine Script logic by:
- Detecting bullish/bearish FVGs based on ATR-scaled gaps.
- Maintaining discount/premium equilibrium zones.
- Refining proximal/distal levels by candle interaction rules.
- Emitting long/short signals based on hunts/sweeps inside the FVG.
- Producing per-bar state and alert events equivalent to the Pine alert calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FVGState:
    direction: str  # "bull" or "bear"
    distal: float
    proximal: float
    point: int
    discount: float
    premium: float
    equilibrium: float


@dataclass
class SignalState:
    index: int
    long_signal: bool
    short_signal: bool


@dataclass
class BarState:
    index: int
    bull_fvg: bool
    bear_fvg: bool
    validity_bull: bool
    validity_bear: bool
    distal_bull: float
    proximal_bull: float
    distal_bear: float
    proximal_bear: float
    bull_point: int
    bear_point: int
    long_count: int
    short_count: int
    low_tracker: float
    high_tracker: float
    long_signal: bool
    short_signal: bool
    discount_bull: float
    premium_bull: float
    equilibrium_bull: float
    discount_bear: float
    premium_bear: float
    equilibrium_bear: float


@dataclass
class AlertEvent:
    index: int
    direction: str
    alert_name: str
    message: str


def _atr(df: pd.DataFrame, length: int = 55) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    return tr.rolling(length).mean()


def calculate_setup_01(
    df: pd.DataFrame,
    *,
    matr: float = 1.0,
    fvg_validity: int = 15,
    discount_premium: bool = False,
    issue_signal_method: str = "Hunt",
    max_signals: int = 3,
    signal_after_hunts: bool = False,
    hunts_needed: int = 2,
    alert: str = "On",
    alert_name: str = "ICT Setup 01 Alerts [TradingFinder]",
    message_bull: str = "Long Signal Position Based on ICT Setup 01 [FVG Hunts]",
    message_bear: str = "Short Signal Position Based on ICT Setup 01 [FVG Hunts]",
) -> dict:
    """Replicate ICT Setup 01 logic and return FVGs and signals.

    Args:
        df: DataFrame with columns open, high, low, close.
        matr: FVG detector multiplier factor.
        fvg_validity: Validity period in bars.
        discount_premium: Whether to use discount/premium refinement.
        issue_signal_method: "Hunt" or "Sweeps".
        max_signals: Max signals allowed from a zone.
        signal_after_hunts: Require hunts_needed hits before signaling.
        hunts_needed: Number of hunts/sweeps to confirm a signal.

    Returns:
        Dict with fvg_states and signal_states.
    """
    atr = _atr(df, 55)

    fvg_states: List[FVGState] = []
    signal_states: List[SignalState] = []
    bar_states: List[BarState] = []
    alert_events: List[AlertEvent] = []

    bull_fvg = False
    bear_fvg = False

    bu_distal = 0.0
    bu_proximal = 0.0
    bu_point = 0
    bu_discount = 0.0
    bu_premium = 0.0
    bu_equilibrium = 0.0

    be_distal = 0.0
    be_proximal = 0.0
    be_point = 0
    be_discount = 0.0
    be_premium = 0.0
    be_equilibrium = 0.0

    distal_lvl_bu = 0.0
    proximal_lvl_bu = 0.0
    distal_lvl_be = 0.0
    proximal_lvl_be = 0.0

    validity_bu = True
    validity_be = True

    low_tracker = 0.0
    high_tracker = 0.0

    long_count = 0
    short_count = 0

    long_signal = False
    short_signal = False

    distal_bu_history = []
    proximal_bu_history = []
    distal_be_history = []
    proximal_be_history = []
    bu_point_history = []
    be_point_history = []

    for i in range(len(df)):
        open_i = float(df["open"].iloc[i])
        high_i = float(df["high"].iloc[i])
        low_i = float(df["low"].iloc[i])
        close_i = float(df["close"].iloc[i])
        atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else np.nan

        def value(series: pd.Series, offset: int) -> Optional[float]:
            idx = i - offset
            if idx < 0:
                return None
            return float(series.iloc[idx])

        high_2 = value(df["high"], 2)
        low_2 = value(df["low"], 2)
        high_1 = value(df["high"], 1)
        low_1 = value(df["low"], 1)
        open_1 = value(df["open"], 1)
        close_1 = value(df["close"], 1)

        body_1 = None if open_1 is None else close_1 - open_1

        bull_fvg = False
        bear_fvg = False

        if high_2 is not None and low_2 is not None and not np.isnan(atr_i):
            if (high_i - low_2) > (matr * atr_i):
                if (
                    low_i > high_2
                    and low_2 < (low_1 if low_1 is not None else low_2)
                    and (high_1 if high_1 is not None else high_i) < high_i
                    and (high_i + low_2) / 2 >= high_2
                ):
                    bu_distal = high_2
                    bu_proximal = low_i
                    bu_point = i
                    bu_discount = low_2
                    bu_premium = high_i
                    bu_equilibrium = (high_i + low_2) / 2
                    bull_fvg = True

            if (high_2 - low_i) > (matr * atr_i):
                if (
                    low_2 > high_i
                    and high_2 > (high_1 if high_1 is not None else high_2)
                    and (low_1 if low_1 is not None else low_i) > low_i
                    and (low_i + high_2) / 2 <= low_2
                ):
                    be_distal = low_2
                    be_proximal = high_i
                    be_point = i
                    be_discount = low_i
                    be_premium = high_2
                    be_equilibrium = (low_i + high_2) / 2
                    bear_fvg = True

        if discount_premium:
            if bull_fvg:
                if bu_equilibrium >= bu_proximal:
                    distal_lvl_bu = bu_distal
                    proximal_lvl_bu = bu_proximal
                else:
                    distal_lvl_bu = bu_distal
                    proximal_lvl_bu = bu_equilibrium
            if bear_fvg:
                if be_equilibrium <= be_proximal:
                    distal_lvl_be = be_distal
                    proximal_lvl_be = be_proximal
                else:
                    distal_lvl_be = be_distal
                    proximal_lvl_be = be_equilibrium
        else:
            if bull_fvg:
                distal_lvl_bu = bu_distal
                proximal_lvl_bu = bu_proximal
            if bear_fvg:
                distal_lvl_be = be_distal
                proximal_lvl_be = be_proximal

        prev_distal_bu = distal_bu_history[-1] if distal_bu_history else 0.0
        prev_prox_bu = proximal_bu_history[-1] if proximal_bu_history else 0.0
        prev_distal_be = distal_be_history[-1] if distal_be_history else 0.0
        prev_prox_be = proximal_be_history[-1] if proximal_be_history else 0.0

        if validity_bu:
            if body_1 is not None and body_1 > 0:
                sweep_check = open_1 < prev_distal_bu if issue_signal_method == "Sweeps" else low_1 < prev_distal_bu
                if sweep_check or i > bu_point + fvg_validity or (not signal_after_hunts and long_count > max_signals - 1):
                    validity_bu = False
                elif open_1 < prev_prox_bu and open_1 > prev_distal_bu:
                    proximal_lvl_bu = open_1
            if body_1 is not None and body_1 <= 0:
                sweep_check = close_1 < prev_distal_bu if issue_signal_method == "Sweeps" else low_1 < prev_distal_bu
                if sweep_check or i > bu_point + fvg_validity or (not signal_after_hunts and long_count > max_signals - 1):
                    validity_bu = False
                elif close_1 < prev_prox_bu and close_1 > prev_distal_bu:
                    proximal_lvl_bu = close_1

        if validity_be:
            if body_1 is not None and body_1 > 0:
                sweep_check = close_1 > prev_distal_be if issue_signal_method == "Sweeps" else high_1 > prev_distal_be
                if sweep_check or i > be_point + fvg_validity or (not signal_after_hunts and short_count > max_signals - 1):
                    validity_be = False
                elif close_1 > prev_prox_be and close_1 < prev_distal_be:
                    proximal_lvl_be = close_1
            if body_1 is not None and body_1 <= 0:
                sweep_check = open_1 > prev_distal_be if issue_signal_method == "Sweeps" else high_1 > prev_distal_be
                if sweep_check or i > be_point + fvg_validity or (not signal_after_hunts and short_count > max_signals - 1):
                    validity_be = False
                elif open_1 > prev_prox_be and open_1 < prev_distal_be:
                    proximal_lvl_be = open_1

        prev_bu_point = bu_point_history[-1] if bu_point_history else bu_point
        prev_be_point = be_point_history[-1] if be_point_history else be_point
        if prev_bu_point != bu_point:
            validity_bu = True
            low_tracker = 0.0
            long_count = 0
            long_signal = False
        if prev_be_point != be_point:
            validity_be = True
            high_tracker = 0.0
            short_count = 0
            short_signal = False

        if validity_bu:
            if low_tracker == 0.0 and low_i < proximal_lvl_bu:
                low_tracker = low_i
            if low_i < low_tracker and low_tracker > 0.0:
                low_tracker = low_i
                if close_i >= proximal_lvl_bu:
                    long_count += 1
                    long_signal = long_count == hunts_needed if signal_after_hunts else True
                else:
                    long_signal = False
            else:
                long_signal = False
        else:
            low_tracker = 0.0
            long_count = 0
            long_signal = False

        if validity_be:
            if high_tracker == 0.0 and high_i > proximal_lvl_be:
                high_tracker = high_i
            if high_i > high_tracker and high_tracker > 0.0:
                high_tracker = high_i
                if close_i <= proximal_lvl_be:
                    short_count += 1
                    short_signal = short_count == hunts_needed if signal_after_hunts else True
                else:
                    short_signal = False
            else:
                short_signal = False
        else:
            high_tracker = 0.0
            short_count = 0
            short_signal = False

        if bull_fvg:
            fvg_states.append(
                FVGState(
                    direction="bull",
                    distal=distal_lvl_bu,
                    proximal=proximal_lvl_bu,
                    point=bu_point,
                    discount=bu_discount,
                    premium=bu_premium,
                    equilibrium=bu_equilibrium,
                )
            )
        if bear_fvg:
            fvg_states.append(
                FVGState(
                    direction="bear",
                    distal=distal_lvl_be,
                    proximal=proximal_lvl_be,
                    point=be_point,
                    discount=be_discount,
                    premium=be_premium,
                    equilibrium=be_equilibrium,
                )
            )

        signal_states.append(SignalState(i, long_signal, short_signal))
        bar_states.append(
            BarState(
                index=i,
                bull_fvg=bull_fvg,
                bear_fvg=bear_fvg,
                validity_bull=validity_bu,
                validity_bear=validity_be,
                distal_bull=distal_lvl_bu,
                proximal_bull=proximal_lvl_bu,
                distal_bear=distal_lvl_be,
                proximal_bear=proximal_lvl_be,
                bull_point=bu_point,
                bear_point=be_point,
                long_count=long_count,
                short_count=short_count,
                low_tracker=low_tracker,
                high_tracker=high_tracker,
                long_signal=long_signal,
                short_signal=short_signal,
                discount_bull=bu_discount,
                premium_bull=bu_premium,
                equilibrium_bull=bu_equilibrium,
                discount_bear=be_discount,
                premium_bear=be_premium,
                equilibrium_bear=be_equilibrium,
            )
        )
        if alert == "On":
            if long_signal:
                alert_events.append(
                    AlertEvent(
                        index=i,
                        direction="long",
                        alert_name=alert_name,
                        message=message_bull,
                    )
                )
            if short_signal:
                alert_events.append(
                    AlertEvent(
                        index=i,
                        direction="short",
                        alert_name=alert_name,
                        message=message_bear,
                    )
                )

        distal_bu_history.append(distal_lvl_bu)
        proximal_bu_history.append(proximal_lvl_bu)
        distal_be_history.append(distal_lvl_be)
        proximal_be_history.append(proximal_lvl_be)
        bu_point_history.append(bu_point)
        be_point_history.append(be_point)

    return {
        "fvg_states": fvg_states,
        "signals": signal_states,
        "bar_states": bar_states,
        "alerts": alert_events,
    }
