"""
Backtest script to compute HTF market structure bias using Backtrader.

Steps:
1) Load OHLC data from CSV and set a datetime index.
2) Compute Smart Money Zones MTF trends (4H/1D) with pandas.
3) Attach Smart Money trend columns to the base dataframe.
4) Identify external liquidity targets (profit targets) via SMC + inducements.
5) Trigger entry signals at FVG zones (Silver Bullet/TradingFinder/Fib OTE).
6) Feed data into Backtrader and resample to 4H/1D.
7) Run HTFBiasStrategy to log consolidated HTF bias.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict
from zoneinfo import ZoneInfo

import backtrader as bt
import pandas as pd
from importlib.machinery import SourceFileLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))

from htf_bias_strategy import HTFBiasStrategy, SilverBulletStrategy
from ICT_Silver_Bullet_with_signals import detect_silver_bullet_signals
from IFVG_Realtime import compute_ifvg_realtime
from Smart_Money_Concept__TradingFinder__Major_Minor_OB___FVG__SMC_ import calculate_smc_tradingfinder
from Smart_Money_Zones__FVG___OB____MTF_Trend_Panel import calculate_smart_money_zones

LIQUIDITY_PATH = os.path.join(
    os.path.dirname(__file__),
    "Liquidity & inducements.py",
)
liquidity_module = SourceFileLoader("liquidity_inducements", LIQUIDITY_PATH).load_module()
calculate_liquidity_inducements = liquidity_module.calculate_liquidity_inducements



SETUP01_PATH = os.path.join(
    os.path.dirname(__file__),
    "ICT Setup 01 [TradingFinder] FVG + Liquidity SweepsHunt Alerts, ICT Setup 01 TFlab.py",
)
setup01_module = SourceFileLoader("ict_setup_01", SETUP01_PATH).load_module()
calculate_setup_01 = setup01_module.calculate_setup_01

FIB_OTE_PATH = os.path.join(
    os.path.dirname(__file__),
    "Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py",
)
fib_ote_module = SourceFileLoader("fib_ote_zeierman", FIB_OTE_PATH).load_module()
calculate_fibonacci_ote = fib_ote_module.calculate_fibonacci_ote

ICT_SESSIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "ICT Sessions_One Setup for Life [MK].py",
)
ict_sessions_module = SourceFileLoader("ict_sessions_mk", ICT_SESSIONS_PATH).load_module()

MARKET_STRUCTURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "Market Structure MTF Trend [Pt].py",
)
market_structure_module = SourceFileLoader(
    "market_structure_mtf_trend",
    MARKET_STRUCTURE_PATH,
).load_module()
calculate_market_structure_mtf = market_structure_module.calculate_market_structure_mtf

MTF_OB_FINDER_PATH = os.path.join(
    os.path.dirname(__file__),
    "MTF Order Block Finder.py",
)
mtf_ob_module = SourceFileLoader("mtf_order_block_finder", MTF_OB_FINDER_PATH).load_module()
compute_mtf_order_block_finder = mtf_ob_module.compute_mtf_order_block_finder
OBSettings = mtf_ob_module.OBSettings

OB_IMBALANCE_PATH = os.path.join(
    os.path.dirname(__file__),
    "Order Blocks & Imbalance MTF.py",
)
ob_imbalance_module = SourceFileLoader("order_blocks_imbalance_mtf", OB_IMBALANCE_PATH).load_module()
OBImbalanceSettings = ob_imbalance_module.OBSettings

MTF_FVG_PATH = os.path.join(
    os.path.dirname(__file__),
    "MTF FVG x2 [MK].py",
)
mtf_fvg_module = SourceFileLoader("mtf_fvg_x2_mk", MTF_FVG_PATH).load_module()
MTFSettings = mtf_fvg_module.MTFSettings

OB_DETECTOR_PATH = os.path.join(
    os.path.dirname(__file__),
    "Order-Block Detector.py",
)
ob_detector_module = SourceFileLoader("order_block_detector", OB_DETECTOR_PATH).load_module()
OrderBlockDetector = ob_detector_module.OrderBlockDetector


# =============================================================================
# INDICATOR CACHE - Stores expensive calculations to avoid redundant computation
# =============================================================================
class IndicatorCache:
    """Cache for expensive indicator calculations to avoid redundant computation."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._silver_bullet_signals = None
        self._smc_tradingfinder = None
        self._smart_money_zones = None
        self._hk_aligned_df = None
        self._ob_detector_results = None

    def get_silver_bullet_signals(self, hk_aligned: pd.DataFrame) -> dict:
        if self._silver_bullet_signals is None:
            self._silver_bullet_signals = detect_silver_bullet_signals(hk_aligned)
        return self._silver_bullet_signals

    def get_smc_tradingfinder(self, df: pd.DataFrame) -> dict:
        if self._smc_tradingfinder is None:
            self._smc_tradingfinder = calculate_smc_tradingfinder(df)
        return self._smc_tradingfinder

    def get_smart_money_zones(self, df: pd.DataFrame, show_ob: bool = True) -> dict:
        if self._smart_money_zones is None:
            self._smart_money_zones = calculate_smart_money_zones(df, show_ob=show_ob)
        return self._smart_money_zones

    def get_ob_detector_results(self, df: pd.DataFrame) -> dict:
        if self._ob_detector_results is None:
            detector = OrderBlockDetector()
            self._ob_detector_results = detector.run(df)
        return self._ob_detector_results


# Global cache instance
_cache = IndicatorCache()


class PandasDataBias(bt.feeds.PandasData):
    lines = (
        "smz_trend_15m",
        "smz_trend_4h",
        "smz_trend_1h",
        "smz_trend_1d",
        "smc_liquidity_high",
        "smc_liquidity_low",
        "liq_buyside_target",
        "liq_sellside_target",
        "sb_sig_ln",
        "sb_sig_am",
        "sb_sig_pm",
        "entry_sb_bull",
        "entry_sb_bear",
        "entry_setup01_bull",
        "entry_setup01_bear",
        "entry_ote_bull",
        "entry_ote_bear",
        "entry_ifvg_bull",
        "entry_ifvg_bear",
        "entry_trigger_bull",
        "entry_trigger_bear",
        "filter_session_active",
        "filter_htf_bias_bull",
        "filter_htf_bias_bear",
        "filter_htf_poi_bull",
        "filter_htf_poi_bear",
        "entry_obdet_bull",
        "entry_obdet_bear",
        "entry_fvg_bull",
        "entry_fvg_bear",
    )
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", -1),
        ("openinterest", -1),
        ("smz_trend_15m", "smz_trend_15m"),
        ("smz_trend_4h", "smz_trend_4h"),
        ("smz_trend_1h", "smz_trend_1h"),
        ("smz_trend_1d", "smz_trend_1d"),
        ("smc_liquidity_high", "smc_liquidity_high"),
        ("smc_liquidity_low", "smc_liquidity_low"),
        ("liq_buyside_target", "liq_buyside_target"),
        ("liq_sellside_target", "liq_sellside_target"),
        ("sb_sig_ln", "sb_sig_ln"),
        ("sb_sig_am", "sb_sig_am"),
        ("sb_sig_pm", "sb_sig_pm"),
        ("entry_sb_bull", "entry_sb_bull"),
        ("entry_sb_bear", "entry_sb_bear"),
        ("entry_setup01_bull", "entry_setup01_bull"),
        ("entry_setup01_bear", "entry_setup01_bear"),
        ("entry_ote_bull", "entry_ote_bull"),
        ("entry_ote_bear", "entry_ote_bear"),
        ("entry_ifvg_bull", "entry_ifvg_bull"),
        ("entry_ifvg_bear", "entry_ifvg_bear"),
        ("entry_trigger_bull", "entry_trigger_bull"),
        ("entry_trigger_bear", "entry_trigger_bear"),
        ("filter_session_active", "filter_session_active"),
        ("filter_htf_bias_bull", "filter_htf_bias_bull"),
        ("filter_htf_bias_bear", "filter_htf_bias_bear"),
        ("filter_htf_poi_bull", "filter_htf_poi_bull"),
        ("filter_htf_poi_bear", "filter_htf_poi_bear"),
        ("entry_obdet_bull", "entry_obdet_bull"),
        ("entry_obdet_bear", "entry_obdet_bear"),
        ("entry_fvg_bull", "entry_fvg_bull"),
        ("entry_fvg_bear", "entry_fvg_bear"),
    )


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_data(csv_file: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df = df.set_index("time").sort_index()
    base_columns = ["open", "high", "low", "close"]
    if "Long" in df.columns:
        base_columns.append("Long")
    if "Short" in df.columns:
        base_columns.append("Short")
    df = df[base_columns].dropna(subset=["open", "high", "low", "close"])
    if max_rows is not None and len(df) > max_rows:
        df = df.tail(max_rows)
    return df


def limit_bars(df: pd.DataFrame, max_bars: int | None) -> pd.DataFrame:
    if max_bars is None or max_bars <= 0 or len(df) <= max_bars:
        return df
    trimmed = df.iloc[-max_bars:]
    print(
        f"Limiting dataset to last {max_bars} bars "
        f"({trimmed.index.min()} to {trimmed.index.max()})."
    )
    return trimmed


def add_smart_money_trends(df: pd.DataFrame) -> pd.DataFrame:
    results = _cache.get_smart_money_zones(df)
    trend_15m = results["mtf_trends"]["15m"].astype(int).replace({0: -1})
    trend_1h = results["mtf_trends"]["1h"].astype(int).replace({0: -1})
    trend_4h = results["mtf_trends"]["4h"].astype(int).replace({0: -1})
    trend_1d = results["mtf_trends"]["1d"].astype(int).replace({0: -1})

    df = df.copy()
    df["smz_trend_15m"] = trend_15m
    df["smz_trend_1h"] = trend_1h
    df["smz_trend_4h"] = trend_4h
    df["smz_trend_1d"] = trend_1d
    return df


def add_external_liquidity_targets(df: pd.DataFrame) -> pd.DataFrame:
    smc = _cache.get_smc_tradingfinder(df)
    liquidity_levels = smc["liquidity"]
    smc_liquidity_high = liquidity_levels.static_high.combine_first(liquidity_levels.dynamic_high)
    smc_liquidity_low = liquidity_levels.static_low.combine_first(liquidity_levels.dynamic_low)

    liquidity = calculate_liquidity_inducements(df)
    buyside_targets = liquidity["buyside_targets"]
    sellside_targets = liquidity["sellside_targets"]

    df = df.copy()
    df["smc_liquidity_high"] = smc_liquidity_high
    df["smc_liquidity_low"] = smc_liquidity_low
    df["liq_buyside_target"] = buyside_targets
    df["liq_sellside_target"] = sellside_targets
    return df


def _align_index_to_hk_as_ny(df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    index = index.tz_convert("Asia/Hong_Kong")
    index = index.tz_localize(None).tz_localize("America/New_York")
    aligned = df.copy()
    aligned.index = index
    return aligned


def add_killzone_windows(df: pd.DataFrame, hk_aligned: pd.DataFrame) -> pd.DataFrame:
    sb_signals = _cache.get_silver_bullet_signals(hk_aligned)
    sb_sig_ln = pd.Series(0, index=df.index, dtype=int)
    sb_sig_am = pd.Series(0, index=df.index, dtype=int)
    sb_sig_pm = pd.Series(0, index=df.index, dtype=int)
    for session in sb_signals["sessions"]:
        if session.name == "LN":
            sb_sig_ln.iloc[session.start_idx : session.end_idx + 1] = 1
        elif session.name == "AM":
            sb_sig_am.iloc[session.start_idx : session.end_idx + 1] = 1
        elif session.name == "PM":
            sb_sig_pm.iloc[session.start_idx : session.end_idx + 1] = 1

    df = df.copy()
    df["sb_sig_ln"] = sb_sig_ln
    df["sb_sig_am"] = sb_sig_am
    df["sb_sig_pm"] = sb_sig_pm
    return df


def add_ict_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    config = ict_sessions_module.IndicatorConfig()
    index = pd.to_datetime(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    ny_index = index.tz_convert(config.timezone)

    asia_start, asia_end = ict_sessions_module._parse_session(config.asia_session)
    london_start, london_end = ict_sessions_module._parse_session(config.europe_session)
    ny_am_start, ny_am_end = ict_sessions_module._parse_session(config.usa_session)
    ny_pm_start, ny_pm_end = ict_sessions_module._parse_session(config.usa2_session)

    session_active = [
        int(
            ict_sessions_module._in_session(ts, asia_start, asia_end)
            or ict_sessions_module._in_session(ts, london_start, london_end)
            or ict_sessions_module._in_session(ts, ny_am_start, ny_am_end)
            or ict_sessions_module._in_session(ts, ny_pm_start, ny_pm_end)
        )
        for ts in ny_index
    ]

    df = df.copy()
    df["ict_session_active"] = session_active
    return df


def _resample_rule_from_resolution(res: str) -> str:
    if res.endswith("D") or res.endswith("W") or res.endswith("M"):
        return res
    return f"{res}min"


def _compute_htf_zones(df: pd.DataFrame, *, resolution: str) -> Dict[pd.Timestamp, list]:
    """Compute HTF order block zones, accumulating them across HTF bars.

    For each HTF bar, detect new order blocks via ``compute_mtf_order_block_finder``
    and add them to a running list.  Channel limits (``bull_channels`` /
    ``bear_channels``) cap the number of active zones per side.  Zones are
    invalidated when the HTF close breaks through them (bullish OB broken when
    close < zone low; bearish OB broken when close > zone high).
    """
    rule = _resample_rule_from_resolution(resolution)
    htf = (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    settings = OBSettings(resolution=resolution)
    accumulated_bull: list = []
    accumulated_bear: list = []
    zones_by_ts: Dict[pd.Timestamp, list] = {}
    for ts in htf.index:
        new_zones = compute_mtf_order_block_finder(df.loc[:ts], settings=settings)["zones"]
        for z in new_zones:
            if z.direction == "bull":
                accumulated_bull.append(z)
                if len(accumulated_bull) > settings.bull_channels:
                    accumulated_bull = accumulated_bull[-settings.bull_channels:]
            else:
                accumulated_bear.append(z)
                if len(accumulated_bear) > settings.bear_channels:
                    accumulated_bear = accumulated_bear[-settings.bear_channels:]

        # Invalidate mitigated zones: bull OB broken when close < low,
        # bear OB broken when close > high.
        close = float(htf.loc[ts, "close"])
        accumulated_bull = [z for z in accumulated_bull if close >= z.low]
        accumulated_bear = [z for z in accumulated_bear if close <= z.high]

        zones_by_ts[ts] = list(accumulated_bull + accumulated_bear)
    return zones_by_ts


def _zones_to_in_zone_series(df: pd.DataFrame, zones_by_ts: Dict[pd.Timestamp, list], *, side: str) -> pd.Series:
    series = pd.Series(False, index=df.index)
    sorted_items = sorted(zones_by_ts.items())
    current_zones: list = []
    next_idx = 0
    for i, ts in enumerate(df.index):
        while next_idx < len(sorted_items) and ts >= sorted_items[next_idx][0]:
            current_zones = sorted_items[next_idx][1]
            next_idx += 1
        if not current_zones:
            continue
        low = df.iloc[i]["low"]
        high = df.iloc[i]["high"]
        for zone in current_zones:
            if zone.direction != side:
                continue
            if low <= zone.high and high >= zone.low:
                series.iat[i] = True
                break
    return series


def add_htf_ob_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Build HTF POI filters from 1H/4H order blocks (mitigated or not).

    A side's filter is true when at least one of the *previous* 10 bars
    traded inside a same-side 1H or 4H order block.  All detected blocks
    are considered — mitigation status is not required.

    Zones are still **removed** when price structurally breaks through them
    (bullish OB: close < zone bottom; bearish OB: close > zone top), since
    those zones are invalidated and no longer meaningful.
    """

    def _compute_in_ob_series(data: pd.DataFrame, timeframe: str) -> tuple[pd.Series, pd.Series]:
        settings = OBImbalanceSettings(timeframe=timeframe, mitigation_type="Wick")
        rule = ob_imbalance_module._resolve_timeframe_rule(data, settings.timeframe)
        if rule is None:
            return pd.Series(False, index=data.index), pd.Series(False, index=data.index)
        htf = (
            data.resample(rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )

        in_bull = pd.Series(False, index=data.index)
        in_bear = pd.Series(False, index=data.index)
        if htf.empty:
            return in_bull, in_bear

        atr = ob_imbalance_module._atr(htf, 14)
        htf_calc = pd.DataFrame(
            {
                "is_bull": (htf["low"] - htf["high"].shift(2)) > (atr.shift(1) * settings.fvg_threshold),
                "is_bear": (htf["low"].shift(2) - htf["high"]) > (atr.shift(1) * settings.fvg_threshold),
                "high_shift2": htf["high"].shift(2),
                "low_shift2": htf["low"].shift(2),
                "time_shift2": htf.index.to_series().shift(2),
            }
        )
        htf_aligned = htf_calc.reindex(data.index, method="ffill")

        zones: list[dict] = []
        last_bull_created_time: pd.Timestamp | None = None
        last_bear_created_time: pd.Timestamp | None = None

        for i, (ts, row) in enumerate(data.iterrows()):
            htf_row = htf_aligned.loc[ts]
            htf_time = htf_row["time_shift2"]

            # --- Zone creation (bull and bear checked independently) ---
            if pd.notna(htf_time):
                if bool(htf_row["is_bull"]) and htf_time != last_bull_created_time:
                    zones.append(
                        {
                            "top": float(htf_row["high_shift2"]),
                            "bottom": float(htf_row["low_shift2"]),
                            "is_bullish": True,
                        }
                    )
                    last_bull_created_time = htf_time
                if bool(htf_row["is_bear"]) and htf_time != last_bear_created_time:
                    zones.append(
                        {
                            "top": float(htf_row["high_shift2"]),
                            "bottom": float(htf_row["low_shift2"]),
                            "is_bullish": False,
                        }
                    )
                    last_bear_created_time = htf_time

            # --- Zone invalidation (remove structurally broken zones) ---
            close_price = float(row["close"])
            low_price = float(row["low"])
            high_price = float(row["high"])
            remove_indices: list[int] = []
            for idx in range(len(zones) - 1, -1, -1):
                zone = zones[idx]
                if zone["is_bullish"]:
                    if close_price < zone["bottom"]:
                        remove_indices.append(idx)
                else:
                    if close_price > zone["top"]:
                        remove_indices.append(idx)

            for idx in remove_indices:
                zones.pop(idx)

            if len(zones) > 450:
                zones.pop(0)

            # --- In-zone check (all surviving zones count) ---
            for zone in zones:
                in_zone = low_price <= zone["top"] and high_price >= zone["bottom"]
                if not in_zone:
                    continue
                if zone["is_bullish"]:
                    in_bull.iat[i] = True
                else:
                    in_bear.iat[i] = True

        return in_bull, in_bear

    df = df.copy()
    in_bull_1h, in_bear_1h = _compute_in_ob_series(df, timeframe="1H")
    in_bull_4h, in_bear_4h = _compute_in_ob_series(df, timeframe="4H")

    lookback_bars = 10
    recent_ob_bull = (
        (in_bull_1h | in_bull_4h).shift(1).fillna(False).rolling(lookback_bars, min_periods=1).max() > 0
    )
    recent_ob_bear = (
        (in_bear_1h | in_bear_4h).shift(1).fillna(False).rolling(lookback_bars, min_periods=1).max() > 0
    )

    df["filter_htf_ob_bull"] = recent_ob_bull.astype(int)
    df["filter_htf_ob_bear"] = recent_ob_bear.astype(int)
    df["filter_htf_poi_bull"] = recent_ob_bull.astype(int)
    df["filter_htf_poi_bear"] = recent_ob_bear.astype(int)
    return df


def add_htf_fvg_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Build HTF FVG filter from 1H/4H uninvalidated Fair Value Gaps.

    Uses MTF FVG x2 [MK] detection logic.  A side's filter is true when at
    least one of the *previous* 10 bars traded inside a same-side 1H or 4H
    FVG that has not been invalidated.

    Invalidation (Normal mode, wicks — matching the reference module):
    - Bullish FVG is **removed** when low < zone bottom.
    - Bearish FVG is **removed** when high > zone top.

    The result is OR-combined with the existing OB-based HTF POI filter so
    that ``filter_htf_poi_bull/bear`` passes if *either* an OB or an FVG
    was touched in the lookback window.
    """

    def _compute_in_fvg_series(data: pd.DataFrame, timeframe: str) -> tuple[pd.Series, pd.Series]:
        import math

        rule = mtf_fvg_module._tf_to_rule(timeframe)
        # _resample_ohlc requires a volume column
        data_with_vol = data.copy()
        if "volume" not in data_with_vol.columns:
            data_with_vol["volume"] = 0
        htf = mtf_fvg_module._resample_ohlc(data_with_vol, rule)

        in_bull = pd.Series(False, index=data.index)
        in_bear = pd.Series(False, index=data.index)
        if htf.empty:
            return in_bull, in_bear

        # Shift on the HTF series BEFORE ffill-alignment to base timeframe.
        # Pine's request.security(sym, tf, high[1]) returns the previous
        # completed HTF bar's high — that is a 1-HTF-bar shift, not a
        # 1-base-bar shift.  Shifting on the aligned series would only move
        # by 5 minutes (1 base bar) and almost never produce a gap.
        h_shift1 = htf["high"].shift(1).reindex(data.index, method="ffill")
        h_shift3 = htf["high"].shift(3).reindex(data.index, method="ffill")
        l_shift1 = htf["low"].shift(1).reindex(data.index, method="ffill")
        l_shift3 = htf["low"].shift(3).reindex(data.index, method="ffill")

        zones: list[dict] = []

        for i in range(len(data)):
            low_now = float(data["low"].iloc[i])
            high_now = float(data["high"].iloc[i])

            # --- FVG detection (requires >= 3 prior aligned bars) ---
            if i >= 3:
                h2 = float(h_shift3.iloc[i])
                l_ = float(l_shift1.iloc[i])
                l2 = float(l_shift3.iloc[i])
                h_ = float(h_shift1.iloc[i])

                prev_h2 = float(h_shift3.iloc[i - 1])
                prev_l = float(l_shift1.iloc[i - 1])
                prev_l2 = float(l_shift3.iloc[i - 1])
                prev_h = float(h_shift1.iloc[i - 1])

                vals_ok = not any(math.isnan(x) for x in (h_, h2, l_, l2))
                if vals_ok:
                    # Bull FVG: gap between 3-bars-ago high and 1-bar-ago low
                    new_bull = h2 < l_
                    if new_bull:
                        prev_ok = not (math.isnan(prev_h2) or math.isnan(prev_l))
                        changed = prev_ok and (h2 != prev_h2 or l_ != prev_l)
                        if changed:
                            zones.append({"top": l_, "bottom": h2, "is_bullish": True})

                    # Bear FVG: gap between 3-bars-ago low and 1-bar-ago high
                    new_bear = l2 > h_
                    if new_bear:
                        prev_ok = not (math.isnan(prev_l2) or math.isnan(prev_h))
                        changed = prev_ok and (l2 != prev_l2 or h_ != prev_h)
                        if changed:
                            zones.append({"top": l2, "bottom": h_, "is_bullish": False})

            # --- Invalidation (Normal mode, wicks) ---
            remove_indices: list[int] = []
            for idx in range(len(zones) - 1, -1, -1):
                zone = zones[idx]
                if zone["is_bullish"]:
                    if low_now < zone["bottom"]:
                        remove_indices.append(idx)
                else:
                    if high_now > zone["top"]:
                        remove_indices.append(idx)

            for idx in remove_indices:
                zones.pop(idx)

            if len(zones) > 450:
                zones.pop(0)

            # --- In-zone check (all surviving / uninvalidated zones) ---
            for zone in zones:
                in_zone = low_now <= zone["top"] and high_now >= zone["bottom"]
                if not in_zone:
                    continue
                if zone["is_bullish"]:
                    in_bull.iat[i] = True
                else:
                    in_bear.iat[i] = True

        return in_bull, in_bear

    df = df.copy()
    in_bull_1h, in_bear_1h = _compute_in_fvg_series(df, timeframe="60")
    in_bull_4h, in_bear_4h = _compute_in_fvg_series(df, timeframe="240")

    lookback_bars = 10
    recent_fvg_bull = (
        (in_bull_1h | in_bull_4h).shift(1).fillna(False).rolling(lookback_bars, min_periods=1).max() > 0
    )
    recent_fvg_bear = (
        (in_bear_1h | in_bear_4h).shift(1).fillna(False).rolling(lookback_bars, min_periods=1).max() > 0
    )

    df["filter_htf_fvg_bull"] = recent_fvg_bull.astype(int)
    df["filter_htf_fvg_bear"] = recent_fvg_bear.astype(int)

    # Combine with existing OB filter: POI passes if EITHER OB or FVG touched
    ob_bull = df.get("filter_htf_ob_bull", pd.Series(0, index=df.index)).astype(bool)
    ob_bear = df.get("filter_htf_ob_bear", pd.Series(0, index=df.index)).astype(bool)
    df["filter_htf_poi_bull"] = (ob_bull | recent_fvg_bull).astype(int)
    df["filter_htf_poi_bear"] = (ob_bear | recent_fvg_bear).astype(int)
    return df


def add_entry_signals(df: pd.DataFrame, hk_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry signals with a two-stage approach:
    1. First detect raw entry triggers (SB FVG retrace, ICT Setup01, Fibonacci OTE)
    2. Store filter states separately (HTF POI, Trend, Time filters)

    The strategy will check triggers first, then apply filters at execution time.
    """
    sb = _cache.get_silver_bullet_signals(hk_aligned)
    sb_signals = sb["signals"]
    sb_entry_bull = pd.Series(sb_signals["bull_fvg_retrace"].to_numpy(), index=df.index).astype(int)
    sb_entry_bear = pd.Series(sb_signals["bear_fvg_retrace"].to_numpy(), index=df.index).astype(int)

    if "Long" in df.columns or "Short" in df.columns:
        setup01_bull = df.get("Long", pd.Series(0, index=df.index)).fillna(0).astype(int)
        setup01_bear = df.get("Short", pd.Series(0, index=df.index)).fillna(0).astype(int)
    else:
        setup01 = calculate_setup_01(df)
        setup01_bull = pd.Series(0, index=df.index, dtype=int)
        setup01_bear = pd.Series(0, index=df.index, dtype=int)
        for signal in setup01["signals"]:
            if 0 <= signal.index < len(setup01_bull):
                if signal.long_signal:
                    setup01_bull.iloc[signal.index] = 1
                if signal.short_signal:
                    setup01_bear.iloc[signal.index] = 1

    fib = calculate_fibonacci_ote(df)
    ote_bull = pd.Series(0, index=df.index, dtype=int)
    ote_bear = pd.Series(0, index=df.index, dtype=int)
    for idx, state in enumerate(fib["states"]):
        if idx >= len(df):
            break
        levels = state.levels
        if not levels or any(pd.isna(level) for level in levels):
            continue
        lower, upper = sorted(levels[:2])
        low = float(df["low"].iloc[idx])
        high = float(df["high"].iloc[idx])
        if low <= upper and high >= lower:
            if state.pos > 0:
                ote_bull.iloc[idx] = 1
            elif state.pos < 0:
                ote_bear.iloc[idx] = 1

    ifvg_signals, _ = compute_ifvg_realtime(
        df,
        mintick=0.01,
        pip_size_multiplier=1.0,
        ifvg_gap_bars=15,
        min_fvg_pips=0.0,
        fvg_eps_points=0.0,
        show_zones=False,
        ma_period=21,
        ma_kind="EMA",
    )
    ifvg_bull = ifvg_signals["buy_signal"].fillna(False).astype(int)
    ifvg_bear = ifvg_signals["sell_signal"].fillna(False).astype(int)

    # Order-Block Detector (OB mitigation + FVG mitigation signals)
    ob_det_results = _cache.get_ob_detector_results(df)
    ob_det_signals = ob_det_results["signals"]
    obdet_bull = (
        (ob_det_signals["ob_signal"] == 1) | (ob_det_signals["fvg_signal"] == 1)
    ).astype(int)
    obdet_bear = (
        (ob_det_signals["ob_signal"] == -1) | (ob_det_signals["fvg_signal"] == -1)
    ).astype(int)

    # Raw entry triggers (unfiltered) - checked FIRST in strategy
    entry_trigger_bull = (
        sb_entry_bull.astype(bool)
        | setup01_bull.astype(bool)
        | ote_bull.astype(bool)
        | ifvg_bull.astype(bool)
        | obdet_bull.astype(bool)
    ).astype(int)
    entry_trigger_bear = (
        sb_entry_bear.astype(bool)
        | setup01_bear.astype(bool)
        | ote_bear.astype(bool)
        | ifvg_bear.astype(bool)
        | obdet_bear.astype(bool)
    ).astype(int)

    # Filter 1: HTF POI Filter (1H/4H unmitigated order block lookback)
    htf_poi_bull = df.get("filter_htf_poi_bull", pd.Series(1, index=df.index)).astype(int)
    htf_poi_bear = df.get("filter_htf_poi_bear", pd.Series(1, index=df.index)).astype(int)

    # Filter 2: Trend Filter - SMZ OR Market Structure (15M/1H) must agree
    smz_trend_15m = df.get("smz_trend_15m", pd.Series(0, index=df.index)).astype(int)
    smz_trend_1h = df.get("smz_trend_1h", pd.Series(0, index=df.index)).astype(int)

    market_structure = calculate_market_structure_mtf(df)
    ms_tf15 = market_structure.tf1
    ms_tf1h = market_structure.tf3

    bull_colors = {"green", (46, 104, 48)}
    bear_colors = {"red", (128, 41, 41)}

    ms_bull_15m = ms_tf15.color.apply(lambda value: value in bull_colors).fillna(False)
    ms_bear_15m = ms_tf15.color.apply(lambda value: value in bear_colors).fillna(False)
    ms_bull_1h = ms_tf1h.color.apply(lambda value: value in bull_colors).fillna(False)
    ms_bear_1h = ms_tf1h.color.apply(lambda value: value in bear_colors).fillna(False)

    htf_bias_bull = (
        (smz_trend_15m == 1) | ms_bull_15m.astype(bool)
    ) & (
        (smz_trend_1h == 1) | ms_bull_1h.astype(bool)
    )
    htf_bias_bear = (
        (smz_trend_15m == -1) | ms_bear_15m.astype(bool)
    ) & (
        (smz_trend_1h == -1) | ms_bear_1h.astype(bool)
    )
    htf_bias_bull = htf_bias_bull.astype(int)
    htf_bias_bear = htf_bias_bear.astype(int)

    # Filter 3: Time Filter - ICT Session must be active
    session_active = df.get("ict_session_active", pd.Series(1, index=df.index)).astype(int)

    # Legacy combined entry signals (for backward compatibility)
    # Note: HTF POI is now enforced here as well so these columns mirror
    # the strategy's trigger->filter flow.
    entry_fvg_bull = (
        entry_trigger_bull.astype(bool)
        & htf_poi_bull.astype(bool)
        & session_active.astype(bool)
        & htf_bias_bull.astype(bool)
    ).astype(int)
    entry_fvg_bear = (
        entry_trigger_bear.astype(bool)
        & htf_poi_bear.astype(bool)
        & session_active.astype(bool)
        & htf_bias_bear.astype(bool)
    ).astype(int)

    df = df.copy()
    # Individual entry trigger signals
    df["entry_sb_bull"] = sb_entry_bull
    df["entry_sb_bear"] = sb_entry_bear
    df["entry_setup01_bull"] = setup01_bull
    df["entry_setup01_bear"] = setup01_bear
    df["entry_ote_bull"] = ote_bull
    df["entry_ote_bear"] = ote_bear
    df["entry_ifvg_bull"] = ifvg_bull
    df["entry_ifvg_bear"] = ifvg_bear
    df["entry_obdet_bull"] = obdet_bull
    df["entry_obdet_bear"] = obdet_bear
    # Raw combined triggers (unfiltered)
    df["entry_trigger_bull"] = entry_trigger_bull
    df["entry_trigger_bear"] = entry_trigger_bear
    # Filter state columns
    df["filter_session_active"] = session_active
    df["filter_htf_bias_bull"] = htf_bias_bull
    df["filter_htf_bias_bear"] = htf_bias_bear
    df["filter_htf_poi_bull"] = htf_poi_bull
    df["filter_htf_poi_bear"] = htf_poi_bear
    # Final filtered entry signals
    df["entry_fvg_bull"] = entry_fvg_bull
    df["entry_fvg_bear"] = entry_fvg_bear
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )




def _export_rejected_triggers_report(rejected_triggers: list[dict], output_csv: str = "backtest_rejected_triggers.csv") -> None:
    if not rejected_triggers:
        print("No rejected entry triggers found.")
        return

    rejected_df = pd.DataFrame(rejected_triggers).copy()
    rejected_df["time_utc"] = pd.to_datetime(rejected_df["time"], utc=True)
    rejected_df["time_hk"] = rejected_df["time_utc"].dt.tz_convert(ZoneInfo("Asia/Hong_Kong"))
    rejected_df["time_hk"] = rejected_df["time_hk"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    rejected_df = rejected_df[[
        "time_utc",
        "time_hk",
        "direction",
        "signal_type",
        "rejection_reason",
        "filter_htf_poi",
        "filter_trend",
        "filter_time",
    ]]
    rejected_df.to_csv(output_csv, index=False)

    summary = rejected_df.groupby(["direction", "rejection_reason"]).size().sort_values(ascending=False)
    print("\n" + "=" * 80)
    print("REJECTED ENTRY TRIGGERS (ALL)")
    print("=" * 80)
    print(f"Total rejected triggers: {len(rejected_df)}")
    print("\nBy direction + filter reason:")
    for (direction, reason), count in summary.items():
        print(f"- {direction} | {reason}: {count}")
    print(f"\nDetailed rejected trigger CSV exported: {output_csv}")

def run_backtest(
    csv_file: str,
    max_bars: int | None = None,
    export_csv: str | None = None,
    fast_mode: bool = False,
) -> None:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Clear the indicator cache at the start of each backtest
    _cache.clear()

    max_rows = _env_int("SB_MAX_ROWS")
    debug_signals = _env_bool("SB_DEBUG_SIGNALS", default=False)

    data_df = add_smart_money_trends(load_data(csv_file, max_rows=max_rows))

    # Create HK-aligned DataFrame once for all functions that need it
    hk_aligned = _align_index_to_hk_as_ny(data_df)

    data_df = add_external_liquidity_targets(data_df)
    data_df = add_killzone_windows(data_df, hk_aligned)
    data_df = add_ict_session_filter(data_df)
    data_df = add_htf_ob_filter(data_df)
    data_df = add_htf_fvg_filter(data_df)
    data_df = add_entry_signals(data_df, hk_aligned)
    if export_csv:
        data_df.to_csv(export_csv)
    data_4h_df = resample_ohlc(data_df, "4h")
    data_1d_df = resample_ohlc(data_df, "1D")

    data = PandasDataBias(dataname=data_df)
    data_4h = bt.feeds.PandasData(dataname=data_4h_df)
    data_1d = bt.feeds.PandasData(dataname=data_1d_df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        SilverBulletStrategy,
        print_trades=True,
        pivot_strength=15,
        risk_per_trade=0.02,
        leverage=100,  # 1:100 leverage for proper 2% risk sizing
        debug_signals=debug_signals,
    )
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.adddata(data)
    cerebro.adddata(data_4h)
    cerebro.adddata(data_1d)
    cerebro.broker.setcash(10000.0)

    print("\n" + "=" * 80)
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print("=" * 80 + "\n")

    results = cerebro.run()
    strategy = results[0]
    _export_rejected_triggers_report(getattr(strategy, "rejected_triggers", []))

    analyzer = strategy.analyzers.trade_analyzer.get_analysis()
    total_trades = analyzer.get("total", {}).get("closed", 0)
    total_won = analyzer.get("won", {}).get("total", 0)
    total_lost = analyzer.get("lost", {}).get("total", 0)

    print("\n" + "=" * 80)
    print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print(
        "Closed Trades: {total} | Wins: {wins} | Losses: {losses}".format(
            total=total_trades,
            wins=total_won,
            losses=total_lost,
        )
    )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Silver Bullet backtrader prep.")
    parser.add_argument(
        "--csv-file",
        default="PEPPERSTONE_XAUUSD, 5.csv",
        help="Path to CSV file containing OHLC data.",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=int(os.environ.get("SILVER_BULLET_MAX_BARS", "0")),
        help="Limit the number of most recent bars processed (0 = no limit).",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="Optional path to write the enriched indicator dataframe.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast mode: skip non-essential indicators for quicker backtesting.",
    )
    args = parser.parse_args()
    max_bars = args.max_bars if args.max_bars > 0 else None
    export_csv = args.export_csv if args.export_csv.strip() else None
    run_backtest(args.csv_file, max_bars=max_bars, export_csv=export_csv, fast_mode=args.fast_mode)


if __name__ == "__main__":
    main()
