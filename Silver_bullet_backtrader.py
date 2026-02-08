"""
Backtest script to compute HTF market structure bias using Backtrader.

Steps:
1) Load OHLC data from CSV and set a datetime index.
2) Compute Smart Money Zones MTF trends (4H/1D) with pandas.
3) Attach Smart Money trend columns to the base dataframe.
4) Identify external liquidity targets (profit targets) via SMC + inducements.
5) Detect liquidity sweeps using inducements and HTF sweeps.
6) Confirm MSS/CHOCH via market structure + CISD sweep signals.
7) Identify FVGs formed during MSS displacement (Silver Bullet/SMZ).
8) Trigger entry signals at FVG zones (Silver Bullet/TradingFinder/Fib OTE).
9) Build stop-loss placement levels from sweep/liquidity targets.
10) Identify target levels from liquidity targets and HTF sweeps.
11) Feed data into Backtrader and resample to 4H/1D.
12) Run HTFBiasStrategy to log consolidated HTF bias.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict

import backtrader as bt
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))

from htf_bias_strategy import HTFBiasStrategy, SilverBulletStrategy
from ICT_Silver_Bullet_with_signals import detect_silver_bullet_signals
from Smart_Money_Concept__TradingFinder__Major_Minor_OB___FVG__SMC_ import calculate_smc_tradingfinder
from Smart_Money_Zones__FVG___OB____MTF_Trend_Panel import calculate_smart_money_zones

LIQUIDITY_PATH = os.path.join(
    os.path.dirname(__file__),
    "Liquidity & inducements.py",
)
liquidity_module = SourceFileLoader("liquidity_inducements", LIQUIDITY_PATH).load_module()
calculate_liquidity_inducements = liquidity_module.calculate_liquidity_inducements

LUXALGO_SB_PATH = os.path.join(
    os.path.dirname(__file__),
    "ICT_Silver_Bullet__LuxAlgo___shorttitle__LuxAlgo_-_ICT_Silver_Bullet.py",
)
luxalgo_module = SourceFileLoader("luxalgo_silver_bullet", LUXALGO_SB_PATH).load_module()
calculate_luxalgo_silver_bullet = luxalgo_module.calculate_luxalgo_silver_bullet

TRADINGFINDER_SB_PATH = os.path.join(
    os.path.dirname(__file__),
    "Silver_Bullet_ICT_Strategy__TradingFinder__10-11_AM_NY_Time__FVG_TFlab_Silver_Bullet.py",
)
tradingfinder_module = SourceFileLoader("tradingfinder_silver_bullet", TRADINGFINDER_SB_PATH).load_module()
calculate_tradingfinder_silver_bullet = tradingfinder_module.calculate_tradingfinder_silver_bullet

HTF_SWEEPS_PATH = os.path.join(
    os.path.dirname(__file__),
    "CandelaCharts - HTF Sweeps.py",
)
htf_sweeps_module = SourceFileLoader("candela_htf_sweeps", HTF_SWEEPS_PATH).load_module()
calculate_htf_sweeps = htf_sweeps_module.calculate_htf_sweeps

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

CD_SWEEP_CISD_PATH = os.path.join(
    os.path.dirname(__file__),
    "cd_sweep&cisd_Cx.py",
)
cd_sweep_module = SourceFileLoader("cd_sweep_cisd", CD_SWEEP_CISD_PATH).load_module()
detect_cd_sweep_cisd = cd_sweep_module.detect_cd_sweep_cisd

BIGBELUGA_SMC_PATH = os.path.join(
    os.path.dirname(__file__),
    "BigBeluga_SMC.py",
)
bigbeluga_module = SourceFileLoader("bigbeluga_smc", BIGBELUGA_SMC_PATH).load_module()
calculate_bigbeluga_smc = bigbeluga_module.calculate_bigbeluga_smc

MTF_OB_FINDER_PATH = os.path.join(
    os.path.dirname(__file__),
    "MTF Order Block Finder.py",
)
mtf_ob_module = SourceFileLoader("mtf_order_block_finder", MTF_OB_FINDER_PATH).load_module()
compute_mtf_order_block_finder = mtf_ob_module.compute_mtf_order_block_finder
OBSettings = mtf_ob_module.OBSettings


# =============================================================================
# INDICATOR CACHE - Stores expensive calculations to avoid redundant computation
# =============================================================================
class IndicatorCache:
    """Cache for expensive indicator calculations to avoid redundant computation."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._htf_sweeps = None
        self._silver_bullet_signals = None
        self._smc_tradingfinder = None
        self._smart_money_zones = None
        self._hk_aligned_df = None

    def get_htf_sweeps(self, df: pd.DataFrame) -> dict:
        if self._htf_sweeps is None:
            self._htf_sweeps = calculate_htf_sweeps(
                df,
                timeframes=[
                    ("4H", 200, True),
                    ("1D", 200, True),
                ],
            )
        return self._htf_sweeps

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


# Global cache instance
_cache = IndicatorCache()


class PandasDataBias(bt.feeds.PandasData):
    lines = (
        "smz_trend_15m",
        "smz_trend_4h",
        "smz_trend_1h",
        "smz_trend_1d",
        "bb_pivot_high",
        "bb_pivot_low",
        "smc_liquidity_high",
        "smc_liquidity_low",
        "liq_buyside_target",
        "liq_sellside_target",
        "sb_sig_ln",
        "sb_sig_am",
        "sb_sig_pm",
        "sb_lux_ln",
        "sb_lux_am",
        "sb_lux_pm",
        "sb_or_range",
        "sb_trading_range",
        "htf_sweep_4h_bull",
        "htf_sweep_4h_bear",
        "htf_sweep_1d_bull",
        "htf_sweep_1d_bear",
        "liquidity_sweep_bull",
        "liquidity_sweep_bear",
        "choch_bull",
        "choch_bear",
        "cisd_bull",
        "cisd_bear",
        "mss_bull",
        "mss_bear",
        "fvg_sb_bull",
        "fvg_sb_bear",
        "fvg_smz_bull",
        "fvg_smz_bear",
        "mss_fvg_bull",
        "mss_fvg_bear",
        "entry_sb_bull",
        "entry_sb_bear",
        "entry_setup01_bull",
        "entry_setup01_bear",
        "entry_ote_bull",
        "entry_ote_bear",
        "entry_trigger_bull",
        "entry_trigger_bear",
        "filter_session_active",
        "filter_htf_bias_bull",
        "filter_htf_bias_bear",
        "filter_htf_poi_bull",
        "filter_htf_poi_bear",
        "entry_fvg_bull",
        "entry_fvg_bear",
        "stop_loss_bull",
        "stop_loss_bear",
        "target_bull",
        "target_bear",
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
        ("bb_pivot_high", "bb_pivot_high"),
        ("bb_pivot_low", "bb_pivot_low"),
        ("smc_liquidity_high", "smc_liquidity_high"),
        ("smc_liquidity_low", "smc_liquidity_low"),
        ("liq_buyside_target", "liq_buyside_target"),
        ("liq_sellside_target", "liq_sellside_target"),
        ("sb_sig_ln", "sb_sig_ln"),
        ("sb_sig_am", "sb_sig_am"),
        ("sb_sig_pm", "sb_sig_pm"),
        ("sb_lux_ln", "sb_lux_ln"),
        ("sb_lux_am", "sb_lux_am"),
        ("sb_lux_pm", "sb_lux_pm"),
        ("sb_or_range", "sb_or_range"),
        ("sb_trading_range", "sb_trading_range"),
        ("htf_sweep_4h_bull", "htf_sweep_4h_bull"),
        ("htf_sweep_4h_bear", "htf_sweep_4h_bear"),
        ("htf_sweep_1d_bull", "htf_sweep_1d_bull"),
        ("htf_sweep_1d_bear", "htf_sweep_1d_bear"),
        ("liquidity_sweep_bull", "liquidity_sweep_bull"),
        ("liquidity_sweep_bear", "liquidity_sweep_bear"),
        ("choch_bull", "choch_bull"),
        ("choch_bear", "choch_bear"),
        ("cisd_bull", "cisd_bull"),
        ("cisd_bear", "cisd_bear"),
        ("mss_bull", "mss_bull"),
        ("mss_bear", "mss_bear"),
        ("fvg_sb_bull", "fvg_sb_bull"),
        ("fvg_sb_bear", "fvg_sb_bear"),
        ("fvg_smz_bull", "fvg_smz_bull"),
        ("fvg_smz_bear", "fvg_smz_bear"),
        ("mss_fvg_bull", "mss_fvg_bull"),
        ("mss_fvg_bear", "mss_fvg_bear"),
        ("entry_sb_bull", "entry_sb_bull"),
        ("entry_sb_bear", "entry_sb_bear"),
        ("entry_setup01_bull", "entry_setup01_bull"),
        ("entry_setup01_bear", "entry_setup01_bear"),
        ("entry_ote_bull", "entry_ote_bull"),
        ("entry_ote_bear", "entry_ote_bear"),
        ("entry_trigger_bull", "entry_trigger_bull"),
        ("entry_trigger_bear", "entry_trigger_bear"),
        ("filter_session_active", "filter_session_active"),
        ("filter_htf_bias_bull", "filter_htf_bias_bull"),
        ("filter_htf_bias_bear", "filter_htf_bias_bear"),
        ("filter_htf_poi_bull", "filter_htf_poi_bull"),
        ("filter_htf_poi_bear", "filter_htf_poi_bear"),
        ("entry_fvg_bull", "entry_fvg_bull"),
        ("entry_fvg_bear", "entry_fvg_bear"),
        ("stop_loss_bull", "stop_loss_bull"),
        ("stop_loss_bear", "stop_loss_bear"),
        ("target_bull", "target_bull"),
        ("target_bear", "target_bear"),
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
    df = df[["open", "high", "low", "close"]].dropna()
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


def add_bigbeluga_pivots(df: pd.DataFrame) -> pd.DataFrame:
    outputs = calculate_bigbeluga_smc(df)
    pivot_high = outputs.swing_highs.ffill()
    pivot_low = outputs.swing_lows.ffill()

    df = df.copy()
    df["bb_pivot_high"] = pivot_high
    df["bb_pivot_low"] = pivot_low
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

    lux = calculate_luxalgo_silver_bullet(hk_aligned)
    sb_lux_ln = pd.Series(0, index=df.index, dtype=int)
    sb_lux_am = pd.Series(0, index=df.index, dtype=int)
    sb_lux_pm = pd.Series(0, index=df.index, dtype=int)
    for state in lux["bar_states"]:
        idx = state.index
        if state.in_ln:
            sb_lux_ln.iloc[idx] = 1
        if state.in_am:
            sb_lux_am.iloc[idx] = 1
        if state.in_pm:
            sb_lux_pm.iloc[idx] = 1

    tradingfinder = calculate_tradingfinder_silver_bullet(hk_aligned)
    session_levels = tradingfinder["session_levels"]
    sb_or_range = pd.Series(session_levels.or_range.to_numpy(), index=df.index).fillna(0).astype(int)
    sb_trading_range = (
        pd.Series(session_levels.trading_range.to_numpy(), index=df.index).fillna(0).astype(int)
    )

    df = df.copy()
    df["sb_sig_ln"] = sb_sig_ln
    df["sb_sig_am"] = sb_sig_am
    df["sb_sig_pm"] = sb_sig_pm
    df["sb_lux_ln"] = sb_lux_ln
    df["sb_lux_am"] = sb_lux_am
    df["sb_lux_pm"] = sb_lux_pm
    df["sb_or_range"] = sb_or_range
    df["sb_trading_range"] = sb_trading_range
    return df


def _build_sweep_series(candles: list, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    bull = pd.Series(0, index=index, dtype=int)
    bear = pd.Series(0, index=index, dtype=int)
    for candle in candles:
        for sweep in candle.htf_sweeps:
            if not sweep.formed or sweep.removed:
                continue
            idx = sweep.x2
            if 0 <= idx < len(index):
                if sweep.bull:
                    bull.iloc[idx] = 1
                else:
                    bear.iloc[idx] = 1
    return bull, bear


def _build_sweep_price_series(candles: list, index: pd.Index, bullish: bool) -> pd.Series:
    series = pd.Series(0.0, index=index, dtype=float)
    for candle in candles:
        for sweep in candle.htf_sweeps:
            if not sweep.formed or sweep.removed or sweep.bull != bullish:
                continue
            idx = sweep.x2
            if 0 <= idx < len(series):
                series.iloc[idx] = float(sweep.y)
    return series


def _bool_series_from_list(values: list[bool], index: pd.Index) -> pd.Series:
    series = pd.Series(0, index=index, dtype=int)
    if not values:
        return series
    limit = min(len(values), len(series))
    series.iloc[:limit] = pd.Series(values[:limit], index=index[:limit]).astype(int)
    return series


def _smz_fvg_series(zones: list, index: pd.Index, bullish: bool) -> pd.Series:
    series = pd.Series(0, index=index, dtype=int)
    for zone in zones:
        if zone.is_bullish != bullish:
            continue
        created_at = int(zone.created_at)
        if 0 <= created_at < len(series):
            series.iloc[created_at] = 1
    return series


def add_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    sweeps = _cache.get_htf_sweeps(df)

    htf_4h_bull, htf_4h_bear = _build_sweep_series(sweeps.get("4H", []), df.index)
    htf_1d_bull, htf_1d_bear = _build_sweep_series(sweeps.get("1D", []), df.index)

    buyside_target_sweep = (
        (df["liq_buyside_target"] > 0)
        & (df["high"] > df["liq_buyside_target"])
        & (df["close"] < df["liq_buyside_target"])
    )
    sellside_target_sweep = (
        (df["liq_sellside_target"] > 0)
        & (df["low"] < df["liq_sellside_target"])
        & (df["close"] > df["liq_sellside_target"])
    )

    liquidity_sweep_bull = (
        buyside_target_sweep
        | (htf_4h_bull.astype(bool))
        | (htf_1d_bull.astype(bool))
    )
    liquidity_sweep_bear = (
        sellside_target_sweep
        | (htf_4h_bear.astype(bool))
        | (htf_1d_bear.astype(bool))
    )

    df = df.copy()
    df["htf_sweep_4h_bull"] = htf_4h_bull.to_numpy()
    df["htf_sweep_4h_bear"] = htf_4h_bear.to_numpy()
    df["htf_sweep_1d_bull"] = htf_1d_bull.to_numpy()
    df["htf_sweep_1d_bear"] = htf_1d_bear.to_numpy()
    df["liquidity_sweep_bull"] = liquidity_sweep_bull.astype(int)
    df["liquidity_sweep_bear"] = liquidity_sweep_bear.astype(int)
    return df


def add_mss_choch_signals(df: pd.DataFrame) -> pd.DataFrame:
    market_structure = calculate_market_structure_mtf(df)
    ms_tf1 = market_structure.tf1

    smc = _cache.get_smc_tradingfinder(df)
    structure = smc["structure"]

    try:
        cisd = detect_cd_sweep_cisd(df)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"CISD sweep detection failed; defaulting to no CISD signals. Error: {exc}",
            RuntimeWarning,
        )
        cisd = {"cisd_signals": [], "xbull_series": [], "xbear_series": []}
    cisd_bull = pd.Series(0, index=df.index, dtype=int)
    cisd_bear = pd.Series(0, index=df.index, dtype=int)
    for signal in cisd["cisd_signals"]:
        if 0 <= signal.idx < len(cisd_bull):
            if signal.is_bullish:
                cisd_bull.iloc[signal.idx] = 1
            else:
                cisd_bear.iloc[signal.idx] = 1

    cisd_xbull = _bool_series_from_list(cisd["xbull_series"], df.index)
    cisd_xbear = _bool_series_from_list(cisd["xbear_series"], df.index)

    choch_bull = (
        ms_tf1.bullish_choch
        | structure.bullish_minor_choch
        | structure.bullish_major_choch
    )
    choch_bear = (
        ms_tf1.bearish_choch
        | structure.bearish_minor_choch
        | structure.bearish_major_choch
    )

    killzone_cols = [
        "sb_sig_ln",
        "sb_sig_am",
        "sb_sig_pm",
        "sb_lux_ln",
        "sb_lux_am",
        "sb_lux_pm",
        "sb_or_range",
        "sb_trading_range",
    ]
    available_cols = [col for col in killzone_cols if col in df.columns]
    if available_cols:
        killzone_active = df[available_cols].astype(bool).any(axis=1)
    else:
        killzone_active = pd.Series(True, index=df.index)

    sweep_bull = df.get("liquidity_sweep_bull", pd.Series(0, index=df.index)).astype(bool)
    sweep_bear = df.get("liquidity_sweep_bear", pd.Series(0, index=df.index)).astype(bool)

    mss_bull = sweep_bull & killzone_active & choch_bull & (
        cisd_bull.astype(bool) | cisd_xbull.astype(bool)
    )
    mss_bear = sweep_bear & killzone_active & choch_bear & (
        cisd_bear.astype(bool) | cisd_xbear.astype(bool)
    )

    df = df.copy()
    df["choch_bull"] = choch_bull.astype(int)
    df["choch_bear"] = choch_bear.astype(int)
    df["cisd_bull"] = cisd_bull
    df["cisd_bear"] = cisd_bear
    df["mss_bull"] = mss_bull.astype(int)
    df["mss_bear"] = mss_bear.astype(int)
    return df


def add_mss_fvg_signals(df: pd.DataFrame, hk_aligned: pd.DataFrame) -> pd.DataFrame:
    sb = _cache.get_silver_bullet_signals(hk_aligned)
    sb_signals = sb["signals"]
    sb_fvg_bull = pd.Series(sb_signals["bull_fvg_formed"].to_numpy(), index=df.index).astype(int)
    sb_fvg_bear = pd.Series(sb_signals["bear_fvg_formed"].to_numpy(), index=df.index).astype(int)

    smz = _cache.get_smart_money_zones(df, show_ob=False)
    smz_fvg_bull = _smz_fvg_series(smz["bull_fvg"], df.index, True)
    smz_fvg_bear = _smz_fvg_series(smz["bear_fvg"], df.index, False)

    mss_bull = df.get("mss_bull", pd.Series(0, index=df.index)).astype(bool)
    mss_bear = df.get("mss_bear", pd.Series(0, index=df.index)).astype(bool)

    fvg_bull_sources = sb_fvg_bull.astype(bool) | smz_fvg_bull.astype(bool)
    fvg_bear_sources = sb_fvg_bear.astype(bool) | smz_fvg_bear.astype(bool)

    lux_cols = ["sb_lux_ln", "sb_lux_am", "sb_lux_pm"]
    lux_available = [col for col in lux_cols if col in df.columns]
    if lux_available:
        lux_active = df[lux_available].astype(bool).any(axis=1)
    else:
        lux_active = pd.Series(True, index=df.index)

    mss_fvg_bull = mss_bull & fvg_bull_sources & lux_active
    mss_fvg_bear = mss_bear & fvg_bear_sources & lux_active

    df = df.copy()
    df["fvg_sb_bull"] = sb_fvg_bull
    df["fvg_sb_bear"] = sb_fvg_bear
    df["fvg_smz_bull"] = smz_fvg_bull
    df["fvg_smz_bear"] = smz_fvg_bear
    df["mss_fvg_bull"] = mss_fvg_bull.astype(int)
    df["mss_fvg_bear"] = mss_fvg_bear.astype(int)
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
    zones_1h = _compute_htf_zones(df, resolution="60")
    zones_4h = _compute_htf_zones(df, resolution="240")

    bull_1h = _zones_to_in_zone_series(df, zones_1h, side="bull")
    bear_1h = _zones_to_in_zone_series(df, zones_1h, side="bear")
    bull_4h = _zones_to_in_zone_series(df, zones_4h, side="bull")
    bear_4h = _zones_to_in_zone_series(df, zones_4h, side="bear")

    bull_recent = (
        pd.concat([bull_1h, bull_4h], axis=1)
        .any(axis=1)
        .rolling(20, min_periods=1)
        .max()
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    bear_recent = (
        pd.concat([bear_1h, bear_4h], axis=1)
        .any(axis=1)
        .rolling(20, min_periods=1)
        .max()
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    df = df.copy()
    df["filter_htf_poi_bull"] = bull_recent
    df["filter_htf_poi_bear"] = bear_recent
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

    # Raw entry triggers (unfiltered) - checked FIRST in strategy
    entry_trigger_bull = (
        sb_entry_bull.astype(bool) | setup01_bull.astype(bool) | ote_bull.astype(bool)
    ).astype(int)
    entry_trigger_bear = (
        sb_entry_bear.astype(bool) | setup01_bear.astype(bool) | ote_bear.astype(bool)
    ).astype(int)

    # Filter 1: HTF POI Filter (OB wick zones)
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
    entry_fvg_bull = (
        entry_trigger_bull.astype(bool)
        & session_active.astype(bool)
        & htf_bias_bull.astype(bool)
    ).astype(int)
    entry_fvg_bear = (
        entry_trigger_bear.astype(bool)
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


def add_stop_loss_levels(df: pd.DataFrame) -> pd.DataFrame:
    sweeps = _cache.get_htf_sweeps(df)
    htf_bull_prices = _build_sweep_price_series(sweeps.get("4H", []), df.index, True)
    htf_bear_prices = _build_sweep_price_series(sweeps.get("4H", []), df.index, False)
    htf1d_bull_prices = _build_sweep_price_series(sweeps.get("1D", []), df.index, True)
    htf1d_bear_prices = _build_sweep_price_series(sweeps.get("1D", []), df.index, False)

    sellside_liq = df.get("liq_sellside_target", pd.Series(0.0, index=df.index))
    buyside_liq = df.get("liq_buyside_target", pd.Series(0.0, index=df.index))

    bull_stop_candidates = pd.concat(
        [
            sellside_liq.replace(0, pd.NA),
            htf_bear_prices.replace(0, pd.NA),
            htf1d_bear_prices.replace(0, pd.NA),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    bear_stop_candidates = pd.concat(
        [
            buyside_liq.replace(0, pd.NA),
            htf_bull_prices.replace(0, pd.NA),
            htf1d_bull_prices.replace(0, pd.NA),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    sweep_bull = df.get("liquidity_sweep_bull", pd.Series(0, index=df.index)).astype(bool)
    sweep_bear = df.get("liquidity_sweep_bear", pd.Series(0, index=df.index)).astype(bool)

    stop_loss_bull = np.where(sweep_bull, bull_stop_candidates, 0.0)
    stop_loss_bear = np.where(sweep_bear, bear_stop_candidates, 0.0)
    stop_loss_bull = pd.Series(stop_loss_bull, index=df.index, dtype=float)
    stop_loss_bear = pd.Series(stop_loss_bear, index=df.index, dtype=float)

    stop_loss_bull = stop_loss_bull.ffill().fillna(0.0)
    stop_loss_bear = stop_loss_bear.ffill().fillna(0.0)

    df = df.copy()
    df["stop_loss_bull"] = stop_loss_bull
    df["stop_loss_bear"] = stop_loss_bear
    return df


def add_target_levels(df: pd.DataFrame) -> pd.DataFrame:
    sweeps = _cache.get_htf_sweeps(df)
    htf_bull_prices = _build_sweep_price_series(sweeps.get("4H", []), df.index, True)
    htf_bear_prices = _build_sweep_price_series(sweeps.get("4H", []), df.index, False)
    htf1d_bull_prices = _build_sweep_price_series(sweeps.get("1D", []), df.index, True)
    htf1d_bear_prices = _build_sweep_price_series(sweeps.get("1D", []), df.index, False)

    buyside_liq = df.get("liq_buyside_target", pd.Series(0.0, index=df.index))
    sellside_liq = df.get("liq_sellside_target", pd.Series(0.0, index=df.index))

    target_bull = pd.concat(
        [
            buyside_liq.replace(0, pd.NA),
            htf_bull_prices.replace(0, pd.NA),
            htf1d_bull_prices.replace(0, pd.NA),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    target_bear = pd.concat(
        [
            sellside_liq.replace(0, pd.NA),
            htf_bear_prices.replace(0, pd.NA),
            htf1d_bear_prices.replace(0, pd.NA),
        ],
        axis=1,
    ).min(axis=1, skipna=True)

    target_bull = target_bull.fillna(0.0)
    target_bear = target_bear.fillna(0.0)

    df = df.copy()
    df["target_bull"] = target_bull
    df["target_bear"] = target_bear
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


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

    # Fast mode skips non-essential indicators for quicker backtesting
    if fast_mode or _env_bool("SB_FAST_MODE", default=False):
        pass

    data_df = add_smart_money_trends(load_data(csv_file, max_rows=max_rows))

    # Create HK-aligned DataFrame once for all functions that need it
    hk_aligned = _align_index_to_hk_as_ny(data_df)

    data_df = add_external_liquidity_targets(data_df)
    data_df = add_killzone_windows(data_df, hk_aligned)
    data_df = add_liquidity_sweeps(data_df)
    data_df = add_mss_choch_signals(data_df)
    data_df = add_mss_fvg_signals(data_df, hk_aligned)
    data_df = add_ict_session_filter(data_df)
    data_df = add_htf_ob_filter(data_df)
    data_df = add_entry_signals(data_df, hk_aligned)

    # Skip pivots in fast mode (not used for trade entries)
    if not fast_mode:
        data_df = add_bigbeluga_pivots(data_df)
    else:
        data_df["bb_pivot_high"] = 0.0
        data_df["bb_pivot_low"] = 0.0

    data_df = add_stop_loss_levels(data_df)
    data_df = add_target_levels(data_df)
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
    analyzer = results[0].analyzers.trade_analyzer.get_analysis()
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
