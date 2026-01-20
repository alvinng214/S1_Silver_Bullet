"""
Backtest script to compute HTF market structure bias using Backtrader.

Steps:
1) Load OHLC data from CSV and set a datetime index.
2) Compute Smart Money Zones MTF trends (4H/1D) with pandas.
3) Attach Smart Money trend columns to the base dataframe.
4) Identify HTF Points of Interest (POI) using MirPapa HTF FVG/OB Threeple.
5) Identify external liquidity targets (profit targets) via SMC + inducements.
6) Feed data into Backtrader and resample to 4H/1D.
7) Run HTFBiasStrategy to log consolidated HTF bias and POI counts.
"""

from __future__ import annotations

import os
import sys
import warnings

import backtrader as bt
import pandas as pd
from importlib.machinery import SourceFileLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "strategies"))

from htf_bias_strategy import HTFBiasStrategy
from ICT_Silver_Bullet_with_signals import detect_silver_bullet_signals
from Smart_Money_Concept__TradingFinder__Major_Minor_OB___FVG__SMC_ import calculate_smc_tradingfinder
from Smart_Money_Zones__FVG___OB____MTF_Trend_Panel import calculate_smart_money_zones

THREEPLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "MirPapa-ICT-HTF- FVG OB Threeple (EN).py",
)
threeple_module = SourceFileLoader("mirpapa_threeple", THREEPLE_PATH).load_module()
calculate_fvg_ob_threeple = threeple_module.calculate_fvg_ob_threeple

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


class PandasDataBias(bt.feeds.PandasData):
    lines = (
        "smz_trend_4h",
        "smz_trend_1d",
        "poi_high_bull",
        "poi_high_bear",
        "poi_mid_bull",
        "poi_mid_bear",
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
    )
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", -1),
        ("openinterest", -1),
        ("smz_trend_4h", "smz_trend_4h"),
        ("smz_trend_1d", "smz_trend_1d"),
        ("poi_high_bull", "poi_high_bull"),
        ("poi_high_bear", "poi_high_bear"),
        ("poi_mid_bull", "poi_mid_bull"),
        ("poi_mid_bear", "poi_mid_bear"),
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
    )


def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close"]].dropna()
    return df


def add_smart_money_trends(df: pd.DataFrame) -> pd.DataFrame:
    results = calculate_smart_money_zones(df)
    trend_4h = results["mtf_trends"]["4h"].astype(int).replace({0: -1})
    trend_1d = results["mtf_trends"]["1d"].astype(int).replace({0: -1})

    df = df.copy()
    df["smz_trend_4h"] = trend_4h
    df["smz_trend_1d"] = trend_1d
    return df


def infer_chart_timeframe_minutes(df: pd.DataFrame) -> int:
    if len(df.index) < 2:
        return 0
    deltas = df.index.to_series().diff().dropna()
    return int(deltas.median().total_seconds() // 60)


def format_chart_timeframe(minutes: int) -> str:
    if minutes <= 0:
        return "5"
    if minutes % (60 * 24) == 0:
        days = minutes // (60 * 24)
        return f"{days}D"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours * 60}"
    return str(minutes)


def add_htf_poi(df: pd.DataFrame) -> pd.DataFrame:
    chart_minutes = infer_chart_timeframe_minutes(df)
    chart_tf = format_chart_timeframe(chart_minutes)
    outputs = calculate_fvg_ob_threeple(
        df,
        chart_timeframe=chart_tf,
        mid_tf_override="60",
        high_tf_override="240",
    )

    length = len(df)
    high_bull = pd.Series(0, index=df.index, dtype=int)
    high_bear = pd.Series(0, index=df.index, dtype=int)
    mid_bull = pd.Series(0, index=df.index, dtype=int)
    mid_bear = pd.Series(0, index=df.index, dtype=int)

    def apply_boxes(series: pd.Series, boxes, is_bull: bool) -> None:
        for box in boxes:
            if not box.is_active or box.is_bullish != is_bull:
                continue
            start = max(0, box.start_index)
            end = min(length - 1, box.end_index)
            if start <= end:
                series.iloc[start : end + 1] += 1

    apply_boxes(high_bull, outputs.high_tf_boxes, True)
    apply_boxes(high_bear, outputs.high_tf_boxes, False)
    apply_boxes(mid_bull, outputs.mid_tf_boxes, True)
    apply_boxes(mid_bear, outputs.mid_tf_boxes, False)

    df = df.copy()
    df["poi_high_bull"] = high_bull
    df["poi_high_bear"] = high_bear
    df["poi_mid_bull"] = mid_bull
    df["poi_mid_bear"] = mid_bear
    return df


def add_external_liquidity_targets(df: pd.DataFrame) -> pd.DataFrame:
    smc = calculate_smc_tradingfinder(df)
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


def add_killzone_windows(df: pd.DataFrame) -> pd.DataFrame:
    hk_aligned = _align_index_to_hk_as_ny(df)

    sb_signals = detect_silver_bullet_signals(hk_aligned)
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


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def run_backtest(csv_file: str) -> None:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    warnings.filterwarnings("ignore", category=FutureWarning)

    data_df = add_smart_money_trends(load_data(csv_file))
    data_df = add_htf_poi(data_df)
    data_df = add_external_liquidity_targets(data_df)
    data_df = add_killzone_windows(data_df)
    data_4h_df = resample_ohlc(data_df, "4H")
    data_1d_df = resample_ohlc(data_df, "1D")

    data = PandasDataBias(dataname=data_df)
    data_4h = bt.feeds.PandasData(dataname=data_4h_df)
    data_1d = bt.feeds.PandasData(dataname=data_1d_df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(HTFBiasStrategy, print_bias=True)
    cerebro.adddata(data)
    cerebro.adddata(data_4h)
    cerebro.adddata(data_1d)
    cerebro.broker.setcash(10000.0)

    print("\n" + "=" * 80)
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print("=" * 80 + "\n")

    cerebro.run()

    print("\n" + "=" * 80)
    print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print("=" * 80)


def main() -> None:
    csv_file = "PEPPERSTONE_XAUUSD, 5.csv"
    run_backtest(csv_file)


if __name__ == "__main__":
    main()
