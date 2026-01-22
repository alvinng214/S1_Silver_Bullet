"""
Backtest script to compute HTF market structure bias using Backtrader.

Steps:
1) Load OHLC data from CSV and set a datetime index.
2) Compute Smart Money Zones MTF trends (4H/1D) with pandas.
3) Attach Smart Money trend columns to the base dataframe.
4) Identify HTF Points of Interest (POI) using MirPapa HTF FVG/OB Threeple.
5) Identify external liquidity targets (profit targets) via SMC + inducements.
6) Mark session highs/lows and daily 50% levels for Step 4 preparation.
7) Detect liquidity sweeps using session levels, inducements, and HTF sweeps.
8) Feed data into Backtrader and resample to 4H/1D.
9) Run HTFBiasStrategy to log consolidated HTF bias and POI counts.
"""

from __future__ import annotations

import argparse
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

ASIALONDON_PATH = os.path.join(
    os.path.dirname(__file__),
    "SW's AsiaLondon HL's.py",
)
asialondon_module = SourceFileLoader("asialondon_levels", ASIALONDON_PATH).load_module()
calculate_asia_london_levels = asialondon_module.calculate_asia_london_levels

ICT_CUSTOM_PATH = os.path.join(
    os.path.dirname(__file__),
    "ICT_Customizable_50__Line___DailyAsiaLondonNew_York_HighLow___True_Day_Open.py",
)
ict_custom_module = SourceFileLoader("ict_custom_50_line", ICT_CUSTOM_PATH).load_module()
calculate_custom_50_line = ict_custom_module.calculate_indicator

IGHODALO_CRT_PATH = os.path.join(
    os.path.dirname(__file__),
    "Ighodalo_Gold_-_CRT__Candles_are_ranges_theory_.py",
)
ighodalo_module = SourceFileLoader("ighodalo_crt", IGHODALO_CRT_PATH).load_module()
calculate_ighodalo_crt = ighodalo_module.calculate_ighodalo_crt

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
        "asia_session_high",
        "asia_session_low",
        "london_session_high",
        "london_session_low",
        "ny_session_high",
        "ny_session_low",
        "pd_high",
        "pd_low",
        "pd_mid",
        "daily_mid_50",
        "true_day_open",
        "crt_buy_signal",
        "crt_sell_signal",
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
        ("asia_session_high", "asia_session_high"),
        ("asia_session_low", "asia_session_low"),
        ("london_session_high", "london_session_high"),
        ("london_session_low", "london_session_low"),
        ("ny_session_high", "ny_session_high"),
        ("ny_session_low", "ny_session_low"),
        ("pd_high", "pd_high"),
        ("pd_low", "pd_low"),
        ("pd_mid", "pd_mid"),
        ("daily_mid_50", "daily_mid_50"),
        ("true_day_open", "true_day_open"),
        ("crt_buy_signal", "crt_buy_signal"),
        ("crt_sell_signal", "crt_sell_signal"),
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
    )


def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close"]].dropna()
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


def _to_utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def add_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    aligned.index = pd.to_datetime(aligned.index)
    if aligned.index.tz is None:
        aligned.index = aligned.index.tz_localize("UTC")

    asia_levels = calculate_asia_london_levels(aligned, timezone_offset=3)

    def build_session_series(session: str, is_high: bool) -> pd.Series:
        values = pd.Series(0.0, index=df.index, dtype=float)
        for line in asia_levels.session_lines:
            if line.session != session or line.is_high != is_high:
                continue
            start = _to_utc_naive(line.start_time)
            end = _to_utc_naive(line.end_time)
            mask = (values.index >= start) & (values.index <= end)
            values.loc[mask] = float(line.price)
        return values

    asia_high = build_session_series("Asia", True)
    asia_low = build_session_series("Asia", False)
    london_high = build_session_series("London", True)
    london_low = build_session_series("London", False)
    ny_high = build_session_series("NY", True)
    ny_low = build_session_series("NY", False)

    pd_high = float(asia_levels.pd_levels.high or 0.0)
    pd_low = float(asia_levels.pd_levels.low or 0.0)
    pd_mid = float(asia_levels.pd_levels.mid or 0.0)

    custom_levels = calculate_custom_50_line(aligned)
    daily_states = custom_levels["daily_states"]
    daily_mid_50 = pd.Series(0.0, index=df.index, dtype=float)
    for idx, state in enumerate(daily_states):
        if idx >= len(daily_mid_50):
            break
        daily_mid_50.iloc[idx] = float(state.mid_level)

    true_day_open_states = custom_levels["true_day_open_states"]
    true_day_open = pd.Series(0, index=df.index, dtype=int)
    for idx in true_day_open_states.keys():
        if 0 <= idx < len(true_day_open):
            true_day_open.iloc[idx] = 1

    crt = calculate_ighodalo_crt(df)
    crt_buy = pd.Series(0, index=df.index, dtype=int)
    crt_sell = pd.Series(0, index=df.index, dtype=int)
    for signal in crt["signals"]:
        if signal.direction == "buy" and 0 <= signal.index < len(crt_buy):
            crt_buy.iloc[signal.index] = 1
        if signal.direction == "sell" and 0 <= signal.index < len(crt_sell):
            crt_sell.iloc[signal.index] = 1

    df = df.copy()
    df["asia_session_high"] = asia_high
    df["asia_session_low"] = asia_low
    df["london_session_high"] = london_high
    df["london_session_low"] = london_low
    df["ny_session_high"] = ny_high
    df["ny_session_low"] = ny_low
    df["pd_high"] = pd_high
    df["pd_low"] = pd_low
    df["pd_mid"] = pd_mid
    df["daily_mid_50"] = daily_mid_50
    df["true_day_open"] = true_day_open
    df["crt_buy_signal"] = crt_buy
    df["crt_sell_signal"] = crt_sell
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


def _build_sweep_series(candles: list, length: int) -> tuple[pd.Series, pd.Series]:
    bull = pd.Series(0, index=range(length), dtype=int)
    bear = pd.Series(0, index=range(length), dtype=int)
    for candle in candles:
        for sweep in candle.htf_sweeps:
            if not sweep.formed or sweep.removed:
                continue
            idx = sweep.x2
            if 0 <= idx < length:
                if sweep.bull:
                    bull.iloc[idx] = 1
                else:
                    bear.iloc[idx] = 1
    return bull, bear


def add_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    sweeps = calculate_htf_sweeps(
        df,
        timeframes=[
            ("4H", 200, True),
            ("1D", 200, True),
        ],
    )

    htf_4h_bull, htf_4h_bear = _build_sweep_series(sweeps.get("4H", []), len(df))
    htf_1d_bull, htf_1d_bear = _build_sweep_series(sweeps.get("1D", []), len(df))

    session_highs = (
        df[["asia_session_high", "london_session_high", "ny_session_high", "pd_high"]]
        .replace(0, pd.NA)
        .max(axis=1, skipna=True)
        .fillna(0.0)
    )
    session_lows = (
        df[["asia_session_low", "london_session_low", "ny_session_low", "pd_low"]]
        .replace(0, pd.NA)
        .min(axis=1, skipna=True)
        .fillna(0.0)
    )

    buyside_session_sweep = (
        (session_highs > 0) & (df["high"] > session_highs) & (df["close"] < session_highs)
    )
    sellside_session_sweep = (
        (session_lows > 0) & (df["low"] < session_lows) & (df["close"] > session_lows)
    )

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
        buyside_session_sweep
        | buyside_target_sweep
        | (htf_4h_bull.astype(bool))
        | (htf_1d_bull.astype(bool))
    )
    liquidity_sweep_bear = (
        sellside_session_sweep
        | sellside_target_sweep
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


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def run_backtest(csv_file: str, max_bars: int | None = None) -> None:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    warnings.filterwarnings("ignore", category=FutureWarning)

    data_df = load_data(csv_file)
    data_df = limit_bars(data_df, max_bars)
    data_df = add_smart_money_trends(data_df)
    data_df = add_htf_poi(data_df)
    data_df = add_external_liquidity_targets(data_df)
    data_df = add_session_levels(data_df)
    data_df = add_liquidity_sweeps(data_df)
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
    args = parser.parse_args()
    max_bars = args.max_bars if args.max_bars > 0 else None
    run_backtest(args.csv_file, max_bars=max_bars)


if __name__ == "__main__":
    main()
