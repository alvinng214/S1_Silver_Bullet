"""
Order-Block Detector
====================
Translated from Pine Script v5 (© veegee82) to Python.
This source code is subject to the terms of the Mozilla Public License 2.0
at https://mozilla.org/MPL/2.0/

Strictly mirrors all logic and details of the original Pine Script indicator.

Dependencies: pandas, numpy, matplotlib (optional, for plotting)

Usage:
    # Programmatic
    from order_block_detector import OrderBlockDetector
    import pandas as pd

    df = pd.read_csv('ohlc.csv', parse_dates=[0], index_col=0)
    detector = OrderBlockDetector()
    results = detector.run(df)
    print(results['signals'])

    # Command line
    python "Order-Block Detector.py" ohlc.csv --plot
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from copy import deepcopy


# ---------------------------------------------------------------------------
# Data types (mirroring Pine Script UDTs)
# ---------------------------------------------------------------------------

@dataclass
class OB:
    """Order Block record.

    Pine Script equivalent:
        type OB
            float max
            float min
            bool  isbull
            int   t = time
            int   idx = bar_index
    """
    max: float = float('nan')
    min: float = float('nan')
    isbull: Optional[bool] = None
    t: Optional[pd.Timestamp] = None   # time of the bar where OB was detected
    idx: int = 0                        # bar_index (in the detection timeframe)


@dataclass
class FVG:
    """Fair Value Gap record.

    Pine Script equivalent:
        type FVG
            float max
            float min
            bool  isbull
            int   t = time
            int   idx = bar_index
    """
    max: float = float('nan')
    min: float = float('nan')
    isbull: Optional[bool] = None
    t: Optional[pd.Timestamp] = None
    idx: int = 0


@dataclass
class Signal:
    """Signal record.

    Pine Script equivalent:
        type Signals
            float candle_open
            float candle_high
            float candle_low
            float candle_close
            float point
            bool  isbull
            bool  entry = false
            int   idx = bar_index
            int   t = time

    Note: Signals.new(0) in Pine sets candle_open=0 and all other floats
    default to na (NaN). isbull defaults to na (None in Python).
    """
    candle_open: float = 0.0
    candle_high: float = float('nan')
    candle_low: float = float('nan')
    candle_close: float = float('nan')
    point: float = float('nan')
    isbull: Optional[bool] = None
    entry: bool = False
    idx: int = 0
    t: Optional[pd.Timestamp] = None


# ---------------------------------------------------------------------------
# Visual element records (for recording boxes / lines without TradingView)
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """Represents a Pine Script box (OB or FVG zone rectangle)."""
    left: int
    top: float
    right: int
    bottom: float
    isbull: bool
    kind: str                               # 'ob' or 'fvg'
    deleted: bool = False
    t_left: Optional[pd.Timestamp] = None
    t_right: Optional[pd.Timestamp] = None


@dataclass
class MitigationLine:
    """Represents a Pine Script dashed line drawn at mitigation."""
    t1: Optional[pd.Timestamp] = None
    price: float = 0.0
    t2: Optional[pd.Timestamp] = None
    isbull: bool = True
    kind: str = 'ob'                        # 'ob' or 'fvg'


# ---------------------------------------------------------------------------
# Heikin-Ashi computation
# Mirrors: request.security(ticker.heikinashi(syminfo.tickerid), tf, ...)
# ---------------------------------------------------------------------------

def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi OHLC from standard OHLC DataFrame.

    HA_Close = (Open + High + Low + Close) / 4
    HA_Open[0] = (Open[0] + Close[0]) / 2
    HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2   for i >= 1
    HA_High = max(High, HA_Open, HA_Close)
    HA_Low  = min(Low,  HA_Open, HA_Close)
    """
    ha = pd.DataFrame(index=df.index)
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    ha_open = np.empty(len(df))
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2.0
    ha_close_vals = ha['close'].values
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close_vals[i - 1]) / 2.0
    ha['open'] = ha_open

    ha['high'] = np.maximum(np.maximum(ha['open'].values, ha['close'].values),
                            df['high'].values)
    ha['low'] = np.minimum(np.minimum(ha['open'].values, ha['close'].values),
                           df['low'].values)
    return ha[['open', 'high', 'low', 'close']]


# ---------------------------------------------------------------------------
# Timeframe resampling helpers
# Mirrors: request.security(syminfo.tickerid, tf, ...)
# ---------------------------------------------------------------------------

def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample OHLC data to a higher timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data with DatetimeIndex.
    tf : str
        Pandas-compatible frequency string (e.g. '5min', '15min', '1h',
        '4h', '1D').

    Returns
    -------
    pd.DataFrame
        Resampled OHLC data.
    """
    resampled = df.resample(tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    return resampled


def align_htf_to_ltf(htf_df: pd.DataFrame,
                      ltf_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align higher-timeframe data back to lower-timeframe index.

    Uses forward-fill to replicate Pine Script's request.security behavior:
    HTF values remain constant until the next HTF bar completes.
    """
    aligned = htf_df.reindex(ltf_index, method='ffill')
    return aligned


# ---------------------------------------------------------------------------
# Detection functions (mirror Pine detect() and detect_fvg())
# ---------------------------------------------------------------------------

def detect_ob_on_bar(open_val: float, high_val: float,
                     low_val: float, close_val: float,
                     open_prev: float, high_prev: float,
                     low_prev: float, close_prev: float,
                     bar_time, bar_idx: int,
                     show_ob: bool = True) -> Tuple[OB, bool]:
    """Detect an Order Block on a single bar.

    Mirrors Pine detect() function (lines 50-67):
        candle_dir = close > open ? 1 : -1
        candle_dir_prev = close[1] > open[1] ? 1 : -1
        Bullish OB:  candle_dir == 1  and candle_dir_prev == -1 and high > high[1]
        Bearish OB:  candle_dir == -1 and candle_dir_prev == 1  and low  < low[1]
        OB zone = (high[1], low[1])  i.e. the previous candle's range.

    Returns
    -------
    (OB, detected: bool)
    """
    candle_dir = 1 if close_val > open_val else -1
    candle_dir_prev = 1 if close_prev > open_prev else -1
    detected = False
    ob = OB()

    # Bullish OB: current candle bullish, previous bearish, current high > prev high
    if candle_dir == 1 and candle_dir_prev == -1 and high_val > high_prev:
        ob = OB(max=high_prev, min=low_prev, isbull=True,
                t=bar_time, idx=bar_idx)
        detected = True

    # Bearish OB: current candle bearish, previous bullish, current low < prev low
    # Note: this is a separate 'if', not 'elif', mirroring Pine exactly.
    # If both conditions are true, the bearish OB overwrites the bullish one.
    if candle_dir == -1 and candle_dir_prev == 1 and low_val < low_prev:
        ob = OB(max=high_prev, min=low_prev, isbull=False,
                t=bar_time, idx=bar_idx)
        detected = True

    if not show_ob:
        detected = False

    return ob, detected


def detect_fvg_on_bar(open_val: float, high_val: float,
                      low_val: float, close_val: float,
                      high_2ago: float, low_2ago: float,
                      bar_time, bar_idx: int,
                      show_fvg: bool = True) -> Tuple[FVG, bool]:
    """Detect a Fair Value Gap on a single bar.

    Mirrors Pine detect_fvg() function (lines 69-87):
        Bullish FVG:  low > high[2]   (gap up)
        Bearish FVG:  low[2] > high   (gap down)

    Returns
    -------
    (FVG, detected: bool)
    """
    detected = False
    fvg = FVG()

    # Bullish FVG: current bar's low is above the high from 2 bars ago
    if low_val > high_2ago:
        fvg = FVG(max=low_val, min=high_2ago, isbull=True,
                  t=bar_time, idx=bar_idx)
        detected = True

    # Bearish FVG: low from 2 bars ago is above current bar's high
    # Note: separate 'if', not 'elif', mirroring Pine exactly.
    if low_2ago > high_val:
        fvg = FVG(max=low_2ago, min=high_val, isbull=False,
                  t=bar_time, idx=bar_idx)
        detected = True

    if not show_fvg:
        detected = False

    return fvg, detected


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class OrderBlockDetector:
    """Order-Block Detector that strictly mirrors the Pine Script logic.

    Parameters mirror Pine Script inputs (lines 7-23):
        tf : str
            Higher timeframe for OB/FVG detection.  Empty string '' means
            use the data's own timeframe (Pine default: '').
            Use pandas freq strings: '5min', '15min', '1h', '4h', '1D'.
        show_ob : bool
            Whether to detect Order Blocks (Pine: Show Order-Blocks).
        show_fvg : bool
            Whether to detect Fair Value Gaps (Pine: Show Fair-Value-Gaps).
        show_signals_ob : bool
            Whether to include OB signals in output (Pine: Show Signals OB).
        show_signals_fvg : bool
            Whether to include FVG signals in output (Pine: Show Signals FVG).
        min_dist : int
            Minimum bar distance between OB creation and mitigation for a
            signal to be generated (Pine: Min dist, default 1).
        min_dist_fvg : int
            Minimum bar distance between FVG creation and mitigation for a
            signal to be generated (Pine: Min dist FVG, default 1).
        use_ha : bool
            Use Heikin-Ashi candles for the close comparison in signal
            generation (Pine: Use Heikin-Ashi, default false).
    """

    def __init__(
        self,
        tf: str = '',
        show_ob: bool = True,
        show_fvg: bool = True,
        show_signals_ob: bool = True,
        show_signals_fvg: bool = True,
        min_dist: int = 1,
        min_dist_fvg: int = 1,
        use_ha: bool = False,
    ):
        self.tf = tf
        self.show_ob = show_ob
        self.show_fvg = show_fvg
        self.show_signals_ob = show_signals_ob
        self.show_signals_fvg = show_signals_fvg
        self.min_dist = min_dist
        self.min_dist_fvg = min_dist_fvg
        self.use_ha = use_ha

    def run(self, df: pd.DataFrame) -> Dict:
        """Run the Order-Block Detector on OHLC data.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: 'open', 'high', 'low', 'close'.
            Index must be a DatetimeIndex.

        Returns
        -------
        dict with keys:
            'signals'            : pd.DataFrame with columns 'ob_signal' and
                                   'fvg_signal'. Values: 1 (buy), -1 (sell),
                                   0 (none).
            'ob_records'         : list[OB]  -- active (unmitigated) OBs at end.
            'fvg_records'        : list[FVG] -- active (unmitigated) FVGs at end.
            'boxes'              : list[Box] -- all box records for visualization.
            'lines'              : list[MitigationLine] -- mitigation lines.
            'signal_details_ob'  : list[Signal | None] per bar.
            'signal_details_fvg' : list[Signal | None] per bar.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {'open', 'high', 'low', 'close'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        # ================================================================
        # Prepare security (higher-TF) data
        # Mirrors: request.security(syminfo.tickerid, tf, detect())
        #          request.security(syminfo.tickerid, tf, detect_fvg())
        # ================================================================
        use_htf = bool(self.tf and self.tf != '')

        if use_htf:
            htf_df = resample_ohlc(df, self.tf)
            sec_df = align_htf_to_ltf(htf_df, df.index)
        else:
            htf_df = df
            sec_df = df.copy()

        # ================================================================
        # Prepare Heikin-Ashi data on the security timeframe
        # Mirrors: request.security(ticker.heikinashi(syminfo.tickerid),
        #                           tf, [open, high, low, close])
        # ================================================================
        ha_src = compute_heikin_ashi(htf_df)
        if use_htf:
            ha_aligned = align_htf_to_ltf(ha_src, df.index)
        else:
            ha_aligned = ha_src.copy()

        # ================================================================
        # Run detection on the detection-source timeframe
        # (HTF bars if tf is set, otherwise every chart bar)
        # ================================================================
        detect_source = htf_df if use_htf else df

        det_open = detect_source['open'].values
        det_high = detect_source['high'].values
        det_low = detect_source['low'].values
        det_close = detect_source['close'].values
        det_times = detect_source.index
        n_det = len(detect_source)

        det_ob_list: List[Tuple[OB, bool]] = []
        det_fvg_list: List[Tuple[FVG, bool]] = []

        for i in range(n_det):
            # OB detection requires [1] lookback
            if i >= 1:
                ob, ob_det = detect_ob_on_bar(
                    det_open[i], det_high[i], det_low[i], det_close[i],
                    det_open[i - 1], det_high[i - 1],
                    det_low[i - 1], det_close[i - 1],
                    det_times[i], i, self.show_ob
                )
            else:
                ob, ob_det = OB(), False

            # FVG detection requires [2] lookback
            if i >= 2:
                fvg, fvg_det = detect_fvg_on_bar(
                    det_open[i], det_high[i], det_low[i], det_close[i],
                    det_high[i - 2], det_low[i - 2],
                    det_times[i], i, self.show_fvg
                )
            else:
                fvg, fvg_det = FVG(), False

            det_ob_list.append((ob, ob_det))
            det_fvg_list.append((fvg, fvg_det))

        # ================================================================
        # Build mapping from chart bar index -> detection-source bar index
        # For same-TF this is identity; for HTF it maps each chart bar to
        # the last completed HTF bar (forward-fill).
        # ================================================================
        n_bars = len(df)
        times = df.index

        if use_htf:
            htf_bar_map = np.zeros(n_bars, dtype=int)
            htf_idx = 0
            for chart_i in range(n_bars):
                while (htf_idx < n_det - 1
                       and det_times[htf_idx + 1] <= times[chart_i]):
                    htf_idx += 1
                htf_bar_map[chart_i] = htf_idx
        else:
            htf_bar_map = np.arange(n_bars, dtype=int)

        # ================================================================
        # Extract numpy arrays for fast access
        # ================================================================
        sec_open = sec_df['open'].values
        sec_high = sec_df['high'].values
        sec_low = sec_df['low'].values
        sec_close = sec_df['close'].values

        chart_open = df['open'].values
        chart_high = df['high'].values
        chart_low = df['low'].values
        chart_close = df['close'].values

        ha_close = ha_aligned['close'].values

        # ================================================================
        # State variables (mirrors Pine 'var' declarations, lines 95-121)
        # ================================================================
        ob_records: List[OB] = []           # var ob_records = array.new<OB>(0)
        ob_boxes: List[Box] = []            # var ob_boxes = array.new<box>(0)
        fvg_records: List[FVG] = []         # var fvg_records = array.new<FVG>(0)
        fvg_boxes: List[Box] = []           # var fvg_boxes = array.new<box>(0)

        t_last_ob = None                    # var t = 0
        t_last_fvg = None                   # var t_fvg = 0

        # var signal = Signals.new(0)  ->  point=na, isbull=na, entry=false
        signal = Signal()
        # var signal_fvg = Signals.new(0)
        signal_fvg = Signal()

        # ================================================================
        # Output storage
        # ================================================================
        all_lines: List[MitigationLine] = []
        all_boxes: List[Box] = []

        ob_signals = np.zeros(n_bars, dtype=int)
        fvg_signals = np.zeros(n_bars, dtype=int)
        signal_details_ob: List[Optional[Signal]] = [None] * n_bars
        signal_details_fvg: List[Optional[Signal]] = [None] * n_bars

        # ================================================================
        # Main bar-by-bar loop (mirrors Pine main body, lines 89-197)
        # Pine processes each chart bar sequentially.
        # ================================================================
        for n in range(n_bars):
            bar_time = times[n]
            htf_i = htf_bar_map[n]

            # --- Detection results for this bar's HTF bar ---
            new_ob, ob_detected = det_ob_list[htf_i]
            new_fvg, fvg_detected = det_fvg_list[htf_i]

            # --- Security OHLC for this bar ---
            # s_open, s_high, s_low, s_close  (from OB security call)
            s_low_val = sec_low[n]
            s_high_val = sec_high[n]
            # fvg_open, fvg_high, fvg_low, fvg_close (from FVG security call)
            # Same ticker+tf, so identical to s_* values.
            fvg_low_val = sec_low[n]
            fvg_high_val = sec_high[n]
            fvg_close_val = sec_close[n]

            # --------------------------------------------------------
            # Add new FVG if detected and not a duplicate
            # Pine lines 111-114:
            #   if fvg_detected and new_fvg.t != t_fvg
            # --------------------------------------------------------
            if fvg_detected and new_fvg.t != t_last_fvg:
                fvg_copy = FVG(
                    max=new_fvg.max, min=new_fvg.min,
                    isbull=new_fvg.isbull, t=new_fvg.t, idx=new_fvg.idx
                )
                # Pine: fvg_records.unshift(new_fvg) -> insert at front
                fvg_records.insert(0, fvg_copy)
                # Pine: fvg_boxes.unshift(box.new(n-1, ...))
                box = Box(
                    left=n - 1, top=fvg_copy.max, right=n,
                    bottom=fvg_copy.min,
                    isbull=fvg_copy.isbull, kind='fvg',
                    t_left=bar_time, t_right=bar_time
                )
                fvg_boxes.insert(0, box)
                all_boxes.append(box)
                t_last_fvg = new_fvg.t

            # --------------------------------------------------------
            # Add new OB if detected and not a duplicate
            # Pine lines 116-119:
            #   if detected and new_ob.t != t
            # --------------------------------------------------------
            if ob_detected and new_ob.t != t_last_ob:
                ob_copy = OB(
                    max=new_ob.max, min=new_ob.min,
                    isbull=new_ob.isbull, t=new_ob.t, idx=new_ob.idx
                )
                ob_records.insert(0, ob_copy)
                box = Box(
                    left=n - 1, top=ob_copy.max, right=n,
                    bottom=ob_copy.min,
                    isbull=ob_copy.isbull, kind='ob',
                    t_left=bar_time, t_right=bar_time
                )
                ob_boxes.insert(0, box)
                all_boxes.append(box)
                t_last_ob = new_ob.t

            # --------------------------------------------------------
            # OB mitigation check
            # Pine lines 125-145:
            #   for i = ob_records.size()-1 to 0
            # Iterates from oldest (end) to newest (front) for safe
            # removal during iteration.
            # --------------------------------------------------------
            if len(ob_records) > 0:
                i = len(ob_records) - 1
                while i >= 0:
                    get = ob_records[i]
                    if get.isbull:
                        # Pine line 130:
                        # if (s_low <= get.max or low <= get.max)
                        #    and get.t < time
                        if ((s_low_val <= get.max
                             or chart_low[n] <= get.max)
                                and get.t < bar_time):
                            # Mitigation line
                            all_lines.append(MitigationLine(
                                t1=get.t, price=get.max, t2=bar_time,
                                isbull=True, kind='ob'
                            ))
                            # Remove box and record
                            ob_box = ob_boxes.pop(i)
                            ob_box.deleted = True
                            ob_records.pop(i)
                            # Signal if min_dist satisfied
                            # Pine line 135: if get.idx + min_dist < bar_index
                            if get.idx + self.min_dist < n:
                                signal = Signal(
                                    candle_open=chart_open[n],
                                    candle_high=chart_high[n],
                                    candle_low=chart_low[n],
                                    candle_close=chart_close[n],
                                    point=get.max,
                                    isbull=True,
                                    entry=False,
                                    idx=n, t=bar_time
                                )
                    else:
                        # Pine lines 139-145:
                        # if (s_high >= get.min or high >= get.min)
                        #    and get.t < time
                        if ((s_high_val >= get.min
                             or chart_high[n] >= get.min)
                                and get.t < bar_time):
                            all_lines.append(MitigationLine(
                                t1=get.t, price=get.min, t2=bar_time,
                                isbull=False, kind='ob'
                            ))
                            ob_box = ob_boxes.pop(i)
                            ob_box.deleted = True
                            ob_records.pop(i)
                            if get.idx + self.min_dist < n:
                                signal = Signal(
                                    candle_open=chart_open[n],
                                    candle_high=chart_high[n],
                                    candle_low=chart_low[n],
                                    candle_close=chart_close[n],
                                    point=get.min,
                                    isbull=False,
                                    entry=False,
                                    idx=n, t=bar_time
                                )
                    i -= 1

            # --------------------------------------------------------
            # FVG mitigation check
            # Pine lines 147-166: same pattern as OB mitigation.
            # --------------------------------------------------------
            if len(fvg_records) > 0:
                i = len(fvg_records) - 1
                while i >= 0:
                    get_fvg = fvg_records[i]
                    if get_fvg.isbull:
                        # Pine line 152:
                        # if (fvg_low <= get_fvg.max or low <= get_fvg.max)
                        #    and get_fvg.t < time
                        if ((fvg_low_val <= get_fvg.max
                             or chart_low[n] <= get_fvg.max)
                                and get_fvg.t < bar_time):
                            all_lines.append(MitigationLine(
                                t1=get_fvg.t, price=get_fvg.max,
                                t2=bar_time,
                                isbull=True, kind='fvg'
                            ))
                            fvg_box = fvg_boxes.pop(i)
                            fvg_box.deleted = True
                            fvg_records.pop(i)
                            if get_fvg.idx + self.min_dist_fvg < n:
                                signal_fvg = Signal(
                                    candle_open=chart_open[n],
                                    candle_high=chart_high[n],
                                    candle_low=chart_low[n],
                                    candle_close=chart_close[n],
                                    point=get_fvg.max,
                                    isbull=True,
                                    entry=False,
                                    idx=n, t=bar_time
                                )
                    else:
                        # Pine line 160:
                        # if (fvg_high >= get_fvg.min or high >= get_fvg.min)
                        #    and get_fvg.t < time
                        if ((fvg_high_val >= get_fvg.min
                             or chart_high[n] >= get_fvg.min)
                                and get_fvg.t < bar_time):
                            all_lines.append(MitigationLine(
                                t1=get_fvg.t, price=get_fvg.min,
                                t2=bar_time,
                                isbull=False, kind='fvg'
                            ))
                            fvg_box = fvg_boxes.pop(i)
                            fvg_box.deleted = True
                            fvg_records.pop(i)
                            if get_fvg.idx + self.min_dist_fvg < n:
                                signal_fvg = Signal(
                                    candle_open=chart_open[n],
                                    candle_high=chart_high[n],
                                    candle_low=chart_low[n],
                                    candle_close=chart_close[n],
                                    point=get_fvg.min,
                                    isbull=False,
                                    entry=False,
                                    idx=n, t=bar_time
                                )
                    i -= 1

            # --------------------------------------------------------
            # Signal generation
            # Pine lines 168-190
            # --------------------------------------------------------
            # Pine line 168: candle_dir = close > open ? 1 : -1
            candle_dir = 1 if chart_close[n] > chart_open[n] else -1

            cond = 0
            cond_fvg_val = 0

            # Pine line 173: s_close := use_ha ? ha_close : fvg_close
            # This reassigns s_close for the OB signal comparison.
            # fvg_close == s_close (same ticker/tf), so effectively:
            #   if use_ha -> ha_close, else -> security close
            ob_sig_close = ha_close[n] if self.use_ha else fvg_close_val

            # Pine lines 174-176:
            # if s_close > signal.point and signal.isbull
            #    and candle_dir == 1 and not signal.entry
            #     signal.entry := true
            #     cond := 1
            if (ob_sig_close > signal.point
                    and signal.isbull
                    and candle_dir == 1
                    and not signal.entry):
                signal.entry = True
                cond = 1

            # Pine lines 178-180:
            # if s_close < signal.point and not signal.isbull
            #    and candle_dir == -1 and not signal.entry
            #     signal.entry := true
            #     cond := -1
            if (ob_sig_close < signal.point
                    and not signal.isbull
                    and candle_dir == -1
                    and not signal.entry):
                signal.entry = True
                cond = -1

            # Pine line 182: fvg_close := use_ha ? ha_close : fvg_close
            fvg_sig_close = ha_close[n] if self.use_ha else fvg_close_val

            # Pine lines 184-186:
            if (fvg_sig_close > signal_fvg.point
                    and signal_fvg.isbull
                    and candle_dir == 1
                    and not signal_fvg.entry):
                signal_fvg.entry = True
                cond_fvg_val = 1

            # Pine lines 188-190:
            if (fvg_sig_close < signal_fvg.point
                    and not signal_fvg.isbull
                    and candle_dir == -1
                    and not signal_fvg.entry):
                signal_fvg.entry = True
                cond_fvg_val = -1

            # --------------------------------------------------------
            # Record signals
            # Pine lines 193-197: plotshape with display toggle
            # --------------------------------------------------------
            if self.show_signals_ob:
                ob_signals[n] = cond
            if self.show_signals_fvg:
                fvg_signals[n] = cond_fvg_val

            if cond != 0:
                signal_details_ob[n] = deepcopy(signal)
            if cond_fvg_val != 0:
                signal_details_fvg[n] = deepcopy(signal_fvg)

            # --------------------------------------------------------
            # Update box right edges for active boxes
            # Pine: extend = extend.right on box creation
            # --------------------------------------------------------
            for bx in ob_boxes:
                if not bx.deleted:
                    bx.right = n
                    bx.t_right = bar_time
            for bx in fvg_boxes:
                if not bx.deleted:
                    bx.right = n
                    bx.t_right = bar_time

        # ================================================================
        # Build output
        # ================================================================
        signals_df = pd.DataFrame({
            'ob_signal': ob_signals,
            'fvg_signal': fvg_signals,
        }, index=df.index)

        return {
            'signals': signals_df,
            'ob_records': ob_records,         # still-active OBs
            'fvg_records': fvg_records,       # still-active FVGs
            'boxes': all_boxes,
            'lines': all_lines,
            'signal_details_ob': signal_details_ob,
            'signal_details_fvg': signal_details_fvg,
        }


# ---------------------------------------------------------------------------
# Plotting utility (optional – requires matplotlib)
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, results: Dict, figsize=(20, 10)):
    """Plot OHLC chart with Order Blocks, FVGs, mitigation lines, and signals.

    Parameters
    ----------
    df : pd.DataFrame
        Original OHLC data.
    results : dict
        Output from OrderBlockDetector.run().
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)

    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # --- Candlesticks ---
    for i in range(n_bars):
        color = 'green' if closes[i] >= opens[i] else 'red'
        # Wick
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5)
        # Body
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        rect = mpatches.FancyBboxPatch(
            (i - 0.3, body_bottom), 0.6, body_height,
            boxstyle="square,pad=0", facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)

    # --- Boxes (OBs and FVGs) ---
    for box in results['boxes']:
        if box.deleted:
            left = box.left
            width = box.right - box.left
        else:
            # Still active: extend to end of chart
            left = box.left
            width = n_bars - box.left

        if box.kind == 'ob':
            color = 'green' if box.isbull else 'red'
        else:
            color = 'blue' if box.isbull else 'orange'

        rect = mpatches.FancyBboxPatch(
            (left, box.bottom), width, box.top - box.bottom,
            boxstyle="square,pad=0", facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=0.5
        )
        ax.add_patch(rect)

    # --- Mitigation lines ---
    for line in results['lines']:
        idx1 = df.index.get_indexer([line.t1], method='nearest')[0]
        idx2 = df.index.get_indexer([line.t2], method='nearest')[0]
        if line.kind == 'ob':
            color = 'green' if line.isbull else 'red'
        else:
            color = 'blue' if line.isbull else 'orange'
        ax.plot([idx1, idx2], [line.price, line.price],
                color=color, linestyle='--', linewidth=1)

    # --- Signals ---
    signals_df = results['signals']
    for i in range(n_bars):
        if signals_df['ob_signal'].iloc[i] == 1:
            ax.annotate(
                'Buy', (i, lows[i]), fontsize=7, color='white',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='green',
                          alpha=0.8),
                xytext=(0, -15), textcoords='offset points')
        elif signals_df['ob_signal'].iloc[i] == -1:
            ax.annotate(
                'Sell', (i, highs[i]), fontsize=7, color='white',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='red',
                          alpha=0.8),
                xytext=(0, 15), textcoords='offset points')

        if signals_df['fvg_signal'].iloc[i] == 1:
            ax.annotate(
                'Buy', (i, lows[i]), fontsize=7, color='white',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='blue',
                          alpha=0.8),
                xytext=(0, -30), textcoords='offset points')
        elif signals_df['fvg_signal'].iloc[i] == -1:
            ax.annotate(
                'Sell', (i, highs[i]), fontsize=7, color='white',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='orange',
                          alpha=0.8),
                xytext=(0, 30), textcoords='offset points')

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.3, label='Bull OB'),
        mpatches.Patch(facecolor='red', alpha=0.3, label='Bear OB'),
        mpatches.Patch(facecolor='blue', alpha=0.3, label='Bull FVG'),
        mpatches.Patch(facecolor='orange', alpha=0.3, label='Bear FVG'),
        Line2D([0], [0], color='green', linestyle='--',
               label='Bull OB Mitigated'),
        Line2D([0], [0], color='red', linestyle='--',
               label='Bear OB Mitigated'),
        Line2D([0], [0], color='blue', linestyle='--',
               label='Bull FVG Mitigated'),
        Line2D([0], [0], color='orange', linestyle='--',
               label='Bear FVG Mitigated'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title('Order-Block Detector')
    ax.set_ylabel('Price')
    ax.set_xlabel('Bar Index')
    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Order-Block Detector (Pine Script -> Python)')
    parser.add_argument(
        'csv',
        help='Path to OHLC CSV file. First column must be a parseable '
             'datetime (used as index). Remaining columns: open, high, '
             'low, close (case-insensitive).')
    parser.add_argument(
        '--tf', default='',
        help='Higher timeframe for detection (pandas freq string, e.g. '
             '"5min", "15min", "1h", "4h", "1D"). Default: same as data.')
    parser.add_argument(
        '--min-dist', type=int, default=1,
        help='Min bar distance for OB signals (default: 1)')
    parser.add_argument(
        '--min-dist-fvg', type=int, default=1,
        help='Min bar distance for FVG signals (default: 1)')
    parser.add_argument(
        '--use-ha', action='store_true',
        help='Use Heikin-Ashi close for signal generation')
    parser.add_argument(
        '--no-ob', action='store_true',
        help='Disable Order Block detection')
    parser.add_argument(
        '--no-fvg', action='store_true',
        help='Disable Fair Value Gap detection')
    parser.add_argument(
        '--no-signals-ob', action='store_true',
        help='Disable OB buy/sell signals')
    parser.add_argument(
        '--no-signals-fvg', action='store_true',
        help='Disable FVG buy/sell signals')
    parser.add_argument(
        '--plot', action='store_true',
        help='Plot results (requires matplotlib)')

    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]

    detector = OrderBlockDetector(
        tf=args.tf,
        show_ob=not args.no_ob,
        show_fvg=not args.no_fvg,
        show_signals_ob=not args.no_signals_ob,
        show_signals_fvg=not args.no_signals_fvg,
        min_dist=args.min_dist,
        min_dist_fvg=args.min_dist_fvg,
        use_ha=args.use_ha,
    )

    results = detector.run(df)

    # Print summary
    signals = results['signals']
    buy_ob = (signals['ob_signal'] == 1).sum()
    sell_ob = (signals['ob_signal'] == -1).sum()
    buy_fvg = (signals['fvg_signal'] == 1).sum()
    sell_fvg = (signals['fvg_signal'] == -1).sum()

    print("Order-Block Detector Results")
    print("============================")
    print(f"Total bars processed:       {len(df)}")
    print(f"Active (unmitigated) OBs:   {len(results['ob_records'])}")
    print(f"Active (unmitigated) FVGs:  {len(results['fvg_records'])}")
    print(f"OB  Buy  signals:           {buy_ob}")
    print(f"OB  Sell signals:           {sell_ob}")
    print(f"FVG Buy  signals:           {buy_fvg}")
    print(f"FVG Sell signals:           {sell_fvg}")
    print(f"Mitigation events:          {len(results['lines'])}")

    # Print signal bars
    signal_bars = signals[
        (signals['ob_signal'] != 0) | (signals['fvg_signal'] != 0)
    ]
    if not signal_bars.empty:
        print(f"\nSignal bars:")
        print(signal_bars.to_string())

    if args.plot:
        plot_results(df, results)
