"""Python translation of `SMC Target Liquidity V.35 (Manual Distance Control)`.

Source: `S1_Silver_Bullet/SMC Target Liquidity V.35 (Manual Distance Control).txt`.

Detection logic, priority order and parameter defaults are mirrored 1:1 with
the Pine source. Purely cosmetic Pine settings (colours, line styles,
`flip_text`, `lbl_offset`, label sizes, `line_extension`) are dropped.

Classification priority per bar
-------------------------------
For a **support** level (`active_buy_line`) at price ``lvl``:
    1. **SFP**  ``close > lvl  AND  close[1] < lvl``      (bullish reclaim)
    2. **MSS**  ``close < lvl  AND  close[1] < lvl  AND  close < close[1]``
       (second bar of a breakdown — confirms bearish shift)
    3. **X**    ``low   <= lvl  AND  close > lvl``        (wick-only sweep)
    4. Otherwise the line keeps waiting.

For a **resistance** level (`active_sell_line`) at price ``lvl``:
    1. **SFP**  ``close < lvl  AND  close[1] > lvl``
    2. **MSS**  ``close > lvl  AND  close[1] > lvl  AND  close > close[1]``
    3. **X**    ``high  >= lvl  AND  close < lvl``

The pivot filter (`is_broken`) that walks the right-side bars is preserved
verbatim even though it is a no-op under strict `ta.pivothigh/pivotlow`
semantics — we keep it so future tweaks to pivot strictness don't silently
change behaviour.

Session tagging
---------------
Sessions are represented as ``HHMM-HHMM`` strings anchored in ``tz_manual``.
When the input DataFrame is daily (or slower), *every* pivot resolves to
``"other"`` — identical to Pine on a daily chart — and the session
information simply contributes no differentiating signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Settings / data types
# ---------------------------------------------------------------------------
@dataclass
class SMCTargetSettings:
    """Mirror of the Pine `input.*` knobs that affect detection."""

    prd: int = 10
    max_active_lines: int = 5
    tz_manual: str = "UTC-4"
    asia_sess: str = "2000-0000"
    london_sess: str = "0200-0500"
    ny_am_sess: str = "0830-1100"
    ny_pm_sess: str = "1330-1600"


@dataclass
class TargetLine:
    """An active pivot-based liquidity level awaiting resolution."""

    side: str  # "support" | "resistance"
    pivot_time: pd.Timestamp  # timestamp of the pivot centre bar
    pivot_price: float
    created_time: pd.Timestamp  # bar that confirmed the pivot
    session: str  # "ny_pm" | "ny_am" | "london" | "asia" | "other"


@dataclass
class TargetEvent:
    """Resolution of a pivot-based liquidity level."""

    resolution: str  # "SFP" | "MSS" | "X"
    side: str  # "support" | "resistance"
    pivot_time: pd.Timestamp
    pivot_price: float
    session: str
    event_time: pd.Timestamp
    event_price: float  # close at resolution


@dataclass
class SMCTargetResult:
    events: List[TargetEvent] = field(default_factory=list)
    active_buy_lines: List[TargetLine] = field(default_factory=list)
    active_sell_lines: List[TargetLine] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session helper — mirrors Pine's `time(timeframe.period, sess + ":1234567",
# tz_manual)` for the common Pine session syntax `HHMM-HHMM`.
# ---------------------------------------------------------------------------
def _parse_tz_offset(tz: str) -> Optional[timedelta]:
    """Accept a subset of Pine tz strings; return timedelta or None if unknown.

    Supported forms:
      * ``"UTC"``, ``"UTC-4"``, ``"UTC+5"``, ``"UTC-05:30"`` etc.
      * Tz-database names like ``"America/New_York"``, ``"Asia/Bangkok"``.
    """
    t = tz.strip()
    if t.upper().startswith("UTC"):
        rest = t[3:].strip()
        if not rest:
            return timedelta(0)
        sign = 1 if rest[0] == "+" else -1 if rest[0] == "-" else None
        if sign is None:
            return None
        body = rest[1:]
        if ":" in body:
            h, m = body.split(":", 1)
            return sign * (timedelta(hours=int(h), minutes=int(m)))
        return sign * timedelta(hours=int(body))
    # Fall back to zoneinfo for named tz.
    try:
        from zoneinfo import ZoneInfo

        # Convert a reference UTC instant into the named zone and read the
        # offset. We use "now" but that's fine — Pine sessions also ignore DST
        # transitions at the tick level.
        zi = ZoneInfo(t)
        utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        local = utc_now.astimezone(zi)
        return local.utcoffset()
    except Exception:
        return None


def _parse_session(sess: str) -> Optional[tuple[dtime, dtime]]:
    """Parse ``"HHMM-HHMM"`` (Pine session syntax) → (start, end) times."""
    try:
        start_str, end_str = sess.split("-")
        start = dtime(int(start_str[:2]), int(start_str[2:]))
        end = dtime(int(end_str[:2]), int(end_str[2:]))
        return start, end
    except Exception:
        return None


def _in_session(ts: pd.Timestamp, sess: str, tz_offset: Optional[timedelta]) -> bool:
    """Pine: `not na(time(timeframe.period, sess + ":1234567", tz_manual))`.

    Pine considers the bar "in session" if its start time falls within the
    session window **in the user's timezone**. On weekdays only (``:1234567``
    means all 7 days in Pine, so we ignore the weekday mask).
    """
    parsed = _parse_session(sess)
    if parsed is None:
        return False
    start, end = parsed

    # Convert bar timestamp to the user's timezone.
    if tz_offset is None:
        local_time = ts.to_pydatetime().time()
    else:
        # If ts is tz-aware convert properly; otherwise treat it as UTC.
        py = ts.to_pydatetime()
        if py.tzinfo is None:
            py = py.replace(tzinfo=None)
            py = py  # treat as UTC reference
        else:
            # Strip tz and re-anchor to UTC for consistent arithmetic.
            py = py.astimezone(tz=None).replace(tzinfo=None)
        local_time = (py + tz_offset).time()

    # Pine session wraps around midnight when end <= start (e.g. 2000-0000).
    if start <= end:
        return start <= local_time < end
    return local_time >= start or local_time < end


# ---------------------------------------------------------------------------
# Daily / intraday detector
# ---------------------------------------------------------------------------
def _is_daily_or_slower(index: pd.DatetimeIndex) -> bool:
    """Heuristic: median bar spacing >= 1 day → treat as daily (or coarser)."""
    if len(index) < 2:
        return True
    # Drop weekend gaps via abs diff in seconds.
    diffs = (index[1:] - index[:-1]).total_seconds()
    # Median is robust to holidays / weekend gaps (which are large) and to
    # the occasional missing bar.
    med = float(pd.Series(diffs).median())
    return med >= 86_000.0  # 1 day = 86_400s; 86_000 threshold absorbs DST slop.


# ---------------------------------------------------------------------------
# Strict pivot detection helpers (copied from the UAlgo module to keep this
# file self-contained).
# ---------------------------------------------------------------------------
def _is_strict_pivot_high(series: pd.Series, centre_idx: int, window: int) -> bool:
    lo = centre_idx - window
    hi = centre_idx + window
    if lo < 0 or hi >= len(series):
        return False
    pivot = series.iat[centre_idx]
    for j in range(lo, hi + 1):
        if j == centre_idx:
            continue
        if series.iat[j] >= pivot:
            return False
    return True


def _is_strict_pivot_low(series: pd.Series, centre_idx: int, window: int) -> bool:
    lo = centre_idx - window
    hi = centre_idx + window
    if lo < 0 or hi >= len(series):
        return False
    pivot = series.iat[centre_idx]
    for j in range(lo, hi + 1):
        if j == centre_idx:
            continue
        if series.iat[j] <= pivot:
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_smc_target_liquidity(
    df: pd.DataFrame,
    settings: Optional[SMCTargetSettings] = None,
) -> SMCTargetResult:
    """Run `SMC Target Liquidity V.35` on an OHLC DataFrame.

    Input
    -----
    df : DatetimeIndex (ascending) + columns ``high``, ``low``, ``close``.
    """
    s = settings or SMCTargetSettings()
    if s.prd < 2:
        raise ValueError("prd must be >= 2 to match Pine's `minval=2`")
    if s.max_active_lines < 1:
        raise ValueError("max_active_lines must be >= 1")
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
    if len(df) < 2 * s.prd + 2:
        return SMCTargetResult()

    highs, lows, closes = df["high"], df["low"], df["close"]
    index = df.index
    prd = s.prd

    tz_offset = _parse_tz_offset(s.tz_manual)

    # Pine's session filter is only meaningful on intraday charts. On daily
    # bars every tick is at the same time-of-day (the bar open), so the tag
    # carries no information and — worse — the midnight-UTC timestamps that
    # FMP emits would spuriously tag every pivot as "asia" under UTC-4.
    # Detect a daily-or-slower DataFrame and short-circuit to "other".
    is_daily_or_slower = _is_daily_or_slower(df.index)

    def _session_of(ts: pd.Timestamp) -> str:
        if is_daily_or_slower:
            return "other"
        if _in_session(ts, s.ny_pm_sess, tz_offset):
            return "ny_pm"
        if _in_session(ts, s.ny_am_sess, tz_offset):
            return "ny_am"
        if _in_session(ts, s.london_sess, tz_offset):
            return "london"
        if _in_session(ts, s.asia_sess, tz_offset):
            return "asia"
        return "other"

    buy_lines: List[TargetLine] = []
    sell_lines: List[TargetLine] = []
    events: List[TargetEvent] = []

    for i in range(len(df)):
        centre_idx = i - prd
        t_i = index[i]
        close_i = float(closes.iat[i])
        close_prev = float(closes.iat[i - 1]) if i > 0 else close_i
        high_i = float(highs.iat[i])
        low_i = float(lows.iat[i])

        # ---- (1) Pivot detection at confirm bar ----
        if centre_idx >= prd:
            # --- Buy line (pivot low) ---
            if _is_strict_pivot_low(lows, centre_idx, prd):
                pl = float(lows.iat[centre_idx])
                # Pine `is_broken` filter: for j = 0..prd-1 check low[j] < pl
                # (low[0] is current bar, low[prd-1] is 1 bar after the centre).
                # Under strict pivotlow semantics this is always False; kept
                # for exact parity with Pine source.
                is_broken = False
                for j in range(0, prd):
                    if float(lows.iat[i - j]) < pl:
                        is_broken = True
                        break
                if not is_broken:
                    sess = _session_of(index[centre_idx])
                    buy_lines.append(
                        TargetLine(
                            side="support",
                            pivot_time=index[centre_idx],
                            pivot_price=pl,
                            created_time=t_i,
                            session=sess,
                        )
                    )
                    if len(buy_lines) > s.max_active_lines:
                        buy_lines.pop(0)  # Pine `array.shift` drops oldest

            # --- Sell line (pivot high) ---
            if _is_strict_pivot_high(highs, centre_idx, prd):
                ph = float(highs.iat[centre_idx])
                is_broken = False
                for j in range(0, prd):
                    if float(highs.iat[i - j]) > ph:
                        is_broken = True
                        break
                if not is_broken:
                    sess = _session_of(index[centre_idx])
                    sell_lines.append(
                        TargetLine(
                            side="resistance",
                            pivot_time=index[centre_idx],
                            pivot_price=ph,
                            created_time=t_i,
                            session=sess,
                        )
                    )
                    if len(sell_lines) > s.max_active_lines:
                        sell_lines.pop(0)

        # ---- (2) Resolve buy lines (supports) — priority SFP > MSS > X ----
        for k in range(len(buy_lines) - 1, -1, -1):
            lvl = buy_lines[k]
            price = lvl.pivot_price
            if close_i > price and close_prev < price:
                events.append(
                    TargetEvent(
                        resolution="SFP",
                        side="support",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        session=lvl.session,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                buy_lines.pop(k)
            elif close_i < price and close_prev < price and close_i < close_prev:
                events.append(
                    TargetEvent(
                        resolution="MSS",
                        side="support",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        session=lvl.session,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                buy_lines.pop(k)
            elif low_i <= price and close_i > price:
                events.append(
                    TargetEvent(
                        resolution="X",
                        side="support",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        session=lvl.session,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                buy_lines.pop(k)

        # ---- (3) Resolve sell lines (resistances) — priority SFP > MSS > X
        for k in range(len(sell_lines) - 1, -1, -1):
            lvl = sell_lines[k]
            price = lvl.pivot_price
            if close_i < price and close_prev > price:
                events.append(
                    TargetEvent(
                        resolution="SFP",
                        side="resistance",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        session=lvl.session,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                sell_lines.pop(k)
            elif close_i > price and close_prev > price and close_i > close_prev:
                events.append(
                    TargetEvent(
                        resolution="MSS",
                        side="resistance",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        session=lvl.session,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                sell_lines.pop(k)
            elif high_i >= price and close_i < price:
                events.append(
                    TargetEvent(
                        resolution="X",
                        side="resistance",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        session=lvl.session,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                sell_lines.pop(k)

    return SMCTargetResult(
        events=events,
        active_buy_lines=list(buy_lines),
        active_sell_lines=list(sell_lines),
    )


__all__ = [
    "SMCTargetSettings",
    "TargetLine",
    "TargetEvent",
    "SMCTargetResult",
    "compute_smc_target_liquidity",
]
