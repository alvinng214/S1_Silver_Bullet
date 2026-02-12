"""ICT MTF Order Block Wicks [MK] — Backtrader Implementation.

Line-by-line parity translation of the Pine Script v5 indicator
``ICT MTF Order Block Wicks [MK]`` (by @malk1903) into a backtrader-compatible
class.  Every detection gate, mitigation branch, intrusion threshold, duplicate
check, and alert flag from the original script is preserved.

Architecture
------------
In Pine the indicator uses ``request.security`` to pull HTF OHLC on every
chart bar.  In backtrader the equivalent is achieved by adding **resampled
data feeds** to cerebro and passing them to this class via *tf_datas*.

Detection of new OBs uses HTF data (fires when a new HTF bar completes).
Mitigation / intrusion checks run on every chart-timeframe bar, exactly as
in the Pine original where ``low``, ``high``, ``close`` refer to chart values.

Usage
-----
::

    import backtrader as bt

    class MyStrategy(bt.Strategy):
        def __init__(self):
            # data0 is the chart timeframe (e.g. 5-min)
            # Add resampled feeds in cerebro before running:
            #   cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=10)
            self.ob = ICTMTFOrderBlockWicks(
                tf_datas={
                    '10 Min': self.data1,
                    '15 Min': self.data2,
                    '30 Min': self.data3,
                    '1 Hr':   self.data4,
                },
            )

        def next(self):
            self.ob.update(
                bar_index=len(self.data0),
                high=self.data0.high[0],
                low=self.data0.low[0],
                close=self.data0.close[0],
                last_high=self.data0.high[-1] if len(self.data0) > 1 else self.data0.high[0],
                last_low=self.data0.low[-1] if len(self.data0) > 1 else self.data0.low[0],
            )

            for zone in self.ob.get_all_bull_zones():
                ...  # e.g. use zone.top, zone.bottom for strategy logic

            for zone in self.ob.get_all_bear_zones():
                ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# Pine-like runtime structures
# -----------------------------------------------------------------------

@dataclass
class OBZone:
    """Single Order Block zone — mirrors a Pine ``box`` plus its label."""
    top: float
    bottom: float
    tf_label: str           # e.g. "10 Min", "1 Hr", "Daily"
    is_bull: bool
    creation_bar: int       # chart bar_index when created
    bgcolor: str
    border_color: str
    border_width: int = 1
    border_style: str = "dotted"
    label_text: str = ""
    label_y: float = 0.0


@dataclass
class TFState:
    """Per-timeframe state — mirrors Pine's per-TF box/label arrays."""
    bull_zones: List[OBZone] = field(default_factory=list)
    bear_zones: List[OBZone] = field(default_factory=list)
    new_bull: bool = False
    new_bear: bool = False
    prev_htf_len: int = 0   # tracks when a new HTF bar forms


# -----------------------------------------------------------------------
# Helper look-ups (Pine _gettf_interval_str / _gettf_label_str)
# -----------------------------------------------------------------------

TF_INDEX_TO_INTERVAL = {
    0: "5", 1: "10", 2: "15", 3: "30", 4: "60",
    5: "240", 6: "480", 7: "720", 8: "D", 9: "W", 10: "M",
}

TF_INDEX_TO_LABEL = {
    0: "5 Min", 1: "10 Min", 2: "15 Min", 3: "30 Min", 4: "1 Hr",
    5: "4 Hr", 6: "8 Hr", 7: "12Hr", 8: "Daily", 9: "Weekly", 10: "Monthly",
}


# -----------------------------------------------------------------------
# Main indicator class
# -----------------------------------------------------------------------

class ICTMTFOrderBlockWicks:
    """Backtrader-compatible ICT MTF Order Block Wicks indicator.

    Parameters mirror the Pine Script's ``input.*`` defaults.  Every branch
    of the original logic (detection, mitigation, intrusion, colour change,
    duplicate prevention, per-TF max limits, alert flags) is reproduced.
    """

    def __init__(
        self,
        tf_datas: Dict[str, object],
        *,
        # ---- Core toggles (Pine lines 11-16) ----
        display: bool = True,
        only_market_hours: bool = False,          # OnlyMktHrs
        fvgmethod_body: bool = True,              # fmthds == "Body"
        show_labels: bool = True,
        show_timeonlabels: bool = False,

        # ---- Label / offset (Pine lines 19-23) ----
        hours_offset_input: float = -5.0,
        label_shift: int = 10,
        label_color: str = "orange",

        # ---- Intrusion / incursion (Pine lines 23-25) ----
        incursion_alerts: bool = True,
        incursion_pct: int = 20,

        # ---- Mitigation (Pine lines 27-43) ----
        mitigation_mode: str = "Normal",          # Normal|Dynamic|None|Half
        show_mitigated_text: bool = False,
        use_body_for_mitigation: bool = False,     # mitig_type default "Wicks"

        # ---- Entry colour change (Pine lines 44-46) ----
        entrychangecolor: bool = True,
        entry_bull_color: str = "white_90",
        entry_bear_color: str = "white_90",

        # ---- OB fill colours (Pine lines 108-109) ----
        bullfvgcolor: str = "yellow_80",
        bearfvgcolor: str = "blue_80",
        nomiticolor: str = "yellow_85",

        # ---- Border (Pine lines 114-118) ----
        box_border_bull_color: str = "yellow_100",
        box_border_bear_color: str = "blue_100",
        box_border_width: int = 1,
        box_border_style: str = "dotted",

        # ---- Per-TF max OBs (Pine lines 84-95) ----
        max_per_tf: Optional[Dict[str, int]] = None,
        default_max_per_tf: int = 8,

        # ---- Session filter callback (optional) ----
        # If only_market_hours is True, provide a callable(datetime)->bool
        # to replicate Pine's ``not na(time(timeframe.period, "0930-1600"))``
        session_filter: Optional[object] = None,
    ):
        # Store every setting verbatim ----------------------------------
        self.display = display
        self.only_market_hours = only_market_hours
        self.fvgmethod_body = fvgmethod_body
        self.show_labels = show_labels
        self.show_timeonlabels = show_timeonlabels
        self.hours_offset_input = hours_offset_input
        self.label_shift = label_shift
        self.label_color = label_color
        self.incursion_alerts = incursion_alerts
        self.incursion_pct = incursion_pct
        self.intrusion_percentage = incursion_pct / 100.0   # Pine line 25
        self.show_mitigated_text = show_mitigated_text
        self.use_body_for_mitigation = use_body_for_mitigation
        self.entrychangecolor = entrychangecolor
        self.entry_bull_color = entry_bull_color
        self.entry_bear_color = entry_bear_color
        self.bullfvgcolor = bullfvgcolor
        self.bearfvgcolor = bearfvgcolor
        self.nomiticolor = nomiticolor
        self.box_border_bull_color = box_border_bull_color
        self.box_border_bear_color = box_border_bear_color
        self.box_border_width = box_border_width
        self.box_border_style = box_border_style
        self.session_filter = session_filter

        # Pine mitigation_mode → integer (Pine lines 28-37)
        self.mitigationaction = self._resolve_mitigation_mode(mitigation_mode)

        # TF data feeds and per-TF state --------------------------------
        # tf_datas: {"10 Min": bt_data_feed, "15 Min": ..., ...}
        self.tf_datas: Dict[str, object] = tf_datas

        # Per-TF max OB limits
        self._max_per_tf: Dict[str, int] = max_per_tf or {}
        self._default_max_per_tf = default_max_per_tf

        # Initialise per-TF state (Pine lines 457-539)
        self.states: Dict[str, TFState] = {
            label: TFState() for label in tf_datas
        }

        # Alert / event accumulators (reset each bar) -------------------
        self.incursion_messages: List[str] = []
        self.bar_events: List[dict] = []

        # Aggregate alert flags (Pine lines 609-614) --------------------
        self.bull_creation_alert: bool = False
        self.bear_creation_alert: bool = False
        self.both_creation_alert: bool = False

    # -------------------------------------------------------------------
    # Static helpers mirroring Pine functions
    # -------------------------------------------------------------------

    @staticmethod
    def _resolve_mitigation_mode(mode: str) -> int:
        """Pine lines 28-37: mitiaction → integer."""
        if mode == "Normal":
            return 1
        if mode == "Dynamic":
            return 2
        if mode == "None":
            return 3
        # "Half" or anything else
        return 4

    # Pine lines 174-181: _isfvgbull
    def _isfvgbull(
        self, open1: float, close1: float, op: float, cl: float,
        high1: float, low1: float,
    ) -> bool:
        if not self.display:
            return False
        if self.fvgmethod_body:
            # Body method: prev bar bearish, current bar bullish,
            # current close > previous high
            return open1 > close1 and op < cl and cl > high1
        else:
            # Wick method
            return op < high1

    # Pine lines 184-191: _isfvgbear
    def _isfvgbear(
        self, open1: float, close1: float, op: float, cl: float,
        high1: float, low1: float,
    ) -> bool:
        if not self.display:
            return False
        if self.fvgmethod_body:
            # Body method: prev bar bullish, current bar bearish,
            # current close < previous low
            return open1 < close1 and op > cl and cl < low1
        else:
            # Wick method
            return op < low1

    # Pine lines 401-409: _duplicate_box — only compares top
    @staticmethod
    def _duplicate_box(zones: List[OBZone], top: float) -> bool:
        return any(z.top == top for z in zones)

    # Pine lines 235-245: _getbullfvgaction
    def _get_bull_action(
        self, top: float, bottom: float, low: float, close: float,
        last_low: float,
    ) -> dict:
        midpt = (top + bottom) / 2.0
        threshold = top - (self.intrusion_percentage * (top - bottom))
        return {
            "have_intrusion": low < threshold and last_low > threshold,
            "lowundertop": low < top,
            "lowunderbtm": low < bottom,
            "lowundermid": low < midpt,
            "closeundertop": close < top,
            "closeunderbtm": close < bottom,
            "closeundermid": low < midpt,     # Pine parity: uses low, not close
        }

    # Pine lines 248-258: _getbearfvgaction
    def _get_bear_action(
        self, top: float, bottom: float, high: float, close: float,
        last_high: float,
    ) -> dict:
        midpt = (top + bottom) / 2.0
        threshold = bottom + (self.intrusion_percentage * (top - bottom))
        return {
            "have_intrusion": high > threshold and last_high < threshold,
            "highovertop": high > top,
            "highoverbtm": high > bottom,
            "highovermid": high > midpt,
            "closeovertop": close > top,
            "closeoverbtm": close > bottom,
            "closeovermid": close > midpt,
        }

    # -------------------------------------------------------------------
    # Bull OB update — Pine lines 280-338: _update_bull_fvgs
    # -------------------------------------------------------------------
    def _update_bull_zones(
        self, st: TFState, tf_label: str, bar_index: int,
        low: float, close: float, last_low: float,
    ) -> None:
        i = len(st.bull_zones) - 1
        while i >= 0:
            zone = st.bull_zones[i]
            top = zone.top
            bottom = zone.bottom
            a = self._get_bull_action(top, bottom, low, close, last_low)

            # --- Incursion alert (Pine lines 286-287) ---
            if (self.mitigationaction in (1, 3)) and a["have_intrusion"] and self.incursion_alerts:
                self.incursion_messages.append(
                    f"Bull OB Wick Incursion {tf_label}"
                )

            # --- Entry colour change (Pine lines 289-291) ---
            if self.entrychangecolor:
                if a["lowundertop"]:
                    zone.bgcolor = self.entry_bull_color

            # --- Label / box position update (Pine lines 293-295) ---
            if self.show_labels:
                zone.label_y = (zone.top + zone.bottom) / 2.0

            # --- Dynamic mitigation: shrink zone (Pine lines 298-307) ---
            if self.mitigationaction == 2 and self.use_body_for_mitigation and a["closeundertop"]:
                zone.top = close
                if self.show_labels:
                    zone.label_y = (close + bottom) / 2.0
            elif self.mitigationaction == 2 and a["lowundertop"]:
                zone.top = low
                if self.show_labels:
                    zone.label_y = (low + bottom) / 2.0

            # --- 'None' mitigation: change colour, add text (Pine lines 310-323) ---
            if self.mitigationaction == 3:
                mitigated_by_body = self.use_body_for_mitigation and a["closeunderbtm"]
                mitigated_by_wick = a["lowunderbtm"]
                if mitigated_by_body or mitigated_by_wick:
                    zone.bgcolor = self.nomiticolor
                    if self.show_labels and self.show_mitigated_text:
                        if "Mitigated" not in zone.label_text:
                            zone.label_text += " Mitigated"

            # --- Normal / Dynamic delete (Pine lines 325-330) ---
            delete_now = False
            if self.mitigationaction in (1, 2):
                if self.use_body_for_mitigation and a["closeunderbtm"]:
                    delete_now = True
                elif a["lowunderbtm"]:
                    delete_now = True

            # --- Half mitigation delete (Pine lines 332-337) ---
            if self.mitigationaction == 4:
                if self.use_body_for_mitigation and a["closeundermid"]:
                    delete_now = True
                elif a["lowundermid"]:
                    delete_now = True

            if delete_now:
                st.bull_zones.pop(i)
            i -= 1

    # -------------------------------------------------------------------
    # Bear OB update — Pine lines 340-398: _update_bear_fvgs
    # -------------------------------------------------------------------
    def _update_bear_zones(
        self, st: TFState, tf_label: str, bar_index: int,
        high: float, close: float, last_high: float,
    ) -> None:
        i = len(st.bear_zones) - 1
        while i >= 0:
            zone = st.bear_zones[i]
            top = zone.top
            bottom = zone.bottom
            a = self._get_bear_action(top, bottom, high, close, last_high)

            # --- Incursion alert (Pine lines 345-347) ---
            if (self.mitigationaction in (1, 3)) and a["have_intrusion"] and self.incursion_alerts:
                self.incursion_messages.append(
                    f"Bear OB Wick Incursion {tf_label}"
                )

            # --- Entry colour change (Pine lines 349-353) ---
            if self.entrychangecolor:
                if a["highoverbtm"]:
                    zone.bgcolor = self.entry_bear_color
                else:
                    zone.bgcolor = self.bearfvgcolor

            # --- Label position update (Pine lines 355-357) ---
            if self.show_labels:
                zone.label_y = (zone.top + zone.bottom) / 2.0

            # --- Dynamic mitigation: shrink zone (Pine lines 360-369) ---
            if self.mitigationaction == 2 and self.use_body_for_mitigation and a["closeoverbtm"]:
                zone.bottom = close
                if self.show_labels:
                    zone.label_y = (close + bottom) / 2.0
            elif self.mitigationaction == 2 and a["highoverbtm"]:
                zone.bottom = high
                if self.show_labels:
                    zone.label_y = (top + high) / 2.0

            # --- 'None' mitigation: change colour, add text (Pine lines 371-384) ---
            if self.mitigationaction == 3:
                mitigated_by_body = self.use_body_for_mitigation and a["closeovertop"]
                mitigated_by_wick = a["highovertop"]
                if mitigated_by_body or mitigated_by_wick:
                    zone.bgcolor = self.nomiticolor
                    if self.show_labels and self.show_mitigated_text:
                        if "Mitigated" not in zone.label_text:
                            zone.label_text += " Mitigated"

            # --- Normal / Dynamic delete (Pine lines 386-391) ---
            delete_now = False
            if self.mitigationaction in (1, 2):
                if self.use_body_for_mitigation and a["closeovertop"]:
                    delete_now = True
                elif a["highovertop"]:
                    delete_now = True

            # --- Half mitigation delete (Pine lines 393-397) ---
            if self.mitigationaction == 4:
                if self.use_body_for_mitigation and a["closeovermid"]:
                    delete_now = True
                elif a["highovermid"]:
                    delete_now = True

            if delete_now:
                st.bear_zones.pop(i)
            i -= 1

    # -------------------------------------------------------------------
    # Per-TF handler — Pine lines 412-453: _handle_all
    # -------------------------------------------------------------------
    def _handle_tf(
        self,
        tf_label: str,
        data_feed,
        st: TFState,
        max_zones: int,
        bar_index: int,
        chart_low: float,
        chart_high: float,
        chart_close: float,
        chart_last_low: float,
        chart_last_high: float,
        in_session: bool,
    ) -> None:
        """Process one timeframe: detect new OBs from HTF data, then update
        existing OBs against chart-timeframe price action."""

        st.new_bull = False
        st.new_bear = False

        # ---- Gate: session filter (Pine line 415) ----
        if self.only_market_hours and not in_session:
            # When market hours required but not in session, skip detection
            pass
        else:
            # ---- Check for new HTF bar ----
            current_htf_len = len(data_feed)
            new_htf_bar = current_htf_len > st.prev_htf_len and current_htf_len >= 2
            st.prev_htf_len = current_htf_len

            if new_htf_bar:
                # HTF OHLC for detection:
                # Pine request.security returns [open[1], close[1], open[0], close[0], high[1], low[1]]
                # In backtrader, data_feed[0] = just-completed bar, [-1] = previous bar
                try:
                    open1 = data_feed.open[-1]      # previous HTF bar open
                    close1 = data_feed.close[-1]    # previous HTF bar close
                    high1 = data_feed.high[-1]      # previous HTF bar high
                    low1 = data_feed.low[-1]        # previous HTF bar low
                    op = data_feed.open[0]           # current (just-completed) HTF bar open
                    cl = data_feed.close[0]          # current (just-completed) HTF bar close
                except IndexError:
                    # Not enough bars yet
                    return

                # ---- Bull OB detection (Pine lines 416-429) ----
                new_bull = self._isfvgbull(open1, close1, op, cl, high1, low1)
                # ---- Bear OB detection (Pine lines 417, 431-440) ----
                new_bear = self._isfvgbear(open1, close1, op, cl, high1, low1)

                st.new_bull = new_bull
                st.new_bear = new_bear

                if new_bull:
                    # Enforce max size (Pine lines 421-422)
                    if len(st.bull_zones) > max_zones:
                        st.bull_zones.pop(0)

                    # Duplicate check — only compares top (Pine line 424)
                    if not self._duplicate_box(st.bull_zones, high1):
                        # Bull OB: top = high1, bottom = open1 (Pine line 425)
                        zone = OBZone(
                            top=high1,
                            bottom=open1,
                            tf_label=tf_label,
                            is_bull=True,
                            creation_bar=bar_index,
                            bgcolor=self.bullfvgcolor,
                            border_color=self.box_border_bull_color,
                            border_width=self.box_border_width,
                            border_style=self.box_border_style,
                            label_text=f"{tf_label} OB BULL",
                            label_y=(high1 + low1) / 2.0,
                        )
                        st.bull_zones.append(zone)

                        self.bar_events.append({
                            "bar_index": bar_index,
                            "timeframe": tf_label,
                            "type": "bull_ob_created",
                            "top": high1,
                            "bottom": open1,
                        })

                if new_bear:
                    # Enforce max size (Pine lines 432-433)
                    if len(st.bear_zones) > max_zones:
                        st.bear_zones.pop(0)

                    # Duplicate check — only compares top (Pine line 435)
                    # For bear: top = open1, bottom = low1
                    if not self._duplicate_box(st.bear_zones, open1):
                        zone = OBZone(
                            top=open1,
                            bottom=low1,
                            tf_label=tf_label,
                            is_bull=False,
                            creation_bar=bar_index,
                            bgcolor=self.bearfvgcolor,
                            border_color=self.box_border_bear_color,
                            border_width=self.box_border_width,
                            border_style=self.box_border_style,
                            label_text=f"{tf_label} OB BEAR",
                            label_y=(high1 + low1) / 2.0,
                        )
                        st.bear_zones.append(zone)

                        self.bar_events.append({
                            "bar_index": bar_index,
                            "timeframe": tf_label,
                            "type": "bear_ob_created",
                            "top": open1,
                            "bottom": low1,
                        })

        # ---- Update existing OBs against chart-TF prices (Pine lines 444-451) ----
        if st.bull_zones:
            self._update_bull_zones(
                st, tf_label, bar_index,
                chart_low, chart_close, chart_last_low,
            )
        if st.bear_zones:
            self._update_bear_zones(
                st, tf_label, bar_index,
                chart_high, chart_close, chart_last_high,
            )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def update(
        self,
        bar_index: int,
        high: float,
        low: float,
        close: float,
        last_high: float,
        last_low: float,
        in_session: bool = True,
    ) -> None:
        """Call once per chart-timeframe bar from ``Strategy.next()``.

        Parameters
        ----------
        bar_index : int
            ``len(self.data0)`` — current chart bar count.
        high, low, close : float
            Current chart bar OHLC (``self.data0.high[0]`` etc.).
        last_high, last_low : float
            Previous chart bar high/low (``self.data0.high[-1]`` etc.).
            Pine uses ``high[1]`` / ``low[1]`` as ``lasthigh`` / ``lastlow``
            (lines 231-232).
        in_session : bool
            True when inside the session window.  Mirrors
            ``not na(time(timeframe.period, "0930-1600"))``.
            When ``only_market_hours`` is False (default), this is ignored.
            A ``session_filter`` callback passed to ``__init__`` can also
            be used; pass its return value here.
        """
        # Reset per-bar accumulators
        self.incursion_messages = []
        self.bar_events = []

        # Process every enabled timeframe (Pine lines 550-604)
        for tf_label, data_feed in self.tf_datas.items():
            st = self.states[tf_label]
            max_zones = self._max_per_tf.get(tf_label, self._default_max_per_tf)

            self._handle_tf(
                tf_label=tf_label,
                data_feed=data_feed,
                st=st,
                max_zones=max_zones,
                bar_index=bar_index,
                chart_low=low,
                chart_high=high,
                chart_close=close,
                chart_last_low=last_low,
                chart_last_high=last_high,
                in_session=in_session,
            )

        # Aggregate alert flags (Pine lines 609-614)
        self.bull_creation_alert = any(
            st.new_bull for st in self.states.values()
        )
        self.bear_creation_alert = any(
            st.new_bear for st in self.states.values()
        )
        self.both_creation_alert = self.bull_creation_alert or self.bear_creation_alert

    # ---- Zone accessors ------------------------------------------------

    def get_all_bull_zones(self) -> List[OBZone]:
        """Return all active bull OB zones across every enabled timeframe."""
        result: List[OBZone] = []
        for st in self.states.values():
            result.extend(st.bull_zones)
        return result

    def get_all_bear_zones(self) -> List[OBZone]:
        """Return all active bear OB zones across every enabled timeframe."""
        result: List[OBZone] = []
        for st in self.states.values():
            result.extend(st.bear_zones)
        return result

    def get_all_zones(self) -> List[OBZone]:
        """Return all active OB zones (bull + bear) across every TF."""
        return self.get_all_bull_zones() + self.get_all_bear_zones()

    def get_zones_for_tf(self, tf_label: str) -> Tuple[List[OBZone], List[OBZone]]:
        """Return ``(bull_zones, bear_zones)`` for a specific timeframe."""
        st = self.states.get(tf_label)
        if st is None:
            return [], []
        return list(st.bull_zones), list(st.bear_zones)

    def get_incursion_messages(self) -> List[str]:
        """Return incursion alert messages generated on the current bar."""
        return list(self.incursion_messages)

    def get_bar_events(self) -> List[dict]:
        """Return OB creation events generated on the current bar."""
        return list(self.bar_events)

    # ---- Convenience: check if price is inside any OB ------------------

    def price_in_bull_ob(self, price: float) -> Optional[OBZone]:
        """Return the first bull OB zone containing *price*, or None."""
        for zone in self.get_all_bull_zones():
            if zone.bottom <= price <= zone.top:
                return zone
        return None

    def price_in_bear_ob(self, price: float) -> Optional[OBZone]:
        """Return the first bear OB zone containing *price*, or None."""
        for zone in self.get_all_bear_zones():
            if zone.bottom <= price <= zone.top:
                return zone
        return None

    def price_in_any_ob(self, price: float) -> Optional[OBZone]:
        """Return the first OB zone (bull or bear) containing *price*."""
        return self.price_in_bull_ob(price) or self.price_in_bear_ob(price)

    # ---- Summary / debug -----------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of currently active OB zones."""
        lines = ["ICT MTF Order Block Wicks — Active Zones"]
        lines.append("=" * 45)
        for tf_label, st in self.states.items():
            if st.bull_zones or st.bear_zones:
                lines.append(f"\n  [{tf_label}]")
                for z in st.bull_zones:
                    lines.append(
                        f"    BULL  top={z.top:.5f}  btm={z.bottom:.5f}"
                        f"  bar={z.creation_bar}  col={z.bgcolor}"
                    )
                for z in st.bear_zones:
                    lines.append(
                        f"    BEAR  top={z.top:.5f}  btm={z.bottom:.5f}"
                        f"  bar={z.creation_bar}  col={z.bgcolor}"
                    )
        if all(not st.bull_zones and not st.bear_zones for st in self.states.values()):
            lines.append("  (no active zones)")
        return "\n".join(lines)
