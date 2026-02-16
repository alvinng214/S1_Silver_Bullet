"""Liquidity & inducements – cTrader Python custom indicator.

Full port of ``Liquidity & inducements.py`` (Pine Script translation) into
the cTrader cAlgo Python indicator framework.  All PriceAction library
logic is embedded inline because cTrader indicators cannot import local
Python modules.

OUTPUTS (DataSeries written each bar):
  LiqBuysideTarget  – price of the nearest buyside external liquidity pool
  LiqSellsideTarget – price of the nearest sellside external liquidity pool

PARAMETERS (set in cTrader UI – names must match the C# bridge):
  MarketLeft, MarketRight              – structure pivot lengths (default 5)
  GrabsEnabled … RetracementEnabled    – feature toggles (bool)
  GrabsLeft, GrabsRight, GrabsLookback – grab pivot params
  BigGrabsLeft, BigGrabsRight, …       – big grab pivot params
  SweepsLeft, SweepsRight, …           – sweep pivot params
  TurtleLeft, TurtleRight, …           – turtle soup pivot params
  TurtleConfirmation                   – turtle confirmation flag
  EqualLeft, EqualRight, EqualAtrFactor, EqualLookback
  RetrLeft, RetrRight, RetrLookback, RetrKeepInvalidated
  ExternalShow                         – how many external liq levels to show
"""

import clr

clr.AddReference("cAlgo.API")

from cAlgo.API import *


# ═══════════════════════════════════════════════════════════════════════════
# Constant
# ═══════════════════════════════════════════════════════════════════════════

NAN = float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# State classes (mirrors PriceAction.py lightweight containers)
# ═══════════════════════════════════════════════════════════════════════════

class Pivot:
    __slots__ = ("price", "bar_index", "type", "break_of_structure_broken",
                 "liquidity_broken", "change_of_character_broken")

    def __init__(self, price, bar_index, ptype):
        self.price = price
        self.bar_index = bar_index
        self.type = ptype                        # 1 = high, -1 = low
        self.break_of_structure_broken = False
        self.liquidity_broken = False
        self.change_of_character_broken = False


class StructureBreak:
    __slots__ = ("x1", "y1", "x2", "y2", "deleted")

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.deleted = False


class LiquidityGrab:
    __slots__ = ("pivot", "taken", "invalidated")

    def __init__(self, pivot):
        self.pivot = pivot
        self.taken = False
        self.invalidated = False


class EqualPivotInducement:
    __slots__ = ("stop_losses", "first_pivot", "second_pivot",
                 "liquidity_taken")

    def __init__(self, stop_losses, first_pivot, second_pivot):
        self.stop_losses = stop_losses
        self.first_pivot = first_pivot
        self.second_pivot = second_pivot
        self.liquidity_taken = False


class RetracementInducement:
    __slots__ = ("pivot", "taken", "invalidated", "stop_index")

    def __init__(self, pivot):
        self.pivot = pivot
        self.taken = False
        self.invalidated = False
        self.stop_index = None


class ExternalLiquidity:
    __slots__ = ("price", "pivot", "hidden")

    def __init__(self, price, pivot, hidden=False):
        self.price = price
        self.pivot = pivot
        self.hidden = hidden


class TurtleSoup:
    __slots__ = ("start", "end", "pivot", "deepest",
                 "line_color", "box_color")

    def __init__(self, start, end, pivot, deepest,
                 line_color=None, box_color=None):
        self.start = start
        self.end = end
        self.pivot = pivot
        self.deepest = deepest
        self.line_color = line_color
        self.box_color = box_color


# ═══════════════════════════════════════════════════════════════════════════
# Indicator class
# ═══════════════════════════════════════════════════════════════════════════

class LiquidityInducements():
    """Liquidity & inducements indicator for cTrader."""

    # ── lifecycle ─────────────────────────────────────────────────────────

    def initialize(self):
        """Set up ATR sub-indicator and all internal state."""

        # ── parameters (read from C# bridge api.XXX) ─────────────────────
        self.market_left = api.MarketLeft if hasattr(api, "MarketLeft") else 5
        self.market_right = api.MarketRight if hasattr(api, "MarketRight") else 5

        self.grabs_enabled = api.GrabsEnabled if hasattr(api, "GrabsEnabled") else True
        self.big_grabs_enabled = api.BigGrabsEnabled if hasattr(api, "BigGrabsEnabled") else True
        self.sweeps_enabled = api.SweepsEnabled if hasattr(api, "SweepsEnabled") else True
        self.turtle_soups_enabled = api.TurtleSoupsEnabled if hasattr(api, "TurtleSoupsEnabled") else True
        self.equal_pivots_enabled = api.EqualPivotsEnabled if hasattr(api, "EqualPivotsEnabled") else True
        self.external_liquidity_enabled = api.ExternalLiquidityEnabled if hasattr(api, "ExternalLiquidityEnabled") else True
        self.retracement_inducements_enabled = api.RetracementEnabled if hasattr(api, "RetracementEnabled") else True

        self.grabs_left = api.GrabsLeft if hasattr(api, "GrabsLeft") else 3
        self.grabs_right = api.GrabsRight if hasattr(api, "GrabsRight") else 3
        self.grabs_lookback = api.GrabsLookback if hasattr(api, "GrabsLookback") else 5

        self.big_grabs_left = api.BigGrabsLeft if hasattr(api, "BigGrabsLeft") else 10
        self.big_grabs_right = api.BigGrabsRight if hasattr(api, "BigGrabsRight") else 10
        self.big_grabs_lookback = api.BigGrabsLookback if hasattr(api, "BigGrabsLookback") else 5

        self.sweeps_left = api.SweepsLeft if hasattr(api, "SweepsLeft") else 3
        self.sweeps_right = api.SweepsRight if hasattr(api, "SweepsRight") else 3
        self.sweeps_lookback = api.SweepsLookback if hasattr(api, "SweepsLookback") else 5

        self.turtle_left = api.TurtleLeft if hasattr(api, "TurtleLeft") else 1
        self.turtle_right = api.TurtleRight if hasattr(api, "TurtleRight") else 1
        self.turtle_lookback = api.TurtleLookback if hasattr(api, "TurtleLookback") else 5
        self.turtle_confirmation = api.TurtleConfirmation if hasattr(api, "TurtleConfirmation") else True

        self.equal_left = api.EqualLeft if hasattr(api, "EqualLeft") else 1
        self.equal_right = api.EqualRight if hasattr(api, "EqualRight") else 1
        self.equal_atr_factor = api.EqualAtrFactor if hasattr(api, "EqualAtrFactor") else 0.5
        self.equal_lookback = api.EqualLookback if hasattr(api, "EqualLookback") else 3

        self.retr_left = api.RetrLeft if hasattr(api, "RetrLeft") else 1
        self.retr_right = api.RetrRight if hasattr(api, "RetrRight") else 1
        self.retr_lookback = api.RetrLookback if hasattr(api, "RetrLookback") else 5
        self.retr_keep_invalidated = api.RetrKeepInvalidated if hasattr(api, "RetrKeepInvalidated") else False

        self.external_show = api.ExternalShow if hasattr(api, "ExternalShow") else 1

        # ── ATR sub-indicator (14-period SMA of True Range) ──────────────
        self.atr_indicator = api.Indicators.AverageTrueRange(14, MovingAverageType.Simple)

        # ── market structure state ───────────────────────────────────────
        self.structure_trend = 0
        self.structure_pivots = []            # list[Pivot], newest first, max 6
        self.structure_bos_list = []          # list[StructureBreak]

        self.change_of_character = None       # Pivot | None
        self.break_of_structure = None        # Pivot | None
        self.previous_structure_break_pivot = None
        self.previous_structure_break_index = None
        self.retracement_structure_break_index = None

        # ── liquidity grab state ─────────────────────────────────────────
        self.grabs_highs = []
        self.grabs_lows = []
        self.big_grabs_highs = []
        self.big_grabs_lows = []

        # ── sweep state ──────────────────────────────────────────────────
        self.sweeps_highs = []
        self.sweeps_lows = []

        # ── turtle soup state ────────────────────────────────────────────
        self.turtle_pivot_highs = []          # list[Pivot]
        self.turtle_pivot_lows = []           # list[Pivot]
        self.turtle_bullish = []              # list[TurtleSoup]
        self.turtle_bearish = []              # list[TurtleSoup]

        # ── equal pivot state ────────────────────────────────────────────
        self.eq_highs = []                    # list[Pivot]
        self.eq_lows = []                     # list[Pivot]
        self.eq_bearish_inducements = []      # list[EqualPivotInducement]
        self.eq_bullish_inducements = []      # list[EqualPivotInducement]

        # ── external liquidity state ─────────────────────────────────────
        self.buyside = []                     # list[ExternalLiquidity]
        self.sellside = []                    # list[ExternalLiquidity]

        # ── retracement inducement state ─────────────────────────────────
        self.retr_highs = []                  # list[RetracementInducement]
        self.retr_lows = []                   # list[RetracementInducement]
        self.retr_high_pivots = []            # list[Pivot]
        self.retr_low_pivots = []             # list[Pivot]

    # ── main calculate ────────────────────────────────────────────────────

    def calculate(self, index):
        """Called once per bar.  Mirrors the body of the original
        ``calculate_liquidity_inducements()`` for-loop."""

        high = api.Bars.HighPrices[index]
        low = api.Bars.LowPrices[index]
        close = api.Bars.ClosePrices[index]

        # ── 1. structure pivot detection ─────────────────────────────────
        self._structure_pivot_step(index)

        last_high = None
        last_low = None
        for p in self.structure_pivots:
            if p.type == 1 and last_high is None:
                last_high = p
            if p.type == -1 and last_low is None:
                last_low = p
            if last_high is not None and last_low is not None:
                break

        # ── 2. change of character / break of structure ──────────────────
        self.change_of_character = self._change_of_character(index)
        structure_break_event = False
        if self.change_of_character:
            self.break_of_structure = None
            self.previous_structure_break_pivot = self.change_of_character
            structure_break_event = True

        bos_pivot = self._break_of_structure(index)
        if bos_pivot:
            self.break_of_structure = bos_pivot
            self.previous_structure_break_pivot = bos_pivot
            structure_break_event = True

        # ── 3. detect grabs/sweeps BEFORE storing new pivots ─────────────
        if index > 0:
            prev_high = api.Bars.HighPrices[index - 1]
            prev_low = api.Bars.LowPrices[index - 1]

            self._process_grabs(
                self.grabs_highs + self.grabs_lows,
                prev_high, prev_low, close, index,
            )
            self._process_grabs(
                self.big_grabs_highs + self.big_grabs_lows,
                prev_high, prev_low, close, index,
            )
            self._process_sweeps(
                prev_high, prev_low, close, index,
            )

        # ── 4. store new grab pivots ─────────────────────────────────────
        if self.grabs_enabled:
            gh = self._detect_pivot_high(index, self.grabs_left, self.grabs_right)
            if gh is not None:
                self.grabs_highs.insert(0, LiquidityGrab(gh))
                self.grabs_highs = self.grabs_highs[:self.grabs_lookback]
            gl = self._detect_pivot_low(index, self.grabs_left, self.grabs_right)
            if gl is not None:
                self.grabs_lows.insert(0, LiquidityGrab(gl))
                self.grabs_lows = self.grabs_lows[:self.grabs_lookback]

        if self.big_grabs_enabled:
            bh = self._detect_pivot_high(index, self.big_grabs_left, self.big_grabs_right)
            if bh is not None:
                self.big_grabs_highs.insert(0, LiquidityGrab(bh))
                self.big_grabs_highs = self.big_grabs_highs[:self.big_grabs_lookback]
            bl = self._detect_pivot_low(index, self.big_grabs_left, self.big_grabs_right)
            if bl is not None:
                self.big_grabs_lows.insert(0, LiquidityGrab(bl))
                self.big_grabs_lows = self.big_grabs_lows[:self.big_grabs_lookback]

        # ── 5. store new sweep pivots ────────────────────────────────────
        if self.sweeps_enabled:
            sh = self._detect_pivot_high(index, self.sweeps_left, self.sweeps_right)
            if sh is not None:
                self.sweeps_highs.insert(0, LiquidityGrab(sh))
                self.sweeps_highs = self.sweeps_highs[:self.sweeps_lookback]
            sl = self._detect_pivot_low(index, self.sweeps_left, self.sweeps_right)
            if sl is not None:
                self.sweeps_lows.insert(0, LiquidityGrab(sl))
                self.sweeps_lows = self.sweeps_lows[:self.sweeps_lookback]
            if self.change_of_character and self.previous_structure_break_index is not None:
                self.sweeps_highs.clear()
                self.sweeps_lows.clear()

        # ── 6. turtle soups ──────────────────────────────────────────────
        if self.turtle_soups_enabled:
            self._visualize_turtle_soups(self.turtle_pivot_highs, self.turtle_bearish, index)
            self._visualize_turtle_soups(self.turtle_pivot_lows, self.turtle_bullish, index)
            if (self.turtle_confirmation
                    and self.change_of_character
                    and self.previous_structure_break_index is not None):
                self._confirm_turtle(self.turtle_bullish, index)
                self._confirm_turtle(self.turtle_bearish, index)
            # store new turtle pivots
            th = self._detect_pivot_high(index, self.turtle_left, self.turtle_right)
            tl = self._detect_pivot_low(index, self.turtle_left, self.turtle_right)
            if th is not None:
                if len(self.turtle_pivot_highs) >= self.turtle_lookback:
                    self.turtle_pivot_highs.pop()
                self.turtle_pivot_highs.insert(0, th)
            if tl is not None:
                if len(self.turtle_pivot_lows) >= self.turtle_lookback:
                    self.turtle_pivot_lows.pop()
                self.turtle_pivot_lows.insert(0, tl)

        # ── 7. equal pivots ──────────────────────────────────────────────
        if self.equal_pivots_enabled:
            self._process_equal_pivots(index, high, low, close, structure_break_event)

        # ── 8. external liquidity ────────────────────────────────────────
        if self.external_liquidity_enabled:
            self._process_external_liquidity(index, high, low, last_high, last_low)

        # ── 9. retracement inducements ───────────────────────────────────
        if self.retracement_inducements_enabled:
            self._process_retracement_inducements(index, high, low, structure_break_event)

        # ── 10. write output DataSeries ──────────────────────────────────
        api.LiqBuysideTarget[index] = self.buyside[0].price if self.buyside else NAN
        api.LiqSellsideTarget[index] = self.sellside[0].price if self.sellside else NAN

        if structure_break_event:
            self.previous_structure_break_index = index

    # ═════════════════════════════════════════════════════════════════════
    # Pivot detection helpers
    # ═════════════════════════════════════════════════════════════════════

    def _detect_pivot_high(self, index, left, right):
        """Return Pivot if bar at index-right is a confirmed pivot high,
        else None.  Mirrors ``_pivot_series`` for same-TF."""
        pivot_idx = index - right
        if pivot_idx - left < 0:
            return None
        center = api.Bars.HighPrices[pivot_idx]
        for j in range(pivot_idx - left, pivot_idx):
            if api.Bars.HighPrices[j] >= center:
                return None
        for j in range(pivot_idx + 1, pivot_idx + right + 1):
            if api.Bars.HighPrices[j] >= center:
                return None
        return Pivot(center, pivot_idx, 1)

    def _detect_pivot_low(self, index, left, right):
        """Return Pivot if bar at index-right is a confirmed pivot low,
        else None."""
        pivot_idx = index - right
        if pivot_idx - left < 0:
            return None
        center = api.Bars.LowPrices[pivot_idx]
        for j in range(pivot_idx - left, pivot_idx):
            if api.Bars.LowPrices[j] <= center:
                return None
        for j in range(pivot_idx + 1, pivot_idx + right + 1):
            if api.Bars.LowPrices[j] <= center:
                return None
        return Pivot(center, pivot_idx, -1)

    # ═════════════════════════════════════════════════════════════════════
    # Market structure (PriceAction.pivot_step / change_of_character / BOS)
    # ═════════════════════════════════════════════════════════════════════

    def _structure_pivot_step(self, index):
        """Detect structure pivots (mirrors PriceAction.pivot_step)."""
        left = self.market_left
        right = self.market_right
        pivot_idx = index - right
        if pivot_idx - left < 0:
            return
        center_high = api.Bars.HighPrices[pivot_idx]
        center_low = api.Bars.LowPrices[pivot_idx]

        is_ph = True
        for j in range(pivot_idx - left, pivot_idx):
            if api.Bars.HighPrices[j] >= center_high:
                is_ph = False
                break
        if is_ph:
            for j in range(pivot_idx + 1, pivot_idx + right + 1):
                if api.Bars.HighPrices[j] >= center_high:
                    is_ph = False
                    break
        if is_ph:
            if len(self.structure_pivots) > 5:
                self.structure_pivots.pop()
            self.structure_pivots.insert(0, Pivot(center_high, pivot_idx, 1))

        is_pl = True
        for j in range(pivot_idx - left, pivot_idx):
            if api.Bars.LowPrices[j] <= center_low:
                is_pl = False
                break
        if is_pl:
            for j in range(pivot_idx + 1, pivot_idx + right + 1):
                if api.Bars.LowPrices[j] <= center_low:
                    is_pl = False
                    break
        if is_pl:
            if len(self.structure_pivots) > 5:
                self.structure_pivots.pop()
            self.structure_pivots.insert(0, Pivot(center_low, pivot_idx, -1))

    def _change_of_character(self, index):
        """Detect CHoCH (mirrors PriceAction.change_of_character).
        Returns the broken Pivot or None."""
        close_now = api.Bars.ClosePrices[index]
        close_prev = api.Bars.ClosePrices[index - 1] if index > 0 else close_now

        for pivot in self.structure_pivots:
            # bullish CHoCH
            if (self.structure_trend <= 0
                    and pivot.type == 1
                    and close_now > pivot.price
                    and close_prev < pivot.price
                    and not pivot.change_of_character_broken):
                pivot.change_of_character_broken = True
                self.structure_trend = 1
                self.structure_bos_list.clear()
                remaining = []
                for p in self.structure_pivots:
                    if p.bar_index <= pivot.bar_index:
                        continue
                    p.break_of_structure_broken = True
                    remaining.append(p)
                for p in remaining:
                    if p.bar_index != pivot.bar_index:
                        p.change_of_character_broken = False
                self.structure_pivots = remaining
                return pivot

            # bearish CHoCH
            if (self.structure_trend >= 0
                    and pivot.type == -1
                    and close_now < pivot.price
                    and close_prev > pivot.price
                    and not pivot.change_of_character_broken):
                pivot.change_of_character_broken = True
                self.structure_trend = -1
                self.structure_bos_list.clear()
                remaining = []
                for p in self.structure_pivots:
                    if p.bar_index <= pivot.bar_index:
                        continue
                    p.break_of_structure_broken = True
                    remaining.append(p)
                for p in remaining:
                    if p.bar_index != pivot.bar_index:
                        p.change_of_character_broken = False
                self.structure_pivots = remaining
                return pivot

        return None

    def _break_of_structure(self, index):
        """Detect BOS (mirrors PriceAction.break_of_structure).
        Returns the broken Pivot or None."""
        close_now = api.Bars.ClosePrices[index]

        for pivot in self.structure_pivots:
            # bullish BOS
            if (self.structure_trend == 1
                    and pivot.type == 1
                    and close_now > pivot.price
                    and not pivot.break_of_structure_broken):
                create = True
                to_remove = []
                for bos in self.structure_bos_list:
                    if bos.x1 > pivot.bar_index:
                        if bos.y1 < pivot.price:
                            to_remove.append(bos)
                            continue
                        create = False
                        break
                for bos in to_remove:
                    self.structure_bos_list.remove(bos)
                if create:
                    self.structure_bos_list.insert(
                        0, StructureBreak(pivot.bar_index, pivot.price, index, pivot.price))
                    pivot.break_of_structure_broken = True
                    return pivot

            # bearish BOS
            if (self.structure_trend == -1
                    and pivot.type == -1
                    and close_now < pivot.price
                    and not pivot.break_of_structure_broken):
                create = True
                to_remove = []
                for bos in self.structure_bos_list:
                    if bos.x1 > pivot.bar_index:
                        if bos.y1 > pivot.price:
                            to_remove.append(bos)
                            continue
                        create = False
                        break
                for bos in to_remove:
                    self.structure_bos_list.remove(bos)
                if create:
                    self.structure_bos_list.insert(
                        0, StructureBreak(pivot.bar_index, pivot.price, index, pivot.price))
                    pivot.break_of_structure_broken = True
                    return pivot

        return None

    # ═════════════════════════════════════════════════════════════════════
    # Liquidity grabs
    # ═════════════════════════════════════════════════════════════════════

    def _process_grabs(self, grabs, prev_high, prev_low, close, index):
        """Process grabs/big-grabs detection on existing pivots.
        Mirrors the inner ``_process_grabs`` closure in the original."""
        for grab in grabs:
            if grab.taken or grab.invalidated:
                continue
            grabbed = False
            if grab.pivot.type == -1:
                if prev_low <= grab.pivot.price and close >= grab.pivot.price:
                    grabbed = True
                elif close < grab.pivot.price:
                    grab.invalidated = True
            else:
                if prev_high >= grab.pivot.price and close <= grab.pivot.price:
                    grabbed = True
                elif close > grab.pivot.price:
                    grab.invalidated = True
            if grabbed:
                grab.taken = True

    # ═════════════════════════════════════════════════════════════════════
    # Sweeps
    # ═════════════════════════════════════════════════════════════════════

    def _process_sweeps(self, prev_high, prev_low, close, index):
        """Process sweep detection on existing sweep pivots.
        Mirrors the sweep detection loop in the original."""
        for sweep in self.sweeps_highs + self.sweeps_lows:
            if sweep.taken or sweep.invalidated:
                continue
            swept = False
            if sweep.pivot.type == -1:
                if prev_low <= sweep.pivot.price and close <= sweep.pivot.price:
                    if (self.previous_structure_break_pivot
                            and sweep.pivot.bar_index == self.previous_structure_break_pivot.bar_index):
                        sweep.invalidated = True
                    else:
                        swept = True
                elif prev_low <= sweep.pivot.price and close >= sweep.pivot.price:
                    sweep.invalidated = True
            else:
                if prev_high >= sweep.pivot.price and close >= sweep.pivot.price:
                    if (self.previous_structure_break_pivot
                            and sweep.pivot.bar_index == self.previous_structure_break_pivot.bar_index):
                        sweep.invalidated = True
                    else:
                        swept = True
                elif prev_high >= sweep.pivot.price and close <= sweep.pivot.price:
                    sweep.invalidated = True
            if swept:
                sweep.taken = True
            elif not sweep.invalidated:
                # check if grabbed (invalidating)
                if sweep.pivot.type == -1:
                    if prev_low <= sweep.pivot.price and close >= sweep.pivot.price:
                        sweep.invalidated = True
                else:
                    if prev_high >= sweep.pivot.price and close <= sweep.pivot.price:
                        sweep.invalidated = True

    # ═════════════════════════════════════════════════════════════════════
    # Turtle soups
    # ═════════════════════════════════════════════════════════════════════

    def _visualize_turtle_soups(self, pivots, turtle_soups, index):
        """Detect turtle soup patterns on tracked pivots.
        Mirrors PriceAction.visualize_turtle_soups."""
        if index < 1:
            return
        for pivot in reversed(pivots):
            if pivot.liquidity_broken:
                continue
            if pivot.type == -1:
                confirmed = (api.Bars.LowPrices[index] > pivot.price
                             and api.Bars.LowPrices[index - 1] <= pivot.price)
            else:
                confirmed = (api.Bars.HighPrices[index] < pivot.price
                             and api.Bars.HighPrices[index - 1] >= pivot.price)
            if not confirmed:
                continue

            i = 2
            if pivot.type == -1:
                deepest = api.Bars.LowPrices[index - 1]
            else:
                deepest = api.Bars.HighPrices[index - 1]

            while True:
                check_idx = index - i
                if check_idx < 0:
                    break
                if pivot.type == -1:
                    price = api.Bars.LowPrices[check_idx]
                    swept = price <= pivot.price
                else:
                    price = api.Bars.HighPrices[check_idx]
                    swept = price >= pivot.price
                if swept:
                    i += 1
                    if pivot.type == -1 and price < deepest:
                        deepest = price
                    if pivot.type == 1 and price > deepest:
                        deepest = price
                else:
                    break

            if i == 2:
                continue

            pivot.liquidity_broken = True
            start = index - i
            end = index - 1
            box_color = None if self.turtle_confirmation else "orange"
            line_color = None if self.turtle_confirmation else "orange"

            ts = TurtleSoup(start, end, pivot, deepest, line_color, box_color)

            # remove subsumed turtle soups
            self._remove_subsumed_turtle(turtle_soups, ts)

            if len(turtle_soups) >= 5:
                turtle_soups.pop()
            turtle_soups.insert(0, ts)

    def _remove_subsumed_turtle(self, turtle_soups, new_ts):
        """Remove turtle soups that are fully contained within the new one."""
        to_remove = [
            j for j, prev in enumerate(turtle_soups)
            if prev.start >= new_ts.start and prev.end <= new_ts.end
        ]
        for j in reversed(to_remove):
            turtle_soups.pop(j)

    def _confirm_turtle(self, turtle_soups, index):
        """Confirm unconfirmed turtle soups after CHoCH.
        Mirrors PriceAction.confirm."""
        if self.previous_structure_break_index is None:
            return
        for ts in turtle_soups:
            if ts.end > self.previous_structure_break_index:
                ts.line_color = "orange"
                ts.box_color = "orange"

    # ═════════════════════════════════════════════════════════════════════
    # Equal pivots
    # ═════════════════════════════════════════════════════════════════════

    def _process_equal_pivots(self, index, high, low, close, structure_break_event):
        """Detect equal pivot inducements.
        Mirrors the equal_pivots_enabled block in the original."""
        atr_val = self.atr_indicator.Result[index]
        if atr_val != atr_val:  # NaN check
            atr_val = 0.0

        eq_high = self._detect_pivot_high(index, self.equal_left, self.equal_right)
        eq_low = self._detect_pivot_low(index, self.equal_left, self.equal_right)

        if eq_high is not None:
            self.eq_highs.insert(0, eq_high)
            self.eq_highs = self.eq_highs[:self.equal_lookback]
        if eq_low is not None:
            self.eq_lows.insert(0, eq_low)
            self.eq_lows = self.eq_lows[:self.equal_lookback]

        # check for equal pivot pairs
        for pivots, inducements, direction in (
            (self.eq_highs, self.eq_bearish_inducements, -1),
            (self.eq_lows, self.eq_bullish_inducements, 1),
        ):
            if len(pivots) < 2:
                continue
            latest = pivots[0]
            # Only process newly confirmed pivots.  The original checks
            # latest.bar_index == i - 1; for same-TF with equal_right=1
            # the pivot center is at index-1, confirmed at index.
            if latest.bar_index != index - self.equal_right:
                continue

            for equal_pivot in pivots[1:]:
                if latest.type == -1:
                    max_price = equal_pivot.price + (atr_val * self.equal_atr_factor)
                    min_price = equal_pivot.price
                else:
                    max_price = equal_pivot.price
                    min_price = equal_pivot.price - (atr_val * self.equal_atr_factor)

                if latest.price > max_price or latest.price < min_price:
                    continue

                # check if the line between pivots was broken
                broken = False
                bar_span = latest.bar_index - equal_pivot.bar_index
                if bar_span == 0:
                    continue
                if latest.type == 1:
                    step = (equal_pivot.price - latest.price) / max(1, bar_span)
                else:
                    step = (latest.price - equal_pivot.price) / bar_span

                for j in range(2, bar_span + 1):
                    bar_price = latest.price + (step * (j - 1))
                    check_idx = index - j
                    if check_idx < 0:
                        broken = True
                        break
                    if latest.type == 1 and api.Bars.HighPrices[check_idx] > bar_price:
                        broken = True
                        break
                    if latest.type == -1 and api.Bars.LowPrices[check_idx] < bar_price:
                        broken = True
                        break
                if broken:
                    continue

                trend_inducement = ((latest.type == 1 and self.structure_trend == -1)
                                    or (latest.type == -1 and self.structure_trend == 1))
                if trend_inducement:
                    if latest.type == 1:
                        stop_price = equal_pivot.price + (atr_val * 0.1)
                    else:
                        stop_price = equal_pivot.price - (atr_val * 0.1)
                    inducements.insert(0, EqualPivotInducement(stop_price, equal_pivot, latest))

        # check if equal inducements are taken
        for inducement in self.eq_bearish_inducements:
            if (self.structure_trend == -1
                    and not inducement.liquidity_taken
                    and high >= inducement.stop_losses):
                inducement.liquidity_taken = True

        for inducement in self.eq_bullish_inducements:
            if (self.structure_trend == 1
                    and not inducement.liquidity_taken
                    and low <= inducement.stop_losses):
                inducement.liquidity_taken = True

        if structure_break_event:
            self.eq_bullish_inducements.clear()
            self.eq_bearish_inducements.clear()

    # ═════════════════════════════════════════════════════════════════════
    # External liquidity
    # ═════════════════════════════════════════════════════════════════════

    def _process_external_liquidity(self, index, high, low, last_high, last_low):
        """Manage buyside/sellside external liquidity pools.
        Mirrors the external_liquidity_enabled block in the original."""

        # new buyside level when a structure pivot high is confirmed
        if last_high and last_high.bar_index == index - self.market_right:
            for pool in self.buyside:
                if not pool.hidden:
                    pool.hidden = True
            self.buyside.insert(0, ExternalLiquidity(last_high.price, last_high, hidden=True))

        # new sellside level when a structure pivot low is confirmed
        if last_low and last_low.bar_index == index - self.market_right:
            for pool in self.sellside:
                if not pool.hidden:
                    pool.hidden = True
            self.sellside.insert(0, ExternalLiquidity(last_low.price, last_low, hidden=True))

        # remove taken sellside pools (price swept below)
        self.sellside = [pool for pool in self.sellside if low > pool.price]

        # remove taken buyside pools (price swept above)
        self.buyside = [pool for pool in self.buyside if high < pool.price]

        # unhide the top N pools
        for i_pool, pool in enumerate(self.buyside):
            if i_pool + 1 <= self.external_show:
                pool.hidden = False
        for i_pool, pool in enumerate(self.sellside):
            if i_pool + 1 <= self.external_show:
                pool.hidden = False

    # ═════════════════════════════════════════════════════════════════════
    # Retracement inducements
    # ═════════════════════════════════════════════════════════════════════

    def _process_retracement_inducements(self, index, high, low, structure_break_event):
        """Detect and manage retracement inducements.
        Mirrors the retracement_inducements_enabled block."""

        rh = self._detect_pivot_high(index, self.retr_left, self.retr_right)
        rl = self._detect_pivot_low(index, self.retr_left, self.retr_right)

        if rh is not None:
            self.retr_high_pivots.insert(0, rh)
            self.retr_high_pivots = self.retr_high_pivots[:self.retr_lookback]
        if rl is not None:
            self.retr_low_pivots.insert(0, rl)
            self.retr_low_pivots = self.retr_low_pivots[:self.retr_lookback]

        if self.structure_trend != 0:
            pivots = self.retr_high_pivots if self.structure_trend == -1 else self.retr_low_pivots
            if len(pivots) > 1:
                latest = pivots[0]
                next_latest = pivots[1]
                if self.retracement_structure_break_index is not None:
                    latest_after_break = latest.bar_index > self.retracement_structure_break_index
                    if (latest.bar_index == index - self.retr_right
                            and latest_after_break
                            and next_latest.bar_index < self.retracement_structure_break_index):
                        target_list = self.retr_highs if self.structure_trend == -1 else self.retr_lows
                        target_list.insert(0, RetracementInducement(latest))

        # stop inducements that are taken
        self._stop_retracement_inducements(high, low, index, "take")

        # invalidate inducements on structure break
        if structure_break_event:
            self._stop_retracement_inducements(high, low, index, "invalidate")

        if structure_break_event:
            self.retracement_structure_break_index = index

    def _stop_retracement_inducements(self, high, low, bar_index, stop_reason):
        """Stop (take or invalidate) retracement inducements.
        Mirrors ``_stop_retracement_inducements`` in the original."""

        remaining_highs = []
        for ind in self.retr_highs:
            stop = stop_reason == "invalidate" or high >= ind.pivot.price
            if stop:
                ind.stop_index = bar_index
                if stop_reason == "take":
                    ind.taken = True
                else:
                    ind.invalidated = True
            else:
                remaining_highs.append(ind)
        self.retr_highs = remaining_highs

        remaining_lows = []
        for ind in self.retr_lows:
            stop = stop_reason == "invalidate" or low <= ind.pivot.price
            if stop:
                ind.stop_index = bar_index
                if stop_reason == "take":
                    ind.taken = True
                else:
                    ind.invalidated = True
            else:
                remaining_lows.append(ind)
        self.retr_lows = remaining_lows
