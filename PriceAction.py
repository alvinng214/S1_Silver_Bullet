"""PriceAction library translated from Pine Script.

This module mirrors the logic of the PriceAction Pine library used by
Liquidity & inducements. Visual objects (line/box/label) are modeled
as lightweight state containers to preserve the logic side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class Line:
    x1: Union[int, pd.Timestamp]
    y1: float
    x2: Union[int, pd.Timestamp]
    y2: float
    color: Optional[str] = None
    style: Optional[str] = None
    extend_right: bool = False
    deleted: bool = False

    def set_x2(self, x2: Union[int, pd.Timestamp]) -> None:
        self.x2 = x2

    def set_color(self, color: Optional[str]) -> None:
        self.color = color

    def set_extend_right(self, extend_right: bool) -> None:
        self.extend_right = extend_right

    def delete(self) -> None:
        self.deleted = True


@dataclass
class Label:
    x: int
    y: float
    text: str
    color: Optional[str] = None
    textcolor: Optional[str] = None
    style: Optional[str] = None
    size: Optional[int] = None
    deleted: bool = False

    def set_textcolor(self, color: Optional[str]) -> None:
        self.textcolor = color

    def delete(self) -> None:
        self.deleted = True


@dataclass
class Box:
    left: int
    top: float
    right: int
    bottom: float
    bgcolor: Optional[str] = None
    text: str = ""
    text_color: Optional[str] = None
    extend_right: bool = False
    border_width: int = 0
    border_color: Optional[str] = None
    deleted: bool = False

    def get_left(self) -> int:
        return self.left

    def get_top(self) -> float:
        return self.top

    def get_bottom(self) -> float:
        return self.bottom

    def set_right(self, right: int) -> None:
        self.right = right

    def set_top(self, top: float) -> None:
        self.top = top

    def set_rightbottom(self, right: int, bottom: float) -> None:
        self.right = right
        self.bottom = bottom

    def set_text(self, text: str) -> None:
        self.text = text

    def set_bgcolor(self, color: Optional[str]) -> None:
        self.bgcolor = color

    def set_text_color(self, color: Optional[str]) -> None:
        self.text_color = color

    def set_extend_right(self, extend_right: bool) -> None:
        self.extend_right = extend_right

    def delete(self) -> None:
        self.deleted = True


@dataclass
class LineFill:
    line1: Line
    line2: Line
    color: Optional[str] = None


@dataclass
class StructureBreak:
    line: Line
    label: Label


@dataclass
class Pivot:
    price: float
    bar_index: int
    type: int  # -1 = low, 1 = high
    time: Optional[pd.Timestamp] = None
    break_of_structure_broken: bool = False
    liquidity_broken: bool = False
    change_of_character_broken: bool = False


class StructureType(Enum):
    INTERNAL = "internal"
    SWING = "swing"


@dataclass
class Structure:
    left_length: int
    right_length: int
    type: StructureType
    trend: int = 0
    equal_pivots_factor: float = 0.0
    extend_equal_pivots_zones: bool = False
    extend_equal_pivots_style: str = ""
    extend_equal_pivots_color: str = ""
    equal_highs: List[Box] = field(default_factory=list)
    equal_lows: List[Box] = field(default_factory=list)
    break_of_structures: List[StructureBreak] = field(default_factory=list)
    pivots: List[Pivot] = field(default_factory=list)
    font_size: int = 7
    alert_change_of_character: bool = False
    alert_break_of_structure: bool = False
    alert_equal_pivots: bool = False


@dataclass
class Liquidity:
    liquidity_pivots_high: List[Pivot]
    liquidity_pivots_low: List[Pivot]
    liquidity_confirmation_bars: int
    liquidity_pivots_lookback: int
    font_size: int


@dataclass
class PriceAction:
    liquidity: Liquidity
    swing: Structure
    internal: Structure


@dataclass
class TurtleSoupSettings:
    pivot_left_length: int
    pivot_right_length: int
    lookback: int
    confirmation: bool
    color: str
    screener_keep: int = 0
    alert_frequency: str = "alert.freq_once_per_bar_close"


@dataclass
class TurtleSoup:
    line: Line
    box: Box
    start: int
    end: int
    pivot: Pivot


@dataclass
class Screener:
    turtle_soup_until_bar_index: Optional[int] = None


@dataclass
class TurtleSoups:
    highs: List[Pivot] = field(default_factory=list)
    lows: List[Pivot] = field(default_factory=list)
    bullish: List[TurtleSoup] = field(default_factory=list)
    bearish: List[TurtleSoup] = field(default_factory=list)
    alert_messages: List[str] = field(default_factory=list)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def set_bar_index(pivot: Pivot, bar_time: pd.Timestamp, times: Sequence[pd.Timestamp], bar_index: int) -> None:
    i = 0
    while True:
        time_back = times[bar_index - i] if bar_index - i >= 0 else None
        if time_back is None or time_back < bar_time:
            pivot.bar_index = bar_index - (i - 1)
            break
        i += 1


def set_bar_indices(pivot_high: Optional[Pivot], pivot_low: Optional[Pivot], times: Sequence[pd.Timestamp], bar_index: int) -> None:
    if pivot_high is not None and pivot_high.bar_index is None and pivot_high.time is not None:
        set_bar_index(pivot_high, pivot_high.time, times, bar_index)
    if pivot_low is not None and pivot_low.bar_index is None and pivot_low.time is not None:
        set_bar_index(pivot_low, pivot_low.time, times, bar_index)


def turtle_soup_alert(message: str, turtle_soup: TurtleSoup, settings: TurtleSoupSettings, bar_index: int) -> str:
    direction = "bullish" if turtle_soup.pivot.type == -1 else "bearish"
    return (
        f"{message} ({direction} of {turtle_soup.end - turtle_soup.start} bars, "
        f"{bar_index - turtle_soup.end} bars ago and with a pivot from "
        f"{turtle_soup.start - turtle_soup.pivot.bar_index} bars ago "
        f"(from the turtle soup start))"
    )


def alert_messages(turtle_soups_context: TurtleSoups) -> List[str]:
    messages = turtle_soups_context.alert_messages[:]
    turtle_soups_context.alert_messages.clear()
    return messages


def alert(message: str) -> str:
    return message


def alert_with_frequency(message: str, frequency: str) -> str:
    return message


def alert_turtle_soups(turtle_soups_context: TurtleSoups, settings: TurtleSoupSettings) -> Optional[str]:
    if not turtle_soups_context.alert_messages:
        return None
    message = "\n".join(turtle_soups_context.alert_messages)
    turtle_soups_context.alert_messages.clear()
    return alert_with_frequency(message, settings.alert_frequency)


def visualize_turtle_soups(
    pivots: List[Pivot],
    turtle_soups: List[TurtleSoup],
    turtle_soups_context: TurtleSoups,
    settings: TurtleSoupSettings,
    *,
    highs: pd.Series,
    lows: pd.Series,
    times: Optional[Sequence[pd.Timestamp]] = None,
    bar_index: int,
) -> None:
    reversed_pivots = list(pivots)
    reversed_pivots.reverse()
    for pivot in reversed_pivots:
        if pivot.liquidity_broken:
            continue
        confirmed = lows.iloc[bar_index] > pivot.price and lows.iloc[bar_index - 1] <= pivot.price
        if pivot.type == 1:
            confirmed = highs.iloc[bar_index] < pivot.price and highs.iloc[bar_index - 1] >= pivot.price
        if not confirmed:
            continue
        i = 2
        deepest = lows.iloc[bar_index - 1] if pivot.type == -1 else highs.iloc[bar_index - 1]
        while True:
            price = lows.iloc[bar_index - i] if pivot.type == -1 else highs.iloc[bar_index - i]
            swept = price <= pivot.price if pivot.type == -1 else price >= pivot.price
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
        start = bar_index - i
        end = bar_index - 1
        box_color = None if settings.confirmation else settings.color
        line_color = None if settings.confirmation else settings.color
        line_start = pivot.time if pivot.time is not None else pivot.bar_index
        line_end = times[bar_index - 1] if times is not None else bar_index - 1
        turtle_soup = TurtleSoup(
            Line(line_start, pivot.price, line_end, pivot.price, color=line_color, style="dotted"),
            Box(
                bar_index - i,
                pivot.price,
                bar_index - 1,
                deepest,
                bgcolor=box_color,
                text="🐢",
                text_color=box_color,
                border_width=0,
            ),
            start,
            end,
            pivot,
        )
        remove_indices = [
            j
            for j, previous in enumerate(turtle_soups)
            if previous.start >= turtle_soup.start and previous.end <= turtle_soup.end
        ]
        for remove_index in remove_indices:
            remove_turtle = turtle_soups[remove_index]
            remove_turtle.box.set_text("")
            if turtle_soup.start > remove_turtle.start:
                remove_turtle.line.set_x2(turtle_soup.start)
        if len(turtle_soups) >= 5:
            turtle_soups.pop()
        turtle_soups.insert(0, turtle_soup)
        if not settings.confirmation:
            turtle_soups_context.alert_messages.insert(
                0, turtle_soup_alert("Turtle soup (NOT confirmed)", turtle_soup, settings, bar_index)
            )


def get_pivots(
    settings: TurtleSoupSettings,
    *,
    highs: pd.Series,
    lows: pd.Series,
    times: Sequence[pd.Timestamp],
    bar_index: int,
) -> Tuple[Optional[Pivot], Optional[Pivot]]:
    pivot_high = pivot_low = None
    left = settings.pivot_left_length
    right = settings.pivot_right_length
    pivot_idx = bar_index - right
    if pivot_idx - left >= 0 and pivot_idx + right < len(highs):
        center_high = highs.iloc[pivot_idx]
        center_low = lows.iloc[pivot_idx]
        if (highs.iloc[pivot_idx - left : pivot_idx] < center_high).all() and (
            highs.iloc[pivot_idx + 1 : pivot_idx + right + 1] < center_high
        ).all():
            pivot_high = Pivot(center_high, pivot_idx, 1, time=times[pivot_idx])
        if (lows.iloc[pivot_idx - left : pivot_idx] > center_low).all() and (
            lows.iloc[pivot_idx + 1 : pivot_idx + right + 1] > center_low
        ).all():
            pivot_low = Pivot(center_low, pivot_idx, -1, time=times[pivot_idx])
    return pivot_high, pivot_low


def set_pivots(turtle_soups_context: TurtleSoups, settings: TurtleSoupSettings, pivot_high: Optional[Pivot], pivot_low: Optional[Pivot]) -> None:
    if pivot_high is not None:
        if len(turtle_soups_context.highs) >= settings.lookback:
            turtle_soups_context.highs.pop()
        turtle_soups_context.highs.insert(0, pivot_high)
    if pivot_low is not None:
        if len(turtle_soups_context.lows) >= settings.lookback:
            turtle_soups_context.lows.pop()
        turtle_soups_context.lows.insert(0, pivot_low)


def confirm(
    turtle_soups: List[TurtleSoup],
    turtle_soups_context: TurtleSoups,
    settings: TurtleSoupSettings,
    previous_structure_break_bar_index: int,
    screener: Screener,
    bar_index: int,
) -> None:
    for turtle_soup in turtle_soups:
        if turtle_soup.end > previous_structure_break_bar_index:
            turtle_soup.line.set_color(settings.color)
            turtle_soup.box.set_bgcolor(settings.color)
            turtle_soup.box.set_text_color(settings.color)
            turtle_soups_context.alert_messages.insert(
                0, turtle_soup_alert("Turtle soup (confirmed)", turtle_soup, settings, bar_index)
            )
            screener.turtle_soup_until_bar_index = bar_index + settings.screener_keep


def liquidation(liquidity: Liquidity, pivot: Pivot, closes: pd.Series, highs: pd.Series, lows: pd.Series, bar_index: int) -> bool:
    bars = liquidity.liquidity_confirmation_bars
    idx = bar_index - bars
    if idx < 0:
        return False
    if pivot.type == -1:
        if lows.iloc[idx] <= pivot.price and closes.iloc[idx] >= pivot.price:
            if bars > 0:
                for i in range(bars - 1, -1, -1):
                    if closes.iloc[bar_index - i] < pivot.price:
                        return False
                return True
            return True
    if pivot.type == 1:
        if highs.iloc[idx] >= pivot.price and closes.iloc[idx] <= pivot.price:
            if bars > 0:
                for i in range(bars - 1, -1, -1):
                    if closes.iloc[bar_index - i] > pivot.price:
                        return False
                return True
            return True
    return False


def visualize_liquidations(
    pivots: List[Pivot],
    liquidity: Liquidity,
    *,
    closes: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    bar_index: int,
) -> None:
    for pivot in pivots:
        if bar_index < pivot.bar_index + liquidity.liquidity_confirmation_bars + 1:
            continue
        if pivot.liquidity_broken:
            continue
        if liquidation(liquidity, pivot, closes, highs, lows, bar_index):
            limit_line = Line(
                pivot.bar_index,
                pivot.price,
                bar_index - liquidity.liquidity_confirmation_bars,
                pivot.price,
                color="orange",
                style="dotted",
            )
            if pivot.type == -1:
                break_price = lows.iloc[bar_index - liquidity.liquidity_confirmation_bars]
            else:
                break_price = highs.iloc[bar_index - liquidity.liquidity_confirmation_bars]
            break_line = Line(
                pivot.bar_index,
                break_price,
                bar_index - liquidity.liquidity_confirmation_bars,
                break_price,
            )
            _ = LineFill(limit_line, break_line, color="orange@80")
            label_x = int(
                bar_index
                - liquidity.liquidity_confirmation_bars
                - ((bar_index - liquidity.liquidity_confirmation_bars - pivot.bar_index) / 2)
            )
            label_style = "label_up" if pivot.type == -1 else None
            Label(
                label_x,
                pivot.price,
                "$$$",
                textcolor="orange@30",
                style=label_style,
                size=liquidity.font_size,
            )
            pivot.liquidity_broken = True
        else:
            if pivot.type == -1 and closes.iloc[bar_index] < pivot.price:
                pivot.liquidity_broken = True
            if pivot.type == 1 and closes.iloc[bar_index] > pivot.price:
                pivot.liquidity_broken = True


def liquidity_step(
    liquidity: Liquidity,
    *,
    closes: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    bar_index: int,
    bar_confirmed: bool = True,
) -> None:
    if bar_confirmed:
        visualize_liquidations(
            liquidity.liquidity_pivots_high,
            liquidity,
            closes=closes,
            highs=highs,
            lows=lows,
            bar_index=bar_index,
        )
        visualize_liquidations(
            liquidity.liquidity_pivots_low,
            liquidity,
            closes=closes,
            highs=highs,
            lows=lows,
            bar_index=bar_index,
        )
    if bar_index < 2:
        return
    pivot_idx = bar_index - 1
    pivot_high = highs.iloc[pivot_idx]
    pivot_low = lows.iloc[pivot_idx]
    if pivot_idx >= 1 and pivot_idx + 1 < len(highs):
        if highs.iloc[pivot_idx] > highs.iloc[pivot_idx - 1] and highs.iloc[pivot_idx] > highs.iloc[pivot_idx + 1]:
            if len(liquidity.liquidity_pivots_high) >= liquidity.liquidity_pivots_lookback:
                liquidity.liquidity_pivots_high.pop()
            liquidity.liquidity_pivots_high.insert(0, Pivot(pivot_high, pivot_idx, 1))
    if pivot_idx >= 1 and pivot_idx + 1 < len(lows):
        if lows.iloc[pivot_idx] < lows.iloc[pivot_idx - 1] and lows.iloc[pivot_idx] < lows.iloc[pivot_idx + 1]:
            if len(liquidity.liquidity_pivots_low) >= liquidity.liquidity_pivots_lookback:
                liquidity.liquidity_pivots_low.pop()
            liquidity.liquidity_pivots_low.insert(0, Pivot(pivot_low, pivot_idx, -1))


def broken_by_bar_pivot(pivot: Pivot, highs: pd.Series, lows: pd.Series, bar_index: int) -> bool:
    for i in range(1, bar_index - pivot.bar_index):
        if pivot.type == 1 and lows.iloc[bar_index - i] > pivot.price:
            return True
        if pivot.type == -1 and highs.iloc[bar_index - i] < pivot.price:
            return True
    return False


def broken_by_bar_box(zone: Box, zone_type: int, highs: pd.Series, lows: pd.Series, bar_index: int) -> bool:
    for i in range(1, bar_index - zone.get_left()):
        if zone_type == 1 and lows.iloc[bar_index - i] > zone.get_top():
            return True
        if zone_type == -1 and highs.iloc[bar_index - i] < zone.get_bottom():
            return True
    return False


def in_limits(high_limit: float, low_limit: float, price: float) -> bool:
    return low_limit <= price <= high_limit


def pivot_step(structure: Structure, highs: pd.Series, lows: pd.Series, times: Sequence[pd.Timestamp], bar_index: int) -> None:
    left = structure.left_length
    right = structure.right_length
    pivot_idx = bar_index - right
    if pivot_idx - left < 0 or pivot_idx + right >= len(highs):
        return
    center_high = highs.iloc[pivot_idx]
    center_low = lows.iloc[pivot_idx]
    if (highs.iloc[pivot_idx - left : pivot_idx] < center_high).all() and (
        highs.iloc[pivot_idx + 1 : pivot_idx + right + 1] < center_high
    ).all():
        if len(structure.pivots) > 5:
            structure.pivots.pop()
        structure.pivots.insert(0, Pivot(center_high, pivot_idx, 1, time=times[pivot_idx]))
    if (lows.iloc[pivot_idx - left : pivot_idx] > center_low).all() and (
        lows.iloc[pivot_idx + 1 : pivot_idx + right + 1] > center_low
    ).all():
        if len(structure.pivots) > 5:
            structure.pivots.pop()
        structure.pivots.insert(0, Pivot(center_low, pivot_idx, -1, time=times[pivot_idx]))


def pivot_labels(structure: Structure, bar_index: int) -> List[Label]:
    labels: List[Label] = []
    txt = ""
    for pivot in structure.pivots:
        if pivot.bar_index != bar_index - structure.right_length:
            continue
        for i in range(1, len(structure.pivots)):
            if len(structure.pivots) == 1:
                break
            previous = structure.pivots[i]
            if previous.type != pivot.type or previous.bar_index == pivot.bar_index:
                continue
            if pivot.price == previous.price:
                txt = "EQ"
            elif pivot.price > previous.price:
                txt = "H"
            else:
                txt = "L"
            break
        transparency = 60 if structure.type == StructureType.INTERNAL else 20
        if pivot.type == -1:
            labels.append(
                Label(
                    pivot.bar_index,
                    pivot.price,
                    f"{txt}L",
                    textcolor=f"teal@{transparency}",
                    style="label_up",
                    size=structure.font_size,
                )
            )
        if pivot.type == 1:
            labels.append(
                Label(
                    pivot.bar_index,
                    pivot.price,
                    f"{txt}H",
                    textcolor=f"red@{transparency}",
                    size=structure.font_size,
                )
            )
    return labels


def equal_high_or_low(
    structure: Structure,
    *,
    highs: pd.Series,
    lows: pd.Series,
    atr: pd.Series,
    bar_index: int,
) -> None:
    if bar_index < 2:
        return
    retest_high = highs.iloc[bar_index] < highs.iloc[bar_index - 1] and highs.iloc[bar_index - 1] > highs.iloc[bar_index - 2]
    retest_low = lows.iloc[bar_index] > lows.iloc[bar_index - 1] and lows.iloc[bar_index - 1] < lows.iloc[bar_index - 2]
    if retest_high:
        for equal_pivot in structure.equal_highs:
            price = highs.iloc[bar_index - 1]
            low_limit = equal_pivot.get_bottom() - (atr.iloc[bar_index] * (structure.equal_pivots_factor / 100.0))
            high_limit = equal_pivot.get_top()
            if in_limits(high_limit, low_limit, price) and not broken_by_bar_box(equal_pivot, 1, highs, lows, bar_index):
                if price < equal_pivot.get_bottom():
                    equal_pivot.set_rightbottom(bar_index - 1, price)
                else:
                    equal_pivot.set_right(bar_index - 1)
                    equal_pivot.set_top(price)
                Label(
                    bar_index - 1,
                    price,
                    "",
                    style=structure.extend_equal_pivots_style,
                    color=f"{structure.extend_equal_pivots_color}@70",
                    size=structure.font_size,
                )
                if structure.alert_equal_pivots:
                    alert("Added bar to existing equal high")
    if retest_low:
        for equal_pivot in structure.equal_lows:
            price = lows.iloc[bar_index - 1]
            low_limit = equal_pivot.get_bottom()
            high_limit = equal_pivot.get_top() + (atr.iloc[bar_index] * (structure.equal_pivots_factor / 100.0))
            if in_limits(high_limit, low_limit, price) and not broken_by_bar_box(equal_pivot, -1, highs, lows, bar_index):
                if price < equal_pivot.get_bottom():
                    equal_pivot.set_rightbottom(bar_index - 1, price)
                else:
                    equal_pivot.set_right(bar_index - 1)
                    equal_pivot.set_top(price)
                Label(
                    bar_index - 1,
                    price,
                    "",
                    style=structure.extend_equal_pivots_style,
                    color=f"{structure.extend_equal_pivots_color}@70",
                    size=structure.font_size,
                )
                if structure.alert_equal_pivots:
                    alert("Added bar to existing equal low")
    for pivot in structure.pivots:
        price = low_limit = high_limit = 0.0
        if pivot.type == -1:
            if not retest_low:
                continue
            price = lows.iloc[bar_index - 1]
            low_limit = pivot.price
            high_limit = pivot.price + (atr.iloc[bar_index] * (structure.equal_pivots_factor / 100.0))
        if pivot.type == 1:
            if not retest_high:
                continue
            price = highs.iloc[bar_index - 1]
            low_limit = pivot.price - (atr.iloc[bar_index] * (structure.equal_pivots_factor / 100.0))
            high_limit = pivot.price
        if in_limits(high_limit, low_limit, price) and not broken_by_bar_pivot(pivot, highs, lows, bar_index):
            top = max(price, pivot.price)
            bottom = min(price, pivot.price)
            exact_same_price = pivot.price == price
            equal_box = Box(
                pivot.bar_index,
                top,
                bar_index - 1,
                bottom,
                bgcolor=f"{structure.extend_equal_pivots_color}@70",
                border_width=1 if exact_same_price else 0,
                border_color=structure.extend_equal_pivots_color,
            )
            equal_box.set_extend_right(structure.extend_equal_pivots_zones)
            Label(
                bar_index - 1,
                price,
                "",
                style=structure.extend_equal_pivots_style,
                color=f"{structure.extend_equal_pivots_color}@70",
                size=structure.font_size,
            )
            Label(
                pivot.bar_index,
                pivot.price,
                "",
                style=structure.extend_equal_pivots_style,
                color=f"{structure.extend_equal_pivots_color}@70",
                size=structure.font_size,
            )
            if pivot.type == -1:
                equal_box.set_text("Equal low")
                structure.equal_lows.insert(0, equal_box)
                if structure.alert_equal_pivots:
                    alert("Equal low appeared")
            if pivot.type == 1:
                equal_box.set_text("Equal high")
                structure.equal_highs.insert(0, equal_box)
                if structure.alert_equal_pivots:
                    alert("Equal high appeared")


def break_of_structure(
    structure: Structure,
    *,
    closes: pd.Series,
    bar_index: int,
) -> Optional[Pivot]:
    break_of_structure_occured = None
    line_style = "solid" if structure.type == StructureType.SWING else "dashed"
    for pivot in structure.pivots:
        if structure.trend == 1 and pivot.type == 1 and closes.iloc[bar_index] > pivot.price and not pivot.break_of_structure_broken:
            create = True
            for bos in list(structure.break_of_structures):
                bar_idx = bos.line.x1
                if bar_idx > pivot.bar_index:
                    price = bos.line.y1
                    if price < pivot.price:
                        bos.line.delete()
                        bos.label.delete()
                        structure.break_of_structures.remove(bos)
                        continue
                    create = False
                    break
            if create:
                structure.break_of_structures.insert(
                    0,
                    StructureBreak(
                        Line(pivot.bar_index, pivot.price, bar_index, pivot.price, color="teal", style=line_style),
                        Label(
                            int(bar_index - ((bar_index - pivot.bar_index) / 2)),
                            pivot.price,
                            "BOS",
                            textcolor="teal@30",
                            size=structure.font_size,
                        ),
                    ),
                )
                pivot.break_of_structure_broken = True
                if structure.alert_break_of_structure:
                    alert_message = "BOS on an uptrend on internal market structure"
                    if structure.type == StructureType.SWING:
                        alert_message = "BOS on an uptrend on swing market structure"
                    alert(alert_message)
                break_of_structure_occured = pivot
                break
        if structure.trend == -1 and pivot.type == -1 and closes.iloc[bar_index] < pivot.price and not pivot.break_of_structure_broken:
            create = True
            for bos in list(structure.break_of_structures):
                bar_idx = bos.line.x1
                if bar_idx > pivot.bar_index:
                    price = bos.line.y1
                    if price > pivot.price:
                        bos.line.delete()
                        bos.label.delete()
                        structure.break_of_structures.remove(bos)
                        continue
                    create = False
                    break
            if create:
                structure.break_of_structures.insert(
                    0,
                    StructureBreak(
                        Line(pivot.bar_index, pivot.price, bar_index, pivot.price, color="red", style=line_style),
                        Label(
                            int(bar_index - ((bar_index - pivot.bar_index) / 2)),
                            pivot.price,
                            "BOS",
                            textcolor="red@30",
                            style="label_up",
                            size=structure.font_size,
                        ),
                    ),
                )
                pivot.break_of_structure_broken = True
                if structure.alert_break_of_structure:
                    alert_message = "BOS on a downtrend on internal market structure"
                    if structure.type == StructureType.SWING:
                        alert_message = "BOS on a downtrend on swing market structure"
                    alert(alert_message)
                break_of_structure_occured = pivot
                break
    return break_of_structure_occured


def change_of_character(
    structure: Structure,
    *,
    closes: pd.Series,
    bar_index: int,
) -> Optional[Pivot]:
    change_of_character_occured = None
    line_style = "dashed" if structure.type == StructureType.INTERNAL else "solid"
    for pivot in structure.pivots:
        if structure.trend <= 0 and pivot.type == 1 and closes.iloc[bar_index] > pivot.price and closes.iloc[bar_index - 1] < pivot.price and not pivot.change_of_character_broken:
            pivot.change_of_character_broken = True
            txt = "CHoCH"
            if len(structure.pivots) >= 2 and structure.trend != 0:
                for i in range(len(structure.pivots) - 1):
                    latest_pivot = structure.pivots[i]
                    if latest_pivot.type != -1:
                        continue
                    for j in range(i + 1, len(structure.pivots) - 1):
                        next_latest = structure.pivots[j]
                        if next_latest.type == -1:
                            if latest_pivot.price > next_latest.price:
                                txt = "CHoCH+"
                            break
                    break
            Line(pivot.bar_index, pivot.price, bar_index, pivot.price, color="teal", style=line_style)
            Label(
                int(bar_index - ((bar_index - pivot.bar_index) / 2)),
                pivot.price,
                txt,
                textcolor="teal@30",
                size=structure.font_size,
            )
            structure.trend = 1
            structure.equal_highs.clear()
            structure.equal_lows.clear()
            structure.break_of_structures.clear()
            remaining = []
            for p in structure.pivots:
                if p.bar_index <= pivot.bar_index:
                    continue
                p.break_of_structure_broken = True
                remaining.append(p)
            for p in remaining:
                if p.bar_index != pivot.bar_index:
                    p.change_of_character_broken = False
            structure.pivots = remaining
            if structure.alert_change_of_character:
                alert_message = f"{txt} to an uptrend on internal market structure"
                if structure.type == StructureType.SWING:
                    alert_message = f"{txt} to an uptrend on swing market structure"
                alert(alert_message)
            change_of_character_occured = pivot
            break
        if structure.trend >= 0 and pivot.type == -1 and closes.iloc[bar_index] < pivot.price and closes.iloc[bar_index - 1] > pivot.price and not pivot.change_of_character_broken:
            pivot.change_of_character_broken = True
            txt = "CHoCH"
            if len(structure.pivots) >= 2 and structure.trend != 0:
                for i in range(len(structure.pivots) - 1):
                    latest_pivot = structure.pivots[i]
                    if latest_pivot.type != 1:
                        continue
                    for j in range(i + 1, len(structure.pivots) - 1):
                        next_latest = structure.pivots[j]
                        if next_latest.type == 1:
                            if latest_pivot.price < next_latest.price:
                                txt = "CHoCH+"
                            break
                    break
            Line(pivot.bar_index, pivot.price, bar_index, pivot.price, color="red", style=line_style)
            Label(
                int(bar_index - ((bar_index - pivot.bar_index) / 2)),
                pivot.price,
                txt,
                textcolor="red@30",
                style="label_up",
                size=structure.font_size,
            )
            structure.trend = -1
            structure.equal_highs.clear()
            structure.equal_lows.clear()
            structure.break_of_structures.clear()
            remaining = []
            for p in structure.pivots:
                if p.bar_index <= pivot.bar_index:
                    continue
                p.break_of_structure_broken = True
                remaining.append(p)
            for p in remaining:
                if p.bar_index != pivot.bar_index:
                    p.change_of_character_broken = False
            structure.pivots = remaining
            if structure.alert_change_of_character:
                alert_message = f"{txt} to a downtrend on internal market structure"
                if structure.type == StructureType.SWING:
                    alert_message = f"{txt} to a downtrend on swing market structure"
                alert(alert_message)
            change_of_character_occured = pivot
            break
    return change_of_character_occured


def visualize_current(structure: Structure, highs: pd.Series, lows: pd.Series, bar_index: int) -> Box:
    trading_range = Box(
        bar_index,
        highs.iloc[bar_index],
        bar_index,
        highs.iloc[bar_index],
        bgcolor="gray@70",
        text="",
        extend_right=True,
    )
    latest_high = None
    latest_low = None
    for pivot in structure.pivots:
        if pivot.type == -1 and latest_low is None:
            latest_low = pivot
        if pivot.type == 1 and latest_high is None:
            latest_high = pivot
    if latest_high is not None and structure.right_length == bar_index - latest_high.bar_index:
        trading_range.left = bar_index - (bar_index - latest_high.bar_index)
        trading_range.top = highs.iloc[latest_high.bar_index]
    if latest_low is not None and structure.right_length == bar_index - latest_low.bar_index:
        trading_range.right = bar_index - (bar_index - latest_low.bar_index)
        trading_range.bottom = lows.iloc[latest_low.bar_index]
    return trading_range
