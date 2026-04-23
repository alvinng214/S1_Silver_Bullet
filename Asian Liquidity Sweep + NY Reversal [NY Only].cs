// =============================================================================
// Asian Liquidity Sweep + NY Reversal [NY Only].cs — C# cTrader port
// Original Pine Script: "Asian Liquidity Sweep + NY Reversal [NY Only]"
// =============================================================================
// Changes vs Pine:
//   - spreadBuffer removed (Q1: was defined in Pine but never used in any calc)
//   - Timezone added as a dropdown parameter (Q2: Pine hardcodes UTC+3)
//   - Session hours added as parameters (naturally pairs with the timezone dropdown)
//   - Session backgrounds implemented as filled rectangles (Q3: no cTrader bgcolor)
//   - Architecture: last-bar-only + ClearObjects + redraw (avoids object duplication)
//
// Pine logic faithfully reproduced:
//   Asia range    : rolling max/min during Asia session; frozen after; reset daily.
//   Sweep detection: after Asia only; sweepUp and sweepDown are mutually exclusive;
//                    each fires at most once per day; second condition overrides first
//                    on the same bar (Pine's if-if order, not if-else).
//   NY signals    : SELL = inNY & sweepUp & (bearish candle & high > asiaHigh)
//                   BUY  = inNY & sweepDown & (bullish candle & low < asiaLow)
//                   Rising-edge only (first bar the condition becomes true).
//   Day boundary  : UTC midnight (matches Pine's ta.change(time("D")) for most brokers).
//   Session bounds: inclusive on both ends (Pine: time >= start AND time <= end).
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;

namespace cAlgo
{
    // ── Timezone enum — must be at namespace level for cTrader dropdown ────────
    public enum TimezoneOption
    {
        America_New_York, Asia_Bangkok,
        GMT_Minus12, GMT_Minus11, GMT_Minus10, GMT_Minus9,  GMT_Minus8,
        GMT_Minus7,  GMT_Minus6,  GMT_Minus5,  GMT_Minus4,  GMT_Minus3,
        GMT_Minus2,  GMT_Minus1,  GMT_0,
        GMT_Plus1,   GMT_Plus2,   GMT_Plus3,   GMT_Plus4,   GMT_Plus5,
        GMT_Plus6,   GMT_Plus7,   GMT_Plus8,   GMT_Plus9,   GMT_Plus10,
        GMT_Plus11,  GMT_Plus12,
    }

    // Signal shape choices — maps to cTrader ChartIconType
    public enum SignalShape
    {
        UpTriangle,    // ▲  Pine default BUY   (shape.triangleup)
        DownTriangle,  // ▼  Pine default SELL  (shape.triangledown)
        UpArrow,       // ↑  Pine default SWEEP↓ (shape.labelup)
        DownArrow,     // ↓  Pine default SWEEP↑ (shape.labeldown)
    }

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class AsianLiquiditySweepNYReversal : Indicator
    {
        private const string Prefix = "ALSNY_";

        // ── Inner types ───────────────────────────────────────────────────────

        // One signal event to draw (collected during scan, drawn at end)
        private struct SignalEvent
        {
            public DateTime   Utc;
            public double     Price;      // bar high (above) or bar low (below)
            public bool       Above;      // true = draw above the bar, false = below
            public string     Text;
            public Color      Col;
            public SignalShape Shape;     // icon shape to draw
            public bool       ShowText;   // whether to show the text label
        }

        // Accumulated state for one calendar day (UTC day boundary)
        private sealed class DayState
        {
            public double?   AsiaHigh;
            public double?   AsiaLow;
            public DateTime  AsiaLineStartUtc;        // first Asia bar → H/L line origin
            public DateTime  DayLastBarUtc;           // last bar of day → H/L line end
            public DateTime? AsiaRectStartUtc;        // background rect
            public DateTime? AsiaRectEndUtc;
            public DateTime? NyRectStartUtc;
            public DateTime? NyRectEndUtc;
            public bool      SweepUpDetected;
            public bool      SweepDownDetected;
            public readonly List<SignalEvent> Events = new List<SignalEvent>();
        }

        // ── Parameters — Sessions ─────────────────────────────────────────────

        // Pine default timezone: GMT+3 (hardcoded). Made configurable here.
        [Parameter("Timezone", Group = "Sessions", DefaultValue = TimezoneOption.GMT_Plus3)]
        public TimezoneOption Timezone { get; set; }

        // Pine: asiaStartHour = 4, asiaEndHour = 10 (in the configured timezone)
        [Parameter("Asia Start Hour", Group = "Sessions", DefaultValue = 4, MinValue = 0, MaxValue = 23)]
        public int AsiaStartHour { get; set; }

        [Parameter("Asia End Hour", Group = "Sessions", DefaultValue = 10, MinValue = 0, MaxValue = 23)]
        public int AsiaEndHour { get; set; }

        // Pine: nyStartHour = 15, nyEndHour = 18
        [Parameter("NY Start Hour", Group = "Sessions", DefaultValue = 15, MinValue = 0, MaxValue = 23)]
        public int NyStartHour { get; set; }

        [Parameter("NY End Hour", Group = "Sessions", DefaultValue = 18, MinValue = 0, MaxValue = 23)]
        public int NyEndHour { get; set; }

        // ── Parameters — Debug ────────────────────────────────────────────────

        [Parameter("Show Sweep Detections", Group = "Debug", DefaultValue = true)]
        public bool ShowSweepDetections { get; set; }

        // ── Parameters — Display ──────────────────────────────────────────────

        // Pine: plot(asiaHigh, color=color.new(color.blue, 70)) → #4D0000FF
        [Parameter("Asia High Line Color", Group = "Display", DefaultValue = "#4D0000FF")]
        public Color AsiaHighColor { get; set; }

        // Pine: plot(asiaLow, color=color.new(color.orange, 70)) → #4DFFA500
        [Parameter("Asia Low Line Color", Group = "Display", DefaultValue = "#4DFFA500")]
        public Color AsiaLowColor { get; set; }

        [Parameter("Show Asia Background", Group = "Display", DefaultValue = true)]
        public bool ShowAsiaBg { get; set; }

        // Pine: bgcolor(inAsia ? color.new(color.blue, 90) : na) → #190000FF
        [Parameter("Asia Background Color", Group = "Display", DefaultValue = "#190000FF")]
        public Color AsiaBgColor { get; set; }

        [Parameter("Show NY Background", Group = "Display", DefaultValue = true)]
        public bool ShowNyBg { get; set; }

        // Pine: bgcolor(inNY ? color.new(color.purple, 90) : na) → #19800080
        [Parameter("NY Background Color", Group = "Display", DefaultValue = "#19800080")]
        public Color NyBgColor { get; set; }

        // Pine: sweepUpPlot color=color.red
        [Parameter("Sweep Up Color", Group = "Display", DefaultValue = "#FFFF0000")]
        public Color SweepUpColor { get; set; }

        // Pine: sweepDownPlot color=color.green
        [Parameter("Sweep Down Color", Group = "Display", DefaultValue = "#FF008000")]
        public Color SweepDownColor { get; set; }

        // Pine: sellPlot color=color.red
        [Parameter("Sell Signal Color", Group = "Display", DefaultValue = "#FFFF0000")]
        public Color SellColor { get; set; }

        // Pine: buyPlot color=color.green
        [Parameter("Buy Signal Color", Group = "Display", DefaultValue = "#FF008000")]
        public Color BuyColor { get; set; }

        // ── Parameters — Signal Shapes ────────────────────────────────────────
        // Matches Pine plotshape() style options for each signal type.

        // Pine: shape.triangleup (▲) for BUY
        [Parameter("Buy Signal Shape", Group = "Signal Shapes", DefaultValue = SignalShape.UpTriangle)]
        public SignalShape BuyShape { get; set; }

        // Pine: shape.triangledown (▼) for SELL
        [Parameter("Sell Signal Shape", Group = "Signal Shapes", DefaultValue = SignalShape.DownTriangle)]
        public SignalShape SellShape { get; set; }

        // Pine: shape.labeldown (↓) for SWEEP↑ — label points down above bar
        [Parameter("Sweep Up Shape", Group = "Signal Shapes", DefaultValue = SignalShape.DownArrow)]
        public SignalShape SweepUpShape { get; set; }

        // Pine: shape.labelup (↑) for SWEEP↓ — label points up below bar
        [Parameter("Sweep Down Shape", Group = "Signal Shapes", DefaultValue = SignalShape.UpArrow)]
        public SignalShape SweepDownShape { get; set; }

        // ── Parameters — Signal Visibility ───────────────────────────────────

        [Parameter("Show BUY Signal", Group = "Signal Visibility", DefaultValue = true)]
        public bool ShowBuySignal { get; set; }

        [Parameter("Show SELL Signal", Group = "Signal Visibility", DefaultValue = true)]
        public bool ShowSellSignal { get; set; }

        [Parameter("Show SWEEP Signal", Group = "Signal Visibility", DefaultValue = true)]
        public bool ShowSweepSignal { get; set; }

        // ── Parameters — Signal Text ──────────────────────────────────────────

        [Parameter("Show BUY Text", Group = "Signal Text", DefaultValue = true)]
        public bool ShowBuyText { get; set; }

        [Parameter("Show SELL Text", Group = "Signal Text", DefaultValue = true)]
        public bool ShowSellText { get; set; }

        [Parameter("Show SWEEP Text", Group = "Signal Text", DefaultValue = true)]
        public bool ShowSweepText { get; set; }

        [Parameter("Text Size", Group = "Signal Text", DefaultValue = 12, MinValue = 6, MaxValue = 48)]
        public int SignalTextSize { get; set; }

        // ── Private fields ────────────────────────────────────────────────────

        private TimeZoneInfo _tz;
        private int          _objId;

        // ── Initialize ────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            _tz = ResolveTimezone(Timezone);
        }

        // ── Calculate ─────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1) return;

            _tz    = ResolveTimezone(Timezone);
            _objId = 0;
            ClearObjects();

            // Chart price range used for full-height session background rectangles.
            // Approximates Pine's bgcolor() which fills the entire chart background.
            double chartTop = double.MinValue;
            double chartBot = double.MaxValue;
            for (int i = 0; i < Bars.Count; i++)
            {
                if (Bars.HighPrices[i] > chartTop) chartTop = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < chartBot) chartBot = Bars.LowPrices[i];
            }
            double rangePad = (chartTop - chartBot) * 0.1;
            chartTop += rangePad;
            chartBot -= rangePad;

            // ── Bar scan — build per-day state ────────────────────────────────
            var days = new List<DayState>();
            DayState day = null;

            // Track previous-bar signal state for rising-edge detection
            bool prevSellSignal = false;
            bool prevBuySignal  = false;

            for (int i = 0; i < Bars.Count; i++)
            {
                DateTime utcOpen  = Bars.OpenTimes[i];
                DateTime local    = ToLocal(utcOpen);
                bool     isClosed = i < Bars.Count - 1; // forming bar guard

                // New UTC day → reset all stateful variables (matches Pine's ta.change(time("D")))
                bool isNewDay = i == 0 || utcOpen.Date != Bars.OpenTimes[i - 1].Date;
                if (isNewDay)
                {
                    day = new DayState();
                    days.Add(day);
                    prevSellSignal = false;
                    prevBuySignal  = false;
                }

                day.DayLastBarUtc = utcOpen; // updated each bar; final value = last bar of day

                // Pine: inAsia = time >= asiaStart AND time <= asiaEnd (both inclusive)
                int  localHHMM = local.Hour * 60 + local.Minute;
                bool inAsia    = localHHMM >= AsiaStartHour * 60 && localHHMM <= AsiaEndHour * 60;
                bool inNY      = localHHMM >= NyStartHour   * 60 && localHHMM <= NyEndHour   * 60;

                double hi = Bars.HighPrices[i];
                double lo = Bars.LowPrices[i];
                double cl = Bars.ClosePrices[i];
                double op = Bars.OpenPrices[i];

                // ── Asia range accumulation ───────────────────────────────────
                // Pine: if inAsia → asiaHigh = na(asiaHigh) ? high : max(asiaHigh, high)
                if (inAsia)
                {
                    if (!day.AsiaHigh.HasValue)
                    {
                        day.AsiaHigh          = hi;
                        day.AsiaLow           = lo;
                        day.AsiaLineStartUtc  = utcOpen;
                        day.AsiaRectStartUtc  = utcOpen;
                    }
                    else
                    {
                        day.AsiaHigh = Math.Max(day.AsiaHigh.Value, hi);
                        day.AsiaLow  = Math.Min(day.AsiaLow.Value,  lo);
                    }
                    day.AsiaRectEndUtc = utcOpen;
                }

                // ── NY background bounds ──────────────────────────────────────
                if (inNY)
                {
                    if (!day.NyRectStartUtc.HasValue) day.NyRectStartUtc = utcOpen;
                    day.NyRectEndUtc = utcOpen;
                }

                // No sweep or signal detection on forming bar
                if (!isClosed) continue;

                // ── Sweep detection ───────────────────────────────────────────
                // Pine: runs only when NOT in Asia AND asiaHigh/Low are known
                if (!inAsia && day.AsiaHigh.HasValue)
                {
                    double ah = day.AsiaHigh.Value;
                    double al = day.AsiaLow.Value;

                    bool sweepUpThisBar   = false;
                    bool sweepDownThisBar = false;

                    // Pine first if: check sweep up
                    if (!day.SweepUpDetected && hi > ah && cl <= ah)
                    {
                        day.SweepUpDetected   = true;
                        day.SweepDownDetected = false; // mutually exclusive in Pine
                        sweepUpThisBar        = true;
                    }

                    // Pine second if: check sweep down — runs REGARDLESS, can override above.
                    // If both conditions met on same bar, sweepDown wins (Pine's if-if, not if-else).
                    if (!day.SweepDownDetected && lo < al && cl >= al)
                    {
                        day.SweepDownDetected = true;
                        day.SweepUpDetected   = false;
                        sweepDownThisBar      = true;
                        sweepUpThisBar        = false; // down overrides up on same bar
                    }

                    // Add rising-edge events (sweep down wins if both fired)
                    if (sweepDownThisBar && ShowSweepDetections && ShowSweepSignal)
                        day.Events.Add(new SignalEvent { Utc = utcOpen, Price = lo, Above = false, Text = "SWEEP↓", Col = SweepDownColor, Shape = SweepDownShape, ShowText = ShowSweepText });
                    else if (sweepUpThisBar && ShowSweepDetections && ShowSweepSignal)
                        day.Events.Add(new SignalEvent { Utc = utcOpen, Price = hi, Above = true,  Text = "SWEEP↑", Col = SweepUpColor,   Shape = SweepUpShape,   ShowText = ShowSweepText });
                }

                // ── NY reversal signals ───────────────────────────────────────
                // Pine: sellSignal = inNY AND sweepUpDetected AND nyCandleBearish
                //       buySignal  = inNY AND sweepDownDetected AND nyCandleBullish
                if (inNY && day.AsiaHigh.HasValue)
                {
                    double ah = day.AsiaHigh.Value;
                    double al = day.AsiaLow.Value;

                    // Pine: nyCandleBearish = close < open AND high > asiaHigh
                    bool nyCandleBearish = cl < op && hi > ah;
                    // Pine: nyCandleBullish = close > open AND low < asiaLow
                    bool nyCandleBullish = cl > op && lo < al;

                    bool sellSignal = day.SweepUpDetected   && nyCandleBearish;
                    bool buySignal  = day.SweepDownDetected && nyCandleBullish;

                    // Pine: sellPlot = sellSignal AND NOT sellSignal[1] — rising edge only
                    if (sellSignal && !prevSellSignal)
                        if (ShowSellSignal)
                            day.Events.Add(new SignalEvent { Utc = utcOpen, Price = hi, Above = true,  Text = "SELL", Col = SellColor, Shape = SellShape, ShowText = ShowSellText });
                    if (buySignal && !prevBuySignal)
                        if (ShowBuySignal)
                            day.Events.Add(new SignalEvent { Utc = utcOpen, Price = lo, Above = false, Text = "BUY",  Col = BuyColor,  Shape = BuyShape,  ShowText = ShowBuyText  });

                    prevSellSignal = sellSignal;
                    prevBuySignal  = buySignal;
                }
                else
                {
                    // Reset when not in NY — ensures rising edge fires correctly on next NY entry
                    prevSellSignal = false;
                    prevBuySignal  = false;
                }
            }

            // ── Draw everything ───────────────────────────────────────────────
            // Label offset: push text slightly above high / below low for readability
            double labelOff = 5 * Symbol.PipSize;

            foreach (DayState d in days)
            {
                // Skip days with no Asia session data
                if (!d.AsiaHigh.HasValue) continue;

                double   ah      = d.AsiaHigh.Value;
                double   al      = d.AsiaLow.Value;
                DateTime lineEnd = d.DayLastBarUtc;

                // Asia H/L lines — from first Asia bar to end of day
                // Pine: plot(asiaHigh, style=plot.style_linebr) — visible when not na
                Chart.DrawTrendLine(Prefix + "ah_" + NextId(),
                    d.AsiaLineStartUtc, ah, lineEnd, ah,
                    AsiaHighColor, 1, LineStyle.Solid);
                Chart.DrawTrendLine(Prefix + "al_" + NextId(),
                    d.AsiaLineStartUtc, al, lineEnd, al,
                    AsiaLowColor, 1, LineStyle.Solid);

                // Asia session background rectangle
                // Pine: bgcolor(inAsia ? color.new(color.blue, 90) : na)
                if (ShowAsiaBg && d.AsiaRectStartUtc.HasValue && d.AsiaRectEndUtc.HasValue)
                {
                    var r = Chart.DrawRectangle(Prefix + "abg_" + NextId(),
                        d.AsiaRectStartUtc.Value, chartTop,
                        d.AsiaRectEndUtc.Value,   chartBot,
                        AsiaBgColor, 1, LineStyle.Solid);
                    r.IsFilled = true;
                    r.Color    = AsiaBgColor;
                }

                // NY session background rectangle
                // Pine: bgcolor(inNY ? color.new(color.purple, 90) : na)
                if (ShowNyBg && d.NyRectStartUtc.HasValue && d.NyRectEndUtc.HasValue)
                {
                    var r = Chart.DrawRectangle(Prefix + "nybg_" + NextId(),
                        d.NyRectStartUtc.Value, chartTop,
                        d.NyRectEndUtc.Value,   chartBot,
                        NyBgColor, 1, LineStyle.Solid);
                    r.IsFilled = true;
                    r.Color    = NyBgColor;
                }

                // Signal icons + labels (SWEEP↑/↓, SELL, BUY)
                foreach (SignalEvent ev in d.Events)
                {
                    double iconY = ev.Above ? ev.Price + labelOff       : ev.Price - labelOff;
                    double txtY  = ev.Above ? ev.Price + labelOff * 2.0 : ev.Price - labelOff * 2.0;
                    // Icon (shape chosen by parameter)
                    Chart.DrawIcon(Prefix + "evi_" + NextId(), ToChartIcon(ev.Shape), ev.Utc, iconY, ev.Col);
                    // Text label — only if enabled for this signal type
                    if (ev.ShowText)
                    {
                        var ct = Chart.DrawText(Prefix + "evt_" + NextId(), ev.Text, ev.Utc, txtY, ev.Col);
                        ct.FontSize = SignalTextSize;
                    }
                }
            }
        }

        // ── Utilities ─────────────────────────────────────────────────────────

        private DateTime ToLocal(DateTime utc)
            => TimeZoneInfo.ConvertTimeFromUtc(
                   DateTime.SpecifyKind(utc, DateTimeKind.Utc), _tz);

        private static TimeZoneInfo ResolveTimezone(TimezoneOption opt)
        {
            switch (opt)
            {
                case TimezoneOption.America_New_York:
                    try   { return TimeZoneInfo.FindSystemTimeZoneById("America/New_York"); }
                    catch { return TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"); }

                case TimezoneOption.Asia_Bangkok:
                    try   { return TimeZoneInfo.FindSystemTimeZoneById("Asia/Bangkok"); }
                    catch { return TimeZoneInfo.FindSystemTimeZoneById("SE Asia Standard Time"); }

                case TimezoneOption.GMT_Minus12: return MakeGmt(-12);
                case TimezoneOption.GMT_Minus11: return MakeGmt(-11);
                case TimezoneOption.GMT_Minus10: return MakeGmt(-10);
                case TimezoneOption.GMT_Minus9:  return MakeGmt(-9);
                case TimezoneOption.GMT_Minus8:  return MakeGmt(-8);
                case TimezoneOption.GMT_Minus7:  return MakeGmt(-7);
                case TimezoneOption.GMT_Minus6:  return MakeGmt(-6);
                case TimezoneOption.GMT_Minus5:  return MakeGmt(-5);
                case TimezoneOption.GMT_Minus4:  return MakeGmt(-4);
                case TimezoneOption.GMT_Minus3:  return MakeGmt(-3);
                case TimezoneOption.GMT_Minus2:  return MakeGmt(-2);
                case TimezoneOption.GMT_Minus1:  return MakeGmt(-1);
                case TimezoneOption.GMT_0:        return MakeGmt(0);
                case TimezoneOption.GMT_Plus1:   return MakeGmt(1);
                case TimezoneOption.GMT_Plus2:   return MakeGmt(2);
                case TimezoneOption.GMT_Plus3:   return MakeGmt(3);   // Pine default
                case TimezoneOption.GMT_Plus4:   return MakeGmt(4);
                case TimezoneOption.GMT_Plus5:   return MakeGmt(5);
                case TimezoneOption.GMT_Plus6:   return MakeGmt(6);
                case TimezoneOption.GMT_Plus7:   return MakeGmt(7);
                case TimezoneOption.GMT_Plus8:   return MakeGmt(8);   // HKT
                case TimezoneOption.GMT_Plus9:   return MakeGmt(9);
                case TimezoneOption.GMT_Plus10:  return MakeGmt(10);
                case TimezoneOption.GMT_Plus11:  return MakeGmt(11);
                case TimezoneOption.GMT_Plus12:  return MakeGmt(12);
                default:                          return TimeZoneInfo.Utc;
            }
        }

        private static TimeZoneInfo MakeGmt(int hours)
            => TimeZoneInfo.CreateCustomTimeZone(
                "GMT" + (hours >= 0 ? "+" : "") + hours,
                TimeSpan.FromHours(hours),
                "GMT" + (hours >= 0 ? "+" : "") + hours,
                "GMT" + (hours >= 0 ? "+" : "") + hours);

        private static ChartIconType ToChartIcon(SignalShape s)
        {
            switch (s)
            {
                case SignalShape.DownTriangle: return ChartIconType.DownTriangle;
                case SignalShape.UpArrow:      return ChartIconType.UpArrow;
                case SignalShape.DownArrow:    return ChartIconType.DownArrow;
                default:                      return ChartIconType.UpTriangle;
            }
        }

        private string NextId()
            => (_objId++).ToString(CultureInfo.InvariantCulture);

        private void ClearObjects()
        {
            var names = new List<string>();
            foreach (var obj in Chart.Objects)
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    names.Add(obj.Name);
            foreach (var n in names)
                Chart.RemoveObject(n);
        }
    }
}
