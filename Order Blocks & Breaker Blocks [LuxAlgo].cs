// This work is licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
// https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo — C# port for cTrader

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    // -------------------------------------------------------------------------
    // Order Blocks & Breaker Blocks [LuxAlgo]
    //
    // Direct port of the Pine Script v5 indicator by LuxAlgo.
    //
    // Logic summary
    // -------------
    // 1. Swing detection (mirrors Pine's swings(len) function)
    //    - Tracks a rolling os (oscillator) that flips between 0 (swing-high
    //      mode) and 1 (swing-low mode) using ta.highest / ta.lowest windows.
    //    - A swing HIGH is confirmed when os transitions 1→0: the pivot is
    //      located at bar[index - length].
    //    - A swing LOW  is confirmed when os transitions 0→1: same offset.
    //
    // 2. Bullish Order Block detection
    //    - Fires once when close crosses above the last confirmed swing high.
    //    - Scans bars between current and the swing top; picks the bar with
    //      the lowest low (or body bottom when UseBody = true) as the OB candle.
    //
    // 3. Bearish Order Block detection
    //    - Fires once when close crosses below the last confirmed swing low.
    //    - Picks the bar with the highest high (or body top) as the OB candle.
    //
    // 4. OB → Breaker promotion
    //    - Bullish OB  : becomes a Breaker when min(close,open) < OB.btm
    //    - Bearish OB  : becomes a Breaker when max(close,open) > OB.top
    //
    // 5. Breaker invalidation / removal
    //    - Bullish Breaker removed when close > OB.top
    //    - Bearish Breaker removed when close < OB.btm
    //
    // 6. Polarity-change labels (ShowLabels)
    //    - Bull polarity: ▼ at swing top when a visible bullish breaker
    //      contains the current swing top (rising edge of bull_break_conf).
    //    - Bear polarity: ▲ at swing low when a visible bearish breaker
    //      contains the current swing low.
    //
    // 7. Drawing (mirrors Pine's barstate.islast redraw pattern)
    //    - All OB/line objects are deleted and redrawn on every call to the
    //      last bar, keeping only the most-recent ShowBull / ShowBear OBs
    //      visible — identical to Pine's behaviour.
    // -------------------------------------------------------------------------
    [Indicator("Order Blocks & Breaker Blocks [LuxAlgo]", IsOverlay = true,
               TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OrderBlocksBreakerBlocksLuxAlgo : Indicator
    {
        // --------------------------------------------------------------------- //
        //  Parameters                                                            //
        // --------------------------------------------------------------------- //

        [Parameter("Swing Lookback", DefaultValue = 10, MinValue = 3)]
        public int Length { get; set; }

        [Parameter("Show Last Bullish OB", DefaultValue = 3, MinValue = 0)]
        public int ShowBull { get; set; }

        [Parameter("Show Last Bearish OB", DefaultValue = 3, MinValue = 0)]
        public int ShowBear { get; set; }

        [Parameter("Use Candle Body", DefaultValue = false)]
        public bool UseBody { get; set; }

        // Style — colours match Pine's color.new(hex, 80) = 20 % opacity → alpha 0x33
        [Parameter("Bullish OB", DefaultValue = "#332157F3", Group = "Style")]
        public Color BullCss { get; set; }

        [Parameter("Bullish Break", DefaultValue = "#33FF1100", Group = "Style")]
        public Color BullBreakCss { get; set; }

        [Parameter("Bearish OB", DefaultValue = "#33FF5D00", Group = "Style")]
        public Color BearCss { get; set; }

        [Parameter("Bearish Break", DefaultValue = "#330CB51A", Group = "Style")]
        public Color BearBreakCss { get; set; }

        [Parameter("Show Historical Polarity Changes", DefaultValue = false)]
        public bool ShowLabels { get; set; }

        // --------------------------------------------------------------------- //
        //  Internal types (mirror Pine's UDTs)                                  //
        // --------------------------------------------------------------------- //

        /// <summary>Mirrors Pine's <c>swing</c> UDT.</summary>
        private sealed class SwingData
        {
            public double Y        = double.NaN;
            public int    BarIndex = -1;
            public bool   Crossed  = false;
        }

        /// <summary>Mirrors Pine's <c>ob</c> UDT.</summary>
        private sealed class ObData
        {
            public double Top;
            public double Btm;
            public int    LocIndex;        // bar index of the OB candle (Pine: ob.loc stored as time)
            public bool   Breaker;
            public int    BreakLocIndex;   // bar index where OB was broken  (Pine: ob.break_loc as time)
        }

        // --------------------------------------------------------------------- //
        //  State                                                                 //
        // --------------------------------------------------------------------- //

        private int       _os;          // swing oscillator: 0 = swing-high mode, 1 = swing-low mode
        private SwingData _topSwing;    // most recent confirmed swing high
        private SwingData _btmSwing;    // most recent confirmed swing low

        // Pine: var bullish_ob = array.new<ob>(0)  — index 0 = most recent
        private readonly List<ObData> _bullishObs = new List<ObData>();
        private readonly List<ObData> _bearishObs = new List<ObData>();

        // Polarity-change signal (mirrors Pine's bull_break_conf / bear_break_conf)
        private int _bullBreakConf;
        private int _bearBreakConf;

        private const string Prefix = "OBB_";   // prefix for all chart objects

        // --------------------------------------------------------------------- //
        //  Initialize                                                            //
        // --------------------------------------------------------------------- //

        protected override void Initialize()
        {
            _os          = 0;
            _topSwing    = new SwingData();
            _btmSwing    = new SwingData();
            _bullBreakConf = 0;
            _bearBreakConf = 0;
        }

        // --------------------------------------------------------------------- //
        //  Calculate                                                             //
        // --------------------------------------------------------------------- //

        public override void Calculate(int index)
        {
            // Need at least Length + 1 bars for swing detection
            if (index < Length + 1)
                return;

            // --- Helpers mirroring Pine's max / min variables ---
            // max = useBody ? math.max(close, open) : high
            // min = useBody ? math.min(close, open) : low
            double MaxAt(int i) => UseBody
                ? Math.Max(Bars.ClosePrices[i], Bars.OpenPrices[i])
                : Bars.HighPrices[i];

            double MinAt(int i) => UseBody
                ? Math.Min(Bars.ClosePrices[i], Bars.OpenPrices[i])
                : Bars.LowPrices[i];

            // ----------------------------------------------------------------- //
            //  Swing detection  (mirrors Pine's swings(len) function)           //
            //                                                                   //
            //  upper = ta.highest(high, length) — window [index-length+1..index]
            //  lower = ta.lowest (low,  length) — same window                  //
            //  os := high[length] > upper ? 0 : low[length] < lower ? 1 : os   //
            // ----------------------------------------------------------------- //

            double upper = double.MinValue;
            double lower = double.MaxValue;
            for (int i = index - Length + 1; i <= index; i++)
            {
                if (Bars.HighPrices[i] > upper) upper = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lower) lower = Bars.LowPrices[i];
            }

            double highAtLen = Bars.HighPrices[index - Length];
            double lowAtLen  = Bars.LowPrices[index - Length];

            int newOs = highAtLen > upper ? 0
                      : lowAtLen  < lower ? 1
                      : _os;

            // os == 0 and os[1] != 0  →  swing HIGH confirmed at bar[index - length]
            if (newOs == 0 && _os != 0)
            {
                _topSwing = new SwingData
                {
                    Y        = Bars.HighPrices[index - Length],
                    BarIndex = index - Length,
                    Crossed  = false
                };
            }

            // os == 1 and os[1] != 1  →  swing LOW confirmed at bar[index - length]
            if (newOs == 1 && _os != 1)
            {
                _btmSwing = new SwingData
                {
                    Y        = Bars.LowPrices[index - Length],
                    BarIndex = index - Length,
                    Crossed  = false
                };
            }

            _os = newOs;

            double close = Bars.ClosePrices[index];
            double open  = Bars.OpenPrices[index];

            // ----------------------------------------------------------------- //
            //  Bullish OB detection                                             //
            //                                                                   //
            //  Pine: if close > top.y and not top.crossed                       //
            //    Scan bars [1 .. (n - top.x) - 1] bars back.                   //
            //    Find bar with minimum min[i]; record its max[i] as OB top.     //
            // ----------------------------------------------------------------- //

            if (!double.IsNaN(_topSwing.Y) && close > _topSwing.Y && !_topSwing.Crossed)
            {
                _topSwing.Crossed = true;

                // rangeLen = n - top.x = index - (index - Length) = Length
                // (but may be larger if swing was confirmed on a prior bar)
                int rangeLen = index - _topSwing.BarIndex;

                // Initial values mirror Pine:  minima = max[1],  maxima = min[1]
                double minima   = MaxAt(index - 1);
                double maxima   = MinAt(index - 1);
                int    obBarIdx = index - 1;

                // Pine: for i = 1 to (n - top.x) - 1  (inclusive upper bound)
                for (int i = 1; i < rangeLen; i++)
                {
                    int    bi   = index - i;
                    double minI = MinAt(bi);
                    double maxI = MaxAt(bi);

                    // Pine: minima := math.min(min[i], minima)
                    //       maxima := minima == min[i] ? max[i] : maxima
                    if (minI <= minima)
                    {
                        minima   = minI;
                        maxima   = maxI;
                        obBarIdx = bi;
                    }
                }

                _bullishObs.Insert(0, new ObData { Top = maxima, Btm = minima, LocIndex = obBarIdx });
            }

            // ----------------------------------------------------------------- //
            //  Bullish OB / Breaker state management                            //
            // ----------------------------------------------------------------- //

            int prevBullBreakConf = _bullBreakConf;
            _bullBreakConf = 0;

            // Iterate from end to start so RemoveAt(i) doesn't skip elements
            for (int i = _bullishObs.Count - 1; i >= 0; i--)
            {
                var ob = _bullishObs[i];

                if (!ob.Breaker)
                {
                    // Bullish OB broken below → becomes Breaker
                    // Pine: if math.min(close, open) < element.btm
                    if (Math.Min(close, open) < ob.Btm)
                    {
                        ob.Breaker       = true;
                        ob.BreakLocIndex = index;
                    }
                }
                else
                {
                    // Breaker fully invalidated (close above top)
                    if (close > ob.Top)
                    {
                        _bullishObs.RemoveAt(i);
                    }
                    // Polarity change: visible breaker contains the current swing top
                    // Pine: else if i < showBull and top.y < element.top and top.y > element.btm
                    else if (i < ShowBull
                          && !double.IsNaN(_topSwing.Y)
                          && _topSwing.Y < ob.Top
                          && _topSwing.Y > ob.Btm)
                    {
                        _bullBreakConf = 1;
                    }
                }
            }

            // Label ▼ at swing top on rising edge of bull_break_conf
            // Pine: if bull_break_conf > bull_break_conf[1] and showLabels
            if (ShowLabels && _bullBreakConf > prevBullBreakConf && !double.IsNaN(_topSwing.Y))
            {
                Chart.DrawIcon(
                    $"{Prefix}pl_b_{_topSwing.BarIndex}",
                    ChartIconType.DownTriangle,
                    _topSwing.BarIndex,
                    _topSwing.Y,
                    NoTransp(BearCss));
            }

            // ----------------------------------------------------------------- //
            //  Bearish OB detection                                             //
            //                                                                   //
            //  Pine: if close < btm.y and not btm.crossed                       //
            //    Scan bars [1 .. (n - btm.x) - 1] bars back.                   //
            //    Find bar with maximum max[i]; record its min[i] as OB bottom.  //
            // ----------------------------------------------------------------- //

            if (!double.IsNaN(_btmSwing.Y) && close < _btmSwing.Y && !_btmSwing.Crossed)
            {
                _btmSwing.Crossed = true;

                int rangeLen = index - _btmSwing.BarIndex;

                // Initial values mirror Pine:  minima = min[1],  maxima = max[1]
                double minima   = MinAt(index - 1);
                double maxima   = MaxAt(index - 1);
                int    obBarIdx = index - 1;

                // Pine: for i = 1 to (n - btm.x) - 1
                for (int i = 1; i < rangeLen; i++)
                {
                    int    bi   = index - i;
                    double minI = MinAt(bi);
                    double maxI = MaxAt(bi);

                    // Pine: maxima := math.max(max[i], maxima)
                    //       minima := maxima == max[i] ? min[i] : minima
                    if (maxI >= maxima)
                    {
                        maxima   = maxI;
                        minima   = minI;
                        obBarIdx = bi;
                    }
                }

                _bearishObs.Insert(0, new ObData { Top = maxima, Btm = minima, LocIndex = obBarIdx });
            }

            // ----------------------------------------------------------------- //
            //  Bearish OB / Breaker state management                            //
            // ----------------------------------------------------------------- //

            int prevBearBreakConf = _bearBreakConf;
            _bearBreakConf = 0;

            for (int i = _bearishObs.Count - 1; i >= 0; i--)
            {
                var ob = _bearishObs[i];

                if (!ob.Breaker)
                {
                    // Bearish OB broken above → becomes Breaker
                    // Pine: if math.max(close, open) > element.top
                    if (Math.Max(close, open) > ob.Top)
                    {
                        ob.Breaker       = true;
                        ob.BreakLocIndex = index;
                    }
                }
                else
                {
                    // Breaker fully invalidated (close below bottom)
                    if (close < ob.Btm)
                    {
                        _bearishObs.RemoveAt(i);
                    }
                    // Polarity change: visible breaker contains the current swing low
                    // Pine: else if i < showBear and btm.y > element.btm and btm.y < element.top
                    else if (i < ShowBear
                          && !double.IsNaN(_btmSwing.Y)
                          && _btmSwing.Y > ob.Btm
                          && _btmSwing.Y < ob.Top)
                    {
                        _bearBreakConf = 1;
                    }
                }
            }

            // Label ▲ at swing low on rising edge of bear_break_conf
            if (ShowLabels && _bearBreakConf > prevBearBreakConf && !double.IsNaN(_btmSwing.Y))
            {
                Chart.DrawIcon(
                    $"{Prefix}pl_r_{_btmSwing.BarIndex}",
                    ChartIconType.UpTriangle,
                    _btmSwing.BarIndex,
                    _btmSwing.Y,
                    NoTransp(BullCss));
            }

            // ----------------------------------------------------------------- //
            //  Draw on last bar  (mirrors Pine's barstate.islast redraw block)  //
            //                                                                   //
            //  Pine deletes all boxes and lines every bar, then redraws only    //
            //  at barstate.islast. We replicate this by clearing OBB_ prefixed  //
            //  box/line objects and redrawing whenever we are at the last bar.  //
            // ----------------------------------------------------------------- //

            if (index != Bars.Count - 1)
                return;

            RemoveObDrawings();

            // Pine: if showBull > 0  →  for i = 0 to math.min(showBull-1, bullish_ob.size())
            if (ShowBull > 0)
            {
                int count = Math.Min(ShowBull, _bullishObs.Count);
                for (int i = 0; i < count; i++)
                    DisplayOb(_bullishObs[i], i, isBull: true, currentIndex: index);
            }

            // Pine: if showBear > 0  →  for i = 0 to math.min(showBear-1, bearish_ob.size())
            if (ShowBear > 0)
            {
                int count = Math.Min(ShowBear, _bearishObs.Count);
                for (int i = 0; i < count; i++)
                    DisplayOb(_bearishObs[i], i, isBull: false, currentIndex: index);
            }
        }

        // --------------------------------------------------------------------- //
        //  Remove all OB box / line drawings                                    //
        //  Mirrors Pine:  for bx in box.all  bx.delete()                        //
        //                 for l  in line.all  l.delete()                         //
        // --------------------------------------------------------------------- //

        private void RemoveObDrawings()
        {
            var names = new List<string>();
            foreach (var obj in Chart.Objects)
            {
                var n = obj.Name;
                if (n.StartsWith(Prefix + "ob_") || n.StartsWith(Prefix + "ln_"))
                    names.Add(n);
            }
            foreach (var n in names)
                Chart.RemoveObject(n);
        }

        // --------------------------------------------------------------------- //
        //  Display a single OB  (mirrors Pine's ob.display() method)            //
        //                                                                       //
        //  Non-breaker                                                           //
        //    box  : ob.loc → now,         top → btm, no border, css fill        //
        //    lines: ob.loc → now+extend,  top and btm, opaque css, extend right //
        //                                                                       //
        //  Breaker                                                               //
        //    box1 : ob.loc     → break_loc, top → btm, opaque border + css fill //
        //    box2 : break_loc  → now+1,     top → btm, no border, break_css fill//
        //    lines: ob.loc     → break_loc, solid   opaque css                  //
        //    lines: break_loc  → now+extend, dashed opaque break_css, ext right //
        // --------------------------------------------------------------------- //

        private void DisplayOb(ObData ob, int listIndex, bool isBull, int currentIndex)
        {
            Color css          = isBull ? BullCss      : BearCss;
            Color breakCss     = isBull ? BullBreakCss : BearBreakCss;
            Color cssOpaque    = NoTransp(css);
            Color breakOpaque  = NoTransp(breakCss);

            string tag = $"{(isBull ? "b" : "r")}_{listIndex}";

            if (ob.Breaker)
            {
                // Box 1: original zone (before break)
                // Pine: box.new(id.loc, id.top, id.break_loc, id.btm,
                //               css.notransp(),  bgcolor = css, xloc = xloc.bar_time)
                var r1 = Chart.DrawRectangle(
                    $"{Prefix}ob_{tag}_1",
                    ob.LocIndex, ob.Top,
                    ob.BreakLocIndex, ob.Btm,
                    cssOpaque);
                r1.IsFilled = true;
                r1.Color    = css;

                // Box 2: continuation zone (after break), extending right
                // Pine: box.new(id.break_loc, id.top, time+1, id.btm,
                //               na, bgcolor = break_css, extend = extend.right)
                var r2 = Chart.DrawRectangle(
                    $"{Prefix}ob_{tag}_2",
                    ob.BreakLocIndex, ob.Top,
                    currentIndex + 1, ob.Btm,
                    Color.Transparent);
                r2.IsFilled = true;
                r2.Color    = breakCss;

                // Solid lines from ob.loc to break_loc  (opaque original colour)
                // Pine: line.new(id.loc, id.top, id.break_loc, id.top, ..., css.notransp())
                Chart.DrawTrendLine(
                    $"{Prefix}ln_{tag}_t1",
                    ob.LocIndex, ob.Top,
                    ob.BreakLocIndex, ob.Top,
                    cssOpaque, 1, LineStyle.Solid);

                Chart.DrawTrendLine(
                    $"{Prefix}ln_{tag}_b1",
                    ob.LocIndex, ob.Btm,
                    ob.BreakLocIndex, ob.Btm,
                    cssOpaque, 1, LineStyle.Solid);

                // Dashed lines from break_loc onward, extending right (opaque break colour)
                // Pine: line.new(id.break_loc, id.top, time+1, id.top,
                //               ..., extend.right, break_css.notransp(), line.style_dashed)
                var lt2 = Chart.DrawTrendLine(
                    $"{Prefix}ln_{tag}_t2",
                    ob.BreakLocIndex, ob.Top,
                    currentIndex, ob.Top,
                    breakOpaque, 1, LineStyle.Lines);
                lt2.ExtendToInfinity = true;

                var lb2 = Chart.DrawTrendLine(
                    $"{Prefix}ln_{tag}_b2",
                    ob.BreakLocIndex, ob.Btm,
                    currentIndex, ob.Btm,
                    breakOpaque, 1, LineStyle.Lines);
                lb2.ExtendToInfinity = true;
            }
            else
            {
                // Active order block — no border, semi-transparent fill, extending right
                // Pine: box.new(id.loc, id.top, time, id.btm,
                //               na, bgcolor = css, extend = extend.right)
                var r = Chart.DrawRectangle(
                    $"{Prefix}ob_{tag}_1",
                    ob.LocIndex, ob.Top,
                    currentIndex, ob.Btm,
                    Color.Transparent);
                r.IsFilled = true;
                r.Color    = css;

                // Top and bottom border lines, extending right
                // Pine: line.new(id.loc, id.top, time, id.top,
                //               xloc.bar_time, extend.right, css.notransp())
                var lt = Chart.DrawTrendLine(
                    $"{Prefix}ln_{tag}_t1",
                    ob.LocIndex, ob.Top,
                    currentIndex, ob.Top,
                    cssOpaque, 1, LineStyle.Solid);
                lt.ExtendToInfinity = true;

                var lb = Chart.DrawTrendLine(
                    $"{Prefix}ln_{tag}_b1",
                    ob.LocIndex, ob.Btm,
                    currentIndex, ob.Btm,
                    cssOpaque, 1, LineStyle.Solid);
                lb.ExtendToInfinity = true;
            }
        }

        // --------------------------------------------------------------------- //
        //  Helper: strip transparency  (mirrors Pine's notransp() method)       //
        //  color.rgb(color.r(css), color.g(css), color.b(css))  → alpha = 255   //
        // --------------------------------------------------------------------- //

        private static Color NoTransp(Color c)
            => Color.FromArgb(255, c.R, c.G, c.B);
    }
}
