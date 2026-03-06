// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © EmreKb — C# port for cTrader

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    // -------------------------------------------------------------------------
    // Market Structure Break & Order Block
    //
    // Pine Script v5 → cTrader/cAlgo C# port.
    //
    // Logic summary
    // -------------
    // 1. ZigZag trend detection
    //    - to_up  = current high >= highest high over ZigZagLen bars
    //    - to_down = current low <= lowest low  over ZigZagLen bars
    //    - trend flips -1→1 on to_up, 1→-1 on to_down
    //    - On flip: swing pivot recorded (swing low for bullish flip,
    //      swing high for bearish flip); up to 5 stored per type.
    //
    // 2. Market Structure Break (MSB)
    //    - Bearish MSB : l0 < l1  AND  l0 < l1 - |h0-l1| * FibFactor
    //    - Bullish MSB : h0 > h1  AND  h0 > h1 + |h1-l0| * FibFactor
    //    - Guarded: market doesn't change if l0/h0 are unchanged since last
    //      market flip (prevents re-triggering on the same pivots).
    //
    // 3. On MSB: draw MSB line + label, then draw Order Block (OB) box and
    //    Breaker Block (BB/MB) box.
    //    - Bull MSB:
    //        OB  = last bearish candle between h1i … l0i
    //        BB  = last bearish candle between (l1i - ZigZagLen) … h1i
    //        label = "Bu-BB" if l0 < l1 else "Bu-MB"
    //    - Bear MSB:
    //        OB  = last bullish candle between l1i … h0i
    //        BB  = last bullish candle between (h1i - ZigZagLen) … l1i
    //        label = "Be-BB" if h0 > h1 else "Be-MB"
    //
    // 4. Box management (per bar)
    //    - Bullish OB/BB: extend right while close > bottom; delete oldest if
    //      close < bottom.
    //    - Bearish OB/BB: extend right while close < top; delete oldest if
    //      close > top.
    //    - Only the 5 most recent boxes of each type are kept (FIFO).
    // -------------------------------------------------------------------------

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MarketStructureBreakOrderBlock : Indicator
    {
        // ------------------------------------------------------------------ //
        //  Parameters                                                         //
        // ------------------------------------------------------------------ //

        [Parameter("ZigZag Length", DefaultValue = 9, Group = "Settings")]
        public int ZigZagLen { get; set; }

        [Parameter("Show Zigzag", DefaultValue = true, Group = "Settings")]
        public bool ShowZigzag { get; set; }

        [Parameter("Fib Factor for breakout confirmation", DefaultValue = 0.33, MinValue = 0, MaxValue = 1, Step = 0.01, Group = "Settings")]
        public double FibFactor { get; set; }

        [Parameter("Delete Old/Broken Boxes", DefaultValue = true, Group = "Settings")]
        public bool DeleteBoxes { get; set; }

        // Bu-OB  (color.new(color.green, 70) fill, solid green border/text)
        [Parameter("Bu-OB Color", DefaultValue = "#4D00FF00", Group = "Bu-OB Display Settings")]
        public Color BuObColor { get; set; }

        [Parameter("Bu-OB Border Color", DefaultValue = "#FF00FF00", Group = "Bu-OB Display Settings")]
        public Color BuObBorderColor { get; set; }

        [Parameter("Bu-OB Text Color", DefaultValue = "#FF00FF00", Group = "Bu-OB Display Settings")]
        public Color BuObTextColor { get; set; }

        // Be-OB
        [Parameter("Be-OB Color", DefaultValue = "#4DFF0000", Group = "Be-OB Display Settings")]
        public Color BeObColor { get; set; }

        [Parameter("Be-OB Border Color", DefaultValue = "#FFFF0000", Group = "Be-OB Display Settings")]
        public Color BeObBorderColor { get; set; }

        [Parameter("Be-OB Text Color", DefaultValue = "#FFFF0000", Group = "Be-OB Display Settings")]
        public Color BeObTextColor { get; set; }

        // Bu-BB & Bu-MB
        [Parameter("Bu-BB Color", DefaultValue = "#4D00FF00", Group = "Bu-BB & Bu-MB Display Settings")]
        public Color BuBbColor { get; set; }

        [Parameter("Bu-BB Border Color", DefaultValue = "#FF00FF00", Group = "Bu-BB & Bu-MB Display Settings")]
        public Color BuBbBorderColor { get; set; }

        [Parameter("Bu-BB Text Color", DefaultValue = "#FF00FF00", Group = "Bu-BB & Bu-MB Display Settings")]
        public Color BuBbTextColor { get; set; }

        // Be-BB & Be-MB
        [Parameter("Be-BB Color", DefaultValue = "#4DFF0000", Group = "Be-BB & Be-MB Display Settings")]
        public Color BeBbColor { get; set; }

        [Parameter("Be-BB Border Color", DefaultValue = "#FFFF0000", Group = "Be-BB & Be-MB Display Settings")]
        public Color BeBbBorderColor { get; set; }

        [Parameter("Be-BB Text Color", DefaultValue = "#FFFF0000", Group = "Be-BB & Be-MB Display Settings")]
        public Color BeBbTextColor { get; set; }

        // ------------------------------------------------------------------ //
        //  State                                                              //
        // ------------------------------------------------------------------ //

        private int _trend = 1;
        private int _market = 1;

        // Last bar index where to_up / to_down were true (for rolling min/max)
        private int _lastToUpBar = -1;
        private int _lastToDownBar = -1;

        // Swing-point arrays (oldest → newest, max 5 each)
        private readonly List<double> _highs = new List<double>();
        private readonly List<int>    _highIdx = new List<int>();
        private readonly List<double> _lows  = new List<double>();
        private readonly List<int>    _lowIdx  = new List<int>();

        // Values of l0/h0 when market last changed (guards re-trigger)
        private double _mktL0 = double.NaN;
        private double _mktH0 = double.NaN;

        // Active OB/BB boxes (FIFO, max 5)
        private readonly List<BoxState> _buObBoxes = new List<BoxState>();
        private readonly List<BoxState> _beObBoxes = new List<BoxState>();
        private readonly List<BoxState> _buBbBoxes = new List<BoxState>();
        private readonly List<BoxState> _beBbBoxes = new List<BoxState>();

        private int _id;

        // ------------------------------------------------------------------ //
        //  Indicator lifecycle                                                //
        // ------------------------------------------------------------------ //

        protected override void Initialize() { }

        public override void Calculate(int index)
        {
            if (index < ZigZagLen)
                return;

            // --- rolling highest/lowest over ZigZagLen bars (inclusive) ----
            double highestHigh = double.MinValue, lowestLow = double.MaxValue;
            for (int k = Math.Max(0, index - ZigZagLen + 1); k <= index; k++)
            {
                if (Bars.HighPrices[k] > highestHigh) highestHigh = Bars.HighPrices[k];
                if (Bars.LowPrices[k]  < lowestLow)  lowestLow  = Bars.LowPrices[k];
            }
            bool toUp   = Bars.HighPrices[index] >= highestHigh;
            bool toDown = Bars.LowPrices[index]  <= lowestLow;

            // --- Pine: low_val / high_val (min/max since last to_up/to_down) -
            // last_trend_up_since = ta.barssince(to_up[1]) — uses *previous*
            // bar's to_up, so we scan from (_lastToUpBar + 1) to index.
            double lowVal; int lowBarIdx;
            double highVal; int highBarIdx;
            ComputeLowSince(_lastToUpBar,   index, out lowVal,  out lowBarIdx);
            ComputeHighSince(_lastToDownBar, index, out highVal, out highBarIdx);

            // Update AFTER computing, so current bar's to_up/to_down is
            // excluded from the scan (mirrors Pine's [1] shift)
            if (toUp)   _lastToUpBar   = index;
            if (toDown) _lastToDownBar = index;

            // --- trend update ---------------------------------------------
            int prevTrend = _trend;
            if      (_trend ==  1 && toDown) _trend = -1;
            else if (_trend == -1 && toUp)   _trend =  1;
            bool trendChanged = _trend != prevTrend;

            if (trendChanged)
            {
                if (_trend == 1)  PushLow(lowVal,  lowBarIdx);
                else              PushHigh(highVal, highBarIdx);

                if (ShowZigzag)
                    DrawZigzag();
            }

            // --- market structure -----------------------------------------
            if (_highs.Count >= 2 && _lows.Count >= 2)
                UpdateMarket(index);

            // --- extend / delete active boxes per bar ---------------------
            double close = Bars.ClosePrices[index];
            UpdateBullBoxes(_buObBoxes, close, index);
            UpdateBullBoxes(_buBbBoxes, close, index);
            UpdateBearBoxes(_beObBoxes, close, index);
            UpdateBearBoxes(_beBbBoxes, close, index);
        }

        // ------------------------------------------------------------------ //
        //  ZigZag helpers                                                     //
        // ------------------------------------------------------------------ //

        private void ComputeLowSince(int sinceBar, int index, out double val, out int idx)
        {
            val = double.MaxValue; idx = index;
            int start = sinceBar >= 0 ? sinceBar + 1 : Math.Max(0, index - ZigZagLen);
            for (int k = start; k <= index; k++)
            {
                if (Bars.LowPrices[k] <= val)
                { val = Bars.LowPrices[k]; idx = k; }
            }
        }

        private void ComputeHighSince(int sinceBar, int index, out double val, out int idx)
        {
            val = double.MinValue; idx = index;
            int start = sinceBar >= 0 ? sinceBar + 1 : Math.Max(0, index - ZigZagLen);
            for (int k = start; k <= index; k++)
            {
                if (Bars.HighPrices[k] >= val)
                { val = Bars.HighPrices[k]; idx = k; }
            }
        }

        private void PushHigh(double val, int idx)
        {
            _highs.Add(val); _highIdx.Add(idx);
            if (_highs.Count > 5) { _highs.RemoveAt(0); _highIdx.RemoveAt(0); }
        }

        private void PushLow(double val, int idx)
        {
            _lows.Add(val); _lowIdx.Add(idx);
            if (_lows.Count > 5) { _lows.RemoveAt(0); _lowIdx.RemoveAt(0); }
        }

        // nth most-recent (0 = newest)
        private double GetHigh(int n) => _highs[_highs.Count - 1 - n];
        private int    GetHighIdx(int n) => _highIdx[_highIdx.Count - 1 - n];
        private double GetLow(int n) => _lows[_lows.Count - 1 - n];
        private int    GetLowIdx(int n) => _lowIdx[_lowIdx.Count - 1 - n];

        private void DrawZigzag()
        {
            if (_highs.Count == 0 || _lows.Count == 0) return;
            if (_trend == 1)
                Chart.DrawTrendLine($"ZZ_{_id++}", GetHighIdx(0), GetHigh(0), GetLowIdx(0), GetLow(0), Color.Gray);
            else
                Chart.DrawTrendLine($"ZZ_{_id++}", GetLowIdx(0), GetLow(0), GetHighIdx(0), GetHigh(0), Color.Gray);
        }

        // ------------------------------------------------------------------ //
        //  Market structure                                                   //
        // ------------------------------------------------------------------ //

        private void UpdateMarket(int index)
        {
            double h0 = GetHigh(0); int h0i = GetHighIdx(0);
            double h1 = GetHigh(1); int h1i = GetHighIdx(1);
            double l0 = GetLow(0);  int l0i = GetLowIdx(0);
            double l1 = GetLow(1);  int l1i = GetLowIdx(1);

            int prevMarket = _market;

            // Pine: last_l0 == l0 or last_h0 == h0 → keep market unchanged
            bool sameL0 = !double.IsNaN(_mktL0) && _mktL0 == l0;
            bool sameH0 = !double.IsNaN(_mktH0) && _mktH0 == h0;
            if (!sameL0 && !sameH0)
            {
                if (_market ==  1 && l0 < l1 && l0 < l1 - Math.Abs(h0 - l1) * FibFactor)
                    _market = -1;
                else if (_market == -1 && h0 > h1 && h0 > h1 + Math.Abs(h1 - l0) * FibFactor)
                    _market =  1;
            }

            if (_market == prevMarket) return;

            // Record the pivot values at the time of this market change
            _mktL0 = l0;
            _mktH0 = h0;

            if (_market == 1)
            {
                // Bullish MSB: horizontal line from h1 to l0 at h1 price level
                Chart.DrawTrendLine($"MSB_{_id++}", h1i, h1, l0i, h1, Color.Green, 2);
                var lbl = Chart.DrawText($"MSBL_{_id++}", "MSB", (h1i + l0i) / 2, h1, Color.Green);
                lbl.VerticalAlignment = VerticalAlignment.Top;

                // Bu-OB: last bearish candle between h1i and l0i
                int buObBar = FindLastBearish(h1i, l0i);
                if (buObBar >= 0)
                    CreateBox(_buObBoxes, "BuOB", buObBar, index,
                              BuObColor, BuObBorderColor, "Bu-OB", BuObTextColor);

                // Bu-BB: last bearish candle between (l1i - ZigZagLen) and h1i
                int buBbBar = FindLastBearish(Math.Max(0, l1i - ZigZagLen), h1i);
                if (buBbBar >= 0)
                    CreateBox(_buBbBoxes, "BuBB", buBbBar, index,
                              BuBbColor, BuBbBorderColor, l0 < l1 ? "Bu-BB" : "Bu-MB", BuBbTextColor);
            }
            else
            {
                // Bearish MSB: horizontal line from l1 to h0 at l1 price level
                Chart.DrawTrendLine($"MSB_{_id++}", l1i, l1, h0i, l1, Color.Red, 2);
                var lbl = Chart.DrawText($"MSBL_{_id++}", "MSB", (l1i + h0i) / 2, l1, Color.Red);
                lbl.VerticalAlignment = VerticalAlignment.Bottom;

                // Be-OB: last bullish candle between l1i and h0i
                int beObBar = FindLastBullish(l1i, h0i);
                if (beObBar >= 0)
                    CreateBox(_beObBoxes, "BeOB", beObBar, index,
                              BeObColor, BeObBorderColor, "Be-OB", BeObTextColor);

                // Be-BB: last bullish candle between (h1i - ZigZagLen) and l1i
                int beBbBar = FindLastBullish(Math.Max(0, h1i - ZigZagLen), l1i);
                if (beBbBar >= 0)
                    CreateBox(_beBbBoxes, "BeBB", beBbBar, index,
                              BeBbColor, BeBbBorderColor, h0 > h1 ? "Be-BB" : "Be-MB", BeBbTextColor);
            }
        }

        // ------------------------------------------------------------------ //
        //  Order block / breaker block scan                                   //
        // ------------------------------------------------------------------ //

        // Last bearish candle (open > close) in [fromBar, toBar]
        private int FindLastBearish(int fromBar, int toBar)
        {
            int result = -1;
            for (int k = Math.Max(0, fromBar); k <= Math.Min(Bars.Count - 1, toBar); k++)
                if (Bars.OpenPrices[k] > Bars.ClosePrices[k])
                    result = k;
            return result;
        }

        // Last bullish candle (open < close) in [fromBar, toBar]
        private int FindLastBullish(int fromBar, int toBar)
        {
            int result = -1;
            for (int k = Math.Max(0, fromBar); k <= Math.Min(Bars.Count - 1, toBar); k++)
                if (Bars.OpenPrices[k] < Bars.ClosePrices[k])
                    result = k;
            return result;
        }

        // ------------------------------------------------------------------ //
        //  Box drawing                                                        //
        // ------------------------------------------------------------------ //

        private void CreateBox(List<BoxState> list, string prefix, int barIdx, int currentIndex,
                               Color fillColor, Color borderColor, string text, Color textColor)
        {
            double top    = Bars.HighPrices[barIdx];
            double bottom = Bars.LowPrices[barIdx];
            int    right  = Math.Min(Bars.Count - 1, currentIndex + 10);

            string rectName = $"{prefix}_R_{_id++}";
            string textName = $"{prefix}_T_{_id++}";

            var rect = Chart.DrawRectangle(rectName, barIdx, top, right, bottom, borderColor);
            rect.IsFilled = true;
            rect.Color    = fillColor;

            var t = Chart.DrawText(textName, text, right, top, textColor);
            t.VerticalAlignment   = VerticalAlignment.Bottom;
            t.HorizontalAlignment = HorizontalAlignment.Right;

            list.Add(new BoxState(rectName, textName, rect, t, top, bottom));

            // Cap at 5 boxes
            if (list.Count > 5)
                ShiftBox(list);
        }

        // ------------------------------------------------------------------ //
        //  Box update per bar                                                 //
        // ------------------------------------------------------------------ //

        // Bullish boxes: extend while close > bottom; delete oldest when close < bottom
        private void UpdateBullBoxes(List<BoxState> boxes, double close, int index)
        {
            int right = Math.Min(Bars.Count - 1, index + 10);
            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];
                if (close < box.Bottom)
                {
                    ShiftBox(boxes);
                    break; // Pine shifts only once per iteration (FIFO oldest)
                }
                else
                {
                    ExtendBox(box, right);
                }
            }
        }

        // Bearish boxes: extend while close < top; delete oldest when close > top
        private void UpdateBearBoxes(List<BoxState> boxes, double close, int index)
        {
            int right = Math.Min(Bars.Count - 1, index + 10);
            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];
                if (close > box.Top)
                {
                    ShiftBox(boxes);
                    break;
                }
                else
                {
                    ExtendBox(box, right);
                }
            }
        }

        private void ExtendBox(BoxState box, int rightBarIndex)
        {
            var t = Bars.OpenTimes[rightBarIndex];
            box.Rect.Time2 = t;
            box.Label.Time = t;
        }

        private void ShiftBox(List<BoxState> list)
        {
            if (list.Count == 0) return;
            if (DeleteBoxes)
            {
                Chart.RemoveObject(list[0].RectName);
                Chart.RemoveObject(list[0].TextName);
            }
            list.RemoveAt(0);
        }

        // ------------------------------------------------------------------ //
        //  Inner types                                                        //
        // ------------------------------------------------------------------ //

        private sealed class BoxState
        {
            public readonly string         RectName;
            public readonly string         TextName;
            public readonly ChartRectangle Rect;
            public readonly ChartText      Label;
            public readonly double         Top;
            public readonly double         Bottom;

            public BoxState(string rectName, string textName,
                            ChartRectangle rect, ChartText label,
                            double top, double bottom)
            {
                RectName = rectName; TextName = textName;
                Rect = rect; Label = label;
                Top = top; Bottom = bottom;
            }
        }
    }
}
