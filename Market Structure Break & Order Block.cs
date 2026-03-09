// =============================================================================
// Market Structure Break & Order Block
// C# cTrader Indicator
// Pine Script original by EmreKb — ported line-by-line
// =============================================================================
//
// ARCHITECTURE MAP (Pine → C#)
// ─────────────────────────────────────────────────────────────────────────────
//  ta.highest(zigzag_len)  → GetHighest(index, len)
//  ta.lowest(zigzag_len)   → GetLowest(index, len)
//  to_up / to_down         → toUp / toDown  (Pine lines 55-56)
//  trend state machine     → _trend        (Pine lines 57-59)
//  ta.barssince(to_up[1])  → _lastToUpBar  (Pine line 62)
//  ta.barssince(to_down[1])→ _lastToDownBar (Pine line 67)
//  low_val / low_index     → lowVal / lowIndex (Pine lines 62-65)
//  high_val / high_index   → highVal / highIndex (Pine lines 67-70)
//  array push on change    → _highPoints/_lowPoints Lists (Pine lines 72-77)
//  f_get_high / f_get_low  → h0,h0i,h1,h1i,l0,l0i,l1,l1i (Pine lines 80-87)
//  zigzag line.new         → Chart.DrawTrendLine (Pine lines 90-94)
//  market state machine    → _market with last_l0/last_h0 guard (Pine lines 96-110)
//  ta.change(market) block → marketChanged event (Pine lines 112-136)
//  bu_ob/be_ob/bu_bb/be_bb search loops → FindLastCandle() (Pine lines 138-175)
//  box.new                 → CreateBox() (Pine lines 113-135)
//  for bull_ob/bear_ob loops → UpdateBoxes() (Pine lines 137-175)
//  f_delete_box            → ShiftOldest() (Pine helper)
//  alertcondition(MSB)     → Print on market change
//
// TL;DR LOGIC
//  • ZigZag detects swing highs/lows using rolling highest/lowest windows
//  • Trend flips when low breaks the lowest low, or high breaks the highest high
//  • Market Structure Break fires when the new low undercuts the previous low
//    by more than fib_factor * swing range (bearish), or vice versa (bullish)
//  • On MSB: draws a horizontal line at the broken structural level + "MSB" label
//  • On MSB: finds the OB candle (last candle of opposite colour in swing range)
//    and the BB candle (last candle of same colour just before the swing)
//  • Boxes extend right each bar; deleted (oldest) when price fully mitigates
//
// KNOWN LIMITATIONS vs Pine
//  • cTrader ChartRectangle has no separate fill vs border Color property.
//    Two rectangle objects are used: one filled (fill colour), one unfilled
//    (border colour) layered over it.
//  • cTrader DrawText is used for box labels; Pine has native box text.
//  • Pine text.align_right is approximated by placing text at the current bar.
//  • label.style_label_down/up approximated with DrawText below/above price.
//  • l0i[zigzag_len] and h0i[zigzag_len] in OB search loops are simplified
//    to use current l0i/h0i values (functionally equivalent at MSB trigger).
//  • Pine box array initialised with 5 na elements; C# starts with empty list
//    (equivalent once initial na shifts are consumed).
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    public enum MsbTextSize { Tiny, Small, Normal, Large, Huge }

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MarketStructureBreakOrderBlock : Indicator
    {
        // =====================================================================
        // PARAMETERS — Settings
        // =====================================================================

        [Parameter("ZigZag Length", DefaultValue = 9, MinValue = 1, Group = "Settings")]
        public int ZigZagLen { get; set; }

        [Parameter("Show Zigzag", DefaultValue = true, Group = "Settings")]
        public bool ShowZigzag { get; set; }

        [Parameter("Fib Factor for breakout confirmation", DefaultValue = 0.33, MinValue = 0.0, MaxValue = 1.0, Step = 0.01, Group = "Settings")]
        public double FibFactor { get; set; }

        [Parameter("Text Size", DefaultValue = MsbTextSize.Tiny, Group = "Settings")]
        public MsbTextSize TxtSize { get; set; }

        [Parameter("Delete Old/Broken Boxes", DefaultValue = true, Group = "Settings")]
        public bool DeleteBoxes { get; set; }

        // =====================================================================
        // PARAMETERS — Bu-OB Display
        // =====================================================================

        [Parameter("Fill Color", DefaultValue = "Green", Group = "Bu-OB Display Settings")]
        public Color BuObFillColor { get; set; }

        [Parameter("Border Color", DefaultValue = "Green", Group = "Bu-OB Display Settings")]
        public Color BuObBorderColor { get; set; }

        [Parameter("Text Color", DefaultValue = "Green", Group = "Bu-OB Display Settings")]
        public Color BuObTextColor { get; set; }

        // =====================================================================
        // PARAMETERS — Be-OB Display
        // =====================================================================

        [Parameter("Fill Color", DefaultValue = "Red", Group = "Be-OB Display Settings")]
        public Color BeObFillColor { get; set; }

        [Parameter("Border Color", DefaultValue = "Red", Group = "Be-OB Display Settings")]
        public Color BeObBorderColor { get; set; }

        [Parameter("Text Color", DefaultValue = "Red", Group = "Be-OB Display Settings")]
        public Color BeObTextColor { get; set; }

        // =====================================================================
        // PARAMETERS — Bu-BB & Bu-MB Display
        // =====================================================================

        [Parameter("Fill Color", DefaultValue = "Green", Group = "Bu-BB & Bu-MB Display Settings")]
        public Color BuBbFillColor { get; set; }

        [Parameter("Border Color", DefaultValue = "Green", Group = "Bu-BB & Bu-MB Display Settings")]
        public Color BuBbBorderColor { get; set; }

        [Parameter("Text Color", DefaultValue = "Green", Group = "Bu-BB & Bu-MB Display Settings")]
        public Color BuBbTextColor { get; set; }

        // =====================================================================
        // PARAMETERS — Be-BB & Be-MB Display
        // =====================================================================

        [Parameter("Fill Color", DefaultValue = "Red", Group = "Be-BB & Be-MB Display Settings")]
        public Color BeBbFillColor { get; set; }

        [Parameter("Border Color", DefaultValue = "Red", Group = "Be-BB & Be-MB Display Settings")]
        public Color BeBbBorderColor { get; set; }

        [Parameter("Text Color", DefaultValue = "Red", Group = "Be-BB & Be-MB Display Settings")]
        public Color BeBbTextColor { get; set; }

        // =====================================================================
        // INTERNAL STATE
        // =====================================================================

        // Trend and Market state machines
        private int _trend  = 1;
        private int _market = 1;
        private int _prevTrend  = 1;
        private int _prevMarket = 1;

        // Market guard: stores l0 and h0 at the time of the last MSB
        // Pine: last_l0 = ta.valuewhen(ta.change(market)!=0, l0, 0)
        //       last_h0 = ta.valuewhen(ta.change(market)!=0, h0, 0)
        private double _lastMsbL0 = double.NaN;
        private double _lastMsbH0 = double.NaN;

        // ZigZag swing point arrays (oldest at [0], newest at [Count-1])
        private readonly List<double> _highPoints  = new List<double>();
        private readonly List<int>    _highIndices = new List<int>();
        private readonly List<double> _lowPoints   = new List<double>();
        private readonly List<int>    _lowIndices  = new List<int>();

        // ta.barssince trackers
        // _lastToUpBar   : most recent bar index where toUp was true (updates at bar end)
        // _lastToDownBar : most recent bar index where toDown was true
        private int _lastToUpBar   = -1;
        private int _lastToDownBar = -1;

        // Box tracking
        private sealed class BoxInfo
        {
            public ChartRectangle FillRect;    // filled with fill colour (transparent)
            public ChartRectangle BorderRect;  // unfilled, border colour, for the outline
            public ChartText      Label;
            public double         Top;
            public double         Bottom;
            public string         BaseId;
        }

        private readonly List<BoxInfo> _buObBoxes = new List<BoxInfo>();
        private readonly List<BoxInfo> _beObBoxes = new List<BoxInfo>();
        private readonly List<BoxInfo> _buBbBoxes = new List<BoxInfo>();
        private readonly List<BoxInfo> _beBbBoxes = new List<BoxInfo>();

        private int _uid = 0;   // monotonically increasing object ID counter

        // =====================================================================
        // INITIALIZE
        // =====================================================================

        protected override void Initialize() { }

        // =====================================================================
        // CALCULATE — called on every bar
        // =====================================================================

        public override void Calculate(int index)
        {
            if (index < ZigZagLen) return;

            double high  = Bars.HighPrices[index];
            double low   = Bars.LowPrices[index];
            double close = Bars.ClosePrices[index];

            // ---- to_up / to_down (Pine lines 55-56) ----
            // to_up   = high >= ta.highest(zigzag_len)  → current high is the period max
            // to_down = low  <= ta.lowest(zigzag_len)   → current low  is the period min
            double highest = GetHighest(index, ZigZagLen);
            double lowest  = GetLowest(index,  ZigZagLen);
            bool toUp   = high >= highest;
            bool toDown = low  <= lowest;

            // ---- barssince(to_up[1]) and barssince(to_down[1]) (Pine lines 62,67) ----
            // _lastToUpBar is updated at END of bar, so it reflects bars BEFORE current bar.
            // barssince(to_up[1]) = (index-1) - _lastToUpBar   (0 means to_up was true last bar)
            int lastTrendUpSince   = (_lastToUpBar   >= 0) ? (index - 1 - _lastToUpBar)   : index;
            int lastTrendDownSince = (_lastToDownBar >= 0) ? (index - 1 - _lastToDownBar) : index;

            // ---- low_val / low_index (Pine lines 62-65) ----
            // low_val   = ta.lowest(  max(last_trend_up_since,   1) )
            // low_index = bar_index - ta.barssince(low_val == low)
            int    lowWindow = Math.Max(lastTrendUpSince,   1);
            double lowVal    = GetLowest(index, lowWindow);
            int    lowIndex  = FindLastLowMatch(index, lowVal, lowWindow);

            // ---- high_val / high_index (Pine lines 67-70) ----
            int    highWindow = Math.Max(lastTrendDownSince, 1);
            double highVal    = GetHighest(index, highWindow);
            int    highIndex  = FindLastHighMatch(index, highVal, highWindow);

            // ---- Trend state machine (Pine lines 57-59) ----
            _prevTrend = _trend;
            if      (_trend ==  1 && toDown) _trend = -1;
            else if (_trend == -1 && toUp)   _trend =  1;
            bool trendChanged = _trend != _prevTrend;

            // ---- Push to ZigZag arrays on trend change (Pine lines 72-77) ----
            if (trendChanged)
            {
                if (_trend == 1)
                {
                    // New uptrend confirmed → record the swing low that ended the downtrend
                    _lowPoints.Add(lowVal);
                    _lowIndices.Add(lowIndex);
                }
                else
                {
                    // New downtrend confirmed → record the swing high that ended the uptrend
                    _highPoints.Add(highVal);
                    _highIndices.Add(highIndex);
                }
                // Trim to keep last 5 — matches Pine's array of size 5
                if (_highPoints.Count > 5)  { _highPoints.RemoveAt(0);  _highIndices.RemoveAt(0); }
                if (_lowPoints.Count  > 5)  { _lowPoints.RemoveAt(0);   _lowIndices.RemoveAt(0);  }
            }

            // ---- Accessor helpers: h0/h1/l0/l1 (Pine lines 80-87) ----
            // f_get_high(0) = newest high = _highPoints[Count-1]
            // f_get_high(1) = second newest = _highPoints[Count-2]
            double h0  = _highPoints.Count >= 1 ? _highPoints[_highPoints.Count - 1] : double.NaN;
            int    h0i = _highIndices.Count >= 1 ? _highIndices[_highIndices.Count - 1] : -1;
            double h1  = _highPoints.Count >= 2 ? _highPoints[_highPoints.Count - 2] : double.NaN;
            int    h1i = _highIndices.Count >= 2 ? _highIndices[_highIndices.Count - 2] : -1;

            double l0  = _lowPoints.Count >= 1 ? _lowPoints[_lowPoints.Count - 1] : double.NaN;
            int    l0i = _lowIndices.Count >= 1 ? _lowIndices[_lowIndices.Count - 1] : -1;
            double l1  = _lowPoints.Count >= 2 ? _lowPoints[_lowPoints.Count - 2] : double.NaN;
            int    l1i = _lowIndices.Count >= 2 ? _lowIndices[_lowIndices.Count - 2] : -1;

            // ---- ZigZag lines on trend change (Pine lines 90-94) ----
            if (trendChanged && ShowZigzag && h0i >= 0 && l0i >= 0)
            {
                string zzId = $"ZZ_{index}";
                if (_trend == 1)
                    // trend flipped up: draw line from swing high down to swing low
                    Chart.DrawTrendLine(zzId, h0i, h0, l0i, l0, Color.Gray, 1, LineStyle.Solid);
                else
                    // trend flipped down: draw line from swing low up to swing high
                    Chart.DrawTrendLine(zzId, l0i, l0, h0i, h0, Color.Gray, 1, LineStyle.Solid);
            }

            // ---- Market structure break logic (Pine lines 96-110) ----
            // Guard: if l0 or h0 haven't changed since last MSB, don't re-evaluate
            // Pine: last_l0 == l0 or last_h0 == h0 → keep market unchanged
            _prevMarket = _market;

            bool allPivotsAvailable = !double.IsNaN(l0) && !double.IsNaN(l1) &&
                                      !double.IsNaN(h0) && !double.IsNaN(h1);
            bool guardBlocked = (!double.IsNaN(_lastMsbL0) && l0 == _lastMsbL0) ||
                                (!double.IsNaN(_lastMsbH0) && h0 == _lastMsbH0);

            if (allPivotsAvailable && !guardBlocked)
            {
                // Bullish → Bearish MSB:
                //   l0 < l1  AND  l0 < l1 - |h0 - l1| * fib_factor
                if (_market == 1 && l0 < l1 && l0 < l1 - Math.Abs(h0 - l1) * FibFactor)
                    _market = -1;

                // Bearish → Bullish MSB:
                //   h0 > h1  AND  h0 > h1 + |h1 - l0| * fib_factor
                else if (_market == -1 && h0 > h1 && h0 > h1 + Math.Abs(h1 - l0) * FibFactor)
                    _market = 1;
            }

            bool marketChanged = _market != _prevMarket;

            // Update guard trackers on every MSB event
            if (marketChanged)
            {
                _lastMsbL0 = l0;
                _lastMsbH0 = h0;

                // Print MSB alert (mirrors Pine alertcondition)
                Print("MSB | market={0} | Bar={1}", _market == 1 ? "Bullish" : "Bearish", index);
            }

            // ---- Draw MSB visuals + create OB/BB boxes on MSB (Pine lines 112-136) ----
            if (marketChanged && allPivotsAvailable && h1i >= 0 && l1i >= 0)
            {
                if (_market == 1) // Bullish MSB
                    OnBullishMsb(index, h0, h0i, h1, h1i, l0, l0i, l1, l1i);
                else              // Bearish MSB
                    OnBearishMsb(index, h0, h0i, h1, h1i, l0, l0i, l1, l1i);
            }

            // ---- Box maintenance every bar (Pine lines 137-175) ----
            UpdateBuObBoxes(index, close);
            UpdateBeObBoxes(index, close);
            UpdateBeBbBoxes(index, close);
            UpdateBuBbBoxes(index, close);

            // ---- Update barssince trackers (AFTER using them this bar) ----
            if (toUp)   _lastToUpBar   = index;
            if (toDown) _lastToDownBar = index;
        }

        // =====================================================================
        // BULLISH MSB EVENT (Pine lines 113-124)
        // =====================================================================

        private void OnBullishMsb(int index,
            double h0, int h0i, double h1, int h1i,
            double l0, int l0i, double l1, int l1i)
        {
            // Horizontal green line at h1 level from h1i to h0i (Pine line 114)
            string msbLineId = $"MSB_LINE_{index}";
            Chart.DrawTrendLine(msbLineId, h1i, h1, h0i, h1, Color.Green, 2, LineStyle.Solid);

            // "MSB" label at midpoint of h1i..l0i (Pine line 115)
            int    midBar = (h1i + l0i) / 2;
            string lblId  = $"MSB_LBL_{index}";
            var    lbl    = Chart.DrawText(lblId, "MSB", midBar, h1, Color.Green);
            lbl.FontSize  = GetFontSize();

            // Bu-OB: search h1i..l0i for LAST bearish candle (Pine lines 138-142)
            int buObBar = FindLastCandle(h1i, l0i, bearish: true, maxBar: index);
            if (buObBar >= 0)
            {
                CreateBox($"BU_OB_{++_uid}", buObBar,
                    Bars.HighPrices[buObBar], Bars.LowPrices[buObBar],
                    index, "Bu-OB",
                    BuObFillColor, BuObBorderColor, BuObTextColor,
                    _buObBoxes);
            }

            // Bu-BB: search (l1i - ZigZagLen)..h1i for LAST bullish candle (Pine lines 162-167)
            // text = l0 < l1 ? "Bu-BB" : "Bu-MB"
            string buBbText = l0 < l1 ? "Bu-BB" : "Bu-MB";
            int buBbBar = FindLastCandle(l1i - ZigZagLen, h1i, bearish: false, maxBar: index);
            if (buBbBar >= 0)
            {
                CreateBox($"BU_BB_{++_uid}", buBbBar,
                    Bars.HighPrices[buBbBar], Bars.LowPrices[buBbBar],
                    index, buBbText,
                    BuBbFillColor, BuBbBorderColor, BuBbTextColor,
                    _buBbBoxes);
            }
        }

        // =====================================================================
        // BEARISH MSB EVENT (Pine lines 125-136)
        // =====================================================================

        private void OnBearishMsb(int index,
            double h0, int h0i, double h1, int h1i,
            double l0, int l0i, double l1, int l1i)
        {
            // Horizontal red line at l1 level from l1i to l0i (Pine line 126)
            string msbLineId = $"MSB_LINE_{index}";
            Chart.DrawTrendLine(msbLineId, l1i, l1, l0i, l1, Color.Red, 2, LineStyle.Solid);

            // "MSB" label at midpoint of l1i..h0i (Pine line 127)
            int    midBar = (l1i + h0i) / 2;
            string lblId  = $"MSB_LBL_{index}";
            var    lbl    = Chart.DrawText(lblId, "MSB", midBar, l1, Color.Red);
            lbl.FontSize  = GetFontSize();

            // Be-OB: search l1i..h0i for LAST bullish candle (Pine lines 149-154)
            int beObBar = FindLastCandle(l1i, h0i, bearish: false, maxBar: index);
            if (beObBar >= 0)
            {
                CreateBox($"BE_OB_{++_uid}", beObBar,
                    Bars.HighPrices[beObBar], Bars.LowPrices[beObBar],
                    index, "Be-OB",
                    BeObFillColor, BeObBorderColor, BeObTextColor,
                    _beObBoxes);
            }

            // Be-BB: search (h1i - ZigZagLen)..l1i for LAST bearish candle (Pine lines 156-161)
            // text = h0 > h1 ? "Be-BB" : "Be-MB"
            string beBbText = h0 > h1 ? "Be-BB" : "Be-MB";
            int beBbBar = FindLastCandle(h1i - ZigZagLen, l1i, bearish: true, maxBar: index);
            if (beBbBar >= 0)
            {
                CreateBox($"BE_BB_{++_uid}", beBbBar,
                    Bars.HighPrices[beBbBar], Bars.LowPrices[beBbBar],
                    index, beBbText,
                    BeBbFillColor, BeBbBorderColor, BeBbTextColor,
                    _beBbBoxes);
            }
        }

        // =====================================================================
        // BOX MAINTENANCE — Bu-OB (Pine lines 137-143)
        //
        // for bull_ob in bu_ob_boxes:
        //   if close < bottom → f_delete_box  [mitigated]
        //   else if close < top → alert "Price in the BU-OB zone"
        //   else → box.set_right(bar_index + 10)
        // =====================================================================

        private void UpdateBuObBoxes(int index, double close)
        {
            for (int i = _buObBoxes.Count - 1; i >= 0; i--)
            {
                var box = _buObBoxes[i];
                if (close < box.Bottom)
                {
                    // Mitigated — delete oldest (Pine: array.shift)
                    ShiftOldest(_buObBoxes);
                    break; // list modified; exit loop (Pine processes one deletion per bar pass)
                }
                else if (close < box.Top)
                {
                    // Price inside BU-OB zone
                    if (index == Bars.Count - 1)
                        Print("Price in the BU-OB zone | Bar={0}", index);
                    ExtendBoxRight(box, index);
                }
                else
                {
                    ExtendBoxRight(box, index);
                }
            }
        }

        // =====================================================================
        // BOX MAINTENANCE — Be-OB (Pine lines 144-151)
        //
        // NOTE: Pine uses "if" (not "else if") for the second condition:
        //   if close > top   → f_delete_box
        //   if close > bottom → alert  (← NOT else if — intentionally a second separate if)
        //   else              → set_right
        //
        // Effect: when close > top, box is deleted AND alert fires on same bar.
        // =====================================================================

        private void UpdateBeObBoxes(int index, double close)
        {
            for (int i = _beObBoxes.Count - 1; i >= 0; i--)
            {
                var box = _beObBoxes[i];
                if (close > box.Top)
                {
                    // Mitigated — delete oldest
                    ShiftOldest(_beObBoxes);
                    break;
                }
                // Second condition: plain "if", NOT "else if" — mirrors Pine exactly
                if (close > box.Bottom)
                {
                    // Price inside BE-OB zone
                    if (index == Bars.Count - 1)
                        Print("Price in the BE-OB zone | Bar={0}", index);
                    // NOTE: no explicit extend here when entering zone — falls through to no-op
                    // Pine's else only covers the case where NEITHER condition is true
                }
                else
                {
                    ExtendBoxRight(box, index);
                }
            }
        }

        // =====================================================================
        // BOX MAINTENANCE — Be-BB (Pine lines 152-158)
        //
        // for bear_bb in be_bb_boxes:
        //   if close > top   → delete
        //   else if close > bottom → alert
        //   else → set_right
        // =====================================================================

        private void UpdateBeBbBoxes(int index, double close)
        {
            for (int i = _beBbBoxes.Count - 1; i >= 0; i--)
            {
                var box = _beBbBoxes[i];
                if (close > box.Top)
                {
                    ShiftOldest(_beBbBoxes);
                    break;
                }
                else if (close > box.Bottom)
                {
                    if (index == Bars.Count - 1)
                        Print("Price in the BE-BB zone | Bar={0}", index);
                    ExtendBoxRight(box, index);
                }
                else
                {
                    ExtendBoxRight(box, index);
                }
            }
        }

        // =====================================================================
        // BOX MAINTENANCE — Bu-BB (Pine lines 159-165)
        //
        // for bull_bb in bu_bb_boxes:
        //   if close < bottom → delete
        //   else if close < top → alert
        //   else → set_right
        // =====================================================================

        private void UpdateBuBbBoxes(int index, double close)
        {
            for (int i = _buBbBoxes.Count - 1; i >= 0; i--)
            {
                var box = _buBbBoxes[i];
                if (close < box.Bottom)
                {
                    ShiftOldest(_buBbBoxes);
                    break;
                }
                else if (close < box.Top)
                {
                    if (index == Bars.Count - 1)
                        Print("Price in the BU-BB zone | Bar={0}", index);
                    ExtendBoxRight(box, index);
                }
                else
                {
                    ExtendBoxRight(box, index);
                }
            }
        }

        // =====================================================================
        // SHIFT OLDEST BOX (Pine: f_delete_box → array.shift + optional box.delete)
        // =====================================================================

        private void ShiftOldest(List<BoxInfo> boxes)
        {
            if (boxes.Count == 0) return;
            var oldest = boxes[0];
            if (DeleteBoxes)
            {
                // Pine: box.delete(array.shift(box_arr))
                if (oldest.FillRect   != null) Chart.RemoveObject(oldest.FillRect.Name);
                if (oldest.BorderRect != null) Chart.RemoveObject(oldest.BorderRect.Name);
                if (oldest.Label      != null) Chart.RemoveObject(oldest.Label.Name);
            }
            // When delete_boxes=false, Pine still calls array.shift (removes from tracking)
            // but does NOT call box.delete — box remains visible on chart
            boxes.RemoveAt(0);
        }

        // =====================================================================
        // EXTEND BOX RIGHT (Pine: box.set_right(box, bar_index + 10))
        // =====================================================================

        private void ExtendBoxRight(BoxInfo box, int currentIndex)
        {
            DateTime futureTime = GetFutureTime(currentIndex, 10);
            if (box.FillRect   != null) box.FillRect.Time2   = futureTime;
            if (box.BorderRect != null) box.BorderRect.Time2 = futureTime;
            if (box.Label      != null)
            {
                // Reposition label to stay at right edge of box
                box.Label.Time = Bars.OpenTimes[currentIndex];
            }
        }

        // =====================================================================
        // CREATE BOX
        // Pine: box.new(left, top, right=bar_index+10, bottom, bgcolor, border_color,
        //               text, text_color, text_halign=right, text_size)
        //
        // cTrader limitation: ChartRectangle.Color applies to both fill and border.
        // We layer two rectangles:
        //  1. FillRect   — IsFilled=true,  Color=fillColor    (semi-transparent)
        //  2. BorderRect — IsFilled=false, Color=borderColor  (solid outline)
        // =====================================================================

        private void CreateBox(string baseId, int candleBar,
            double top, double bottom, int currentIndex,
            string text,
            Color fillColor, Color borderColor, Color textColor,
            List<BoxInfo> targetList)
        {
            DateTime time1 = Bars.OpenTimes[candleBar];
            DateTime time2 = GetFutureTime(currentIndex, 10);

            // Semi-transparent fill (70% transparency ≈ alpha=77/255, matching Pine color.new(...,70))
            Color semiTransparent = Color.FromArgb(77,
                fillColor.R, fillColor.G, fillColor.B);

            var fillRect = Chart.DrawRectangle(baseId + "_F", time1, top, time2, bottom, semiTransparent);
            fillRect.IsFilled      = true;
            fillRect.IsInteractive = false;

            var borderRect = Chart.DrawRectangle(baseId + "_B", time1, top, time2, bottom, borderColor);
            borderRect.IsFilled      = false;
            borderRect.IsInteractive = false;
            borderRect.Thickness     = 1;

            // Label: positioned at current bar (right edge), vertically centred in box
            double midPrice = (top + bottom) / 2.0;
            var lbl = Chart.DrawText(baseId + "_T", text, currentIndex, midPrice, textColor);
            lbl.FontSize = GetFontSize();

            targetList.Add(new BoxInfo
            {
                FillRect   = fillRect,
                BorderRect = borderRect,
                Label      = lbl,
                Top        = top,
                Bottom     = bottom,
                BaseId     = baseId
            });
        }

        // =====================================================================
        // FIND LAST CANDLE IN BAR-INDEX RANGE
        //
        // Mirrors Pine OB/BB search loops (lines 138-175):
        //   for i = fromBar to toBar        (ascending bar_index iteration)
        //     index = bar_index - i         (bars ago)
        //     if open[index] > close[index] → bearish match → bu_ob_index := bar_index[index]
        //
        // The loop OVERWRITES bu_ob_index on each match → last (newest bar_index) match wins.
        // We iterate ascending so the last write is the highest bar_index.
        //
        // maxBar: safety cap — don't scan bars after this index (current bar)
        // =====================================================================

        private int FindLastCandle(int fromBar, int toBar, bool bearish, int maxBar)
        {
            int start  = Math.Max(0, Math.Min(fromBar, toBar));
            int end    = Math.Min(maxBar, Math.Max(fromBar, toBar));
            int result = -1;

            for (int i = start; i <= end; i++)
            {
                double o = Bars.OpenPrices[i];
                double c = Bars.ClosePrices[i];
                bool match = bearish ? (o > c) : (o < c);
                if (match) result = i; // overwrite → last match wins
            }

            return result;
        }

        // =====================================================================
        // HELPERS
        // =====================================================================

        /// <summary>Highest high over the last <paramref name="length"/> bars including current.</summary>
        private double GetHighest(int index, int length)
        {
            double max = double.MinValue;
            for (int i = index; i >= Math.Max(0, index - length + 1); i--)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return max;
        }

        /// <summary>Lowest low over the last <paramref name="length"/> bars including current.</summary>
        private double GetLowest(int index, int length)
        {
            double min = double.MaxValue;
            for (int i = index; i >= Math.Max(0, index - length + 1); i--)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return min;
        }

        /// <summary>
        /// Most recent bar index (scanning backward from index) where low == value.
        /// Mirrors Pine: bar_index - ta.barssince(low_val == low)
        /// </summary>
        private int FindLastLowMatch(int index, double value, int window)
        {
            for (int i = index; i >= Math.Max(0, index - window + 1); i--)
                if (Bars.LowPrices[i] == value) return i;
            return index; // fallback
        }

        /// <summary>
        /// Most recent bar index where high == value.
        /// Mirrors Pine: bar_index - ta.barssince(high_val == high)
        /// </summary>
        private int FindLastHighMatch(int index, double value, int window)
        {
            for (int i = index; i >= Math.Max(0, index - window + 1); i--)
                if (Bars.HighPrices[i] == value) return i;
            return index;
        }

        /// <summary>Compute the DateTime of currentIndex + barsAhead bars.</summary>
        private DateTime GetFutureTime(int currentIndex, int barsAhead)
        {
            if (currentIndex < 1)
                return Bars.OpenTimes[currentIndex];
            TimeSpan barDuration = Bars.OpenTimes[currentIndex] - Bars.OpenTimes[currentIndex - 1];
            return Bars.OpenTimes[currentIndex] + TimeSpan.FromTicks(barDuration.Ticks * barsAhead);
        }

        /// <summary>Maps MsbTextSize enum to a font size in points.</summary>
        private double GetFontSize()
        {
            switch (TxtSize)
            {
                case MsbTextSize.Small:  return 10;
                case MsbTextSize.Normal: return 12;
                case MsbTextSize.Large:  return 14;
                case MsbTextSize.Huge:   return 18;
                default:                 return 8;    // Tiny
            }
        }
    }
}
