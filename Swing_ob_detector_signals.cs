using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    // ════════════════════════════════════════════════════════════════════════════
    //  Swing_OB_Detector  (indicator)
    //
    //  ── SIGNAL TRANSFER — EXACT ICT_01 PATTERN ───────────────────────────────
    //
    //  ICT_01 approach (studied from ICT_01_LongShortSignals.cs):
    //    _isLongSignal  is a boolean computed on EVERY tick.
    //    _isShortSignal is a boolean computed on EVERY tick.
    //    LongSignal[index]  = _isLongSignal  ? 1.0 : 0.0   — ALWAYS written.
    //    ShortSignal[index] = _isShortSignal ? 1.0 : 0.0   — ALWAYS written.
    //    cBot reads: IsSignal(v) = !double.IsNaN(v) && v != 0.0
    //
    //  This file replicates that pattern exactly:
    //    _isLongSignal  ← true on the bar where the UP   arrow is drawn
    //    _isShortSignal ← true on the bar where the DOWN arrow is drawn
    //    LongSignal[index]  = _isLongSignal  ? 1.0 : 0.0   — ALWAYS written
    //    ShortSignal[index] = _isShortSignal ? 1.0 : 0.0   — ALWAYS written
    //
    //  Because _isLongSignal/_isShortSignal are derived from condSwing after
    //  EvaluateSignal runs (with ConfirmedBar enabling same-bar re-evaluation),
    //  the closing tick of bar N always writes the definitive 1.0 or 0.0.
    //  OnBar() reads LongSignal[Bars.Count-2] at bar N+1 and gets 1.0 when
    //  the signal is confirmed, 0.0 when it is not — identical to ICT_01.
    //
    //  LongSwingObBottom and ShortSwingObTop remain NaN-based (written only
    //  when condSwing fires) and carry the OB price level for SL reference.
    //  Being near current price they never compress the Y-axis.
    //
    //  ── OUTPUT SERIES ────────────────────────────────────────────────────────
    //  LongSignal[N]         1.0 = LONG  signal on bar N,  0.0 = no signal.
    //                        cBot detects with: IsSignal(v) = !NaN(v) && v != 0.0
    //  ShortSignal[N]        1.0 = SHORT signal on bar N,  0.0 = no signal.
    //  LongSwingObBottom[N]  ob.Bottom when long  signal fires, NaN otherwise.
    //  ShortSwingObTop[N]    ob.Top    when short signal fires, NaN otherwise.
    // ════════════════════════════════════════════════════════════════════════════

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class Swing_OB_Detector_Signals : Indicator
    {
        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Swing Detection
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "Swing Detection")]
        public int SwingsLengthInput { get; set; }

        [Parameter("BOS/CHoCH Source (Close=0 / HighLow=1)", DefaultValue = 0, MinValue = 0, MaxValue = 1, Group = "Swing Detection")]
        public int StructureSourceInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "Swing Detection")]
        public int ObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation (Close=0 / HighLow=1)", DefaultValue = 1, MinValue = 0, MaxValue = 1, Group = "Swing Detection")]
        public int MitigationModeInput { get; set; }

        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Swing Detection")]
        public int MinDist { get; set; }

        [Parameter("Min Bars After Structure Break", DefaultValue = 0, MinValue = 0, MaxValue = 200, Group = "Swing Detection")]
        public int MinBarsAfterStructureBreak { get; set; }

        [Parameter("Show All Historical OBs", DefaultValue = true, Group = "Swing Detection")]
        public bool ShowAllHistoricalObs { get; set; }

        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Swing Detection")]
        public int SwingOBSize { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Visuals
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Show OB Boxes", DefaultValue = true, Group = "Visuals")]
        public bool ShowObBoxes { get; set; }

        [Parameter("Show Mitigated OBs", DefaultValue = true, Group = "Visuals")]
        public bool ShowMitigatedObs { get; set; }

        [Parameter("Mitigated OB Opacity (%)", DefaultValue = 30, MinValue = 1, MaxValue = 99, Group = "Visuals")]
        public int MitigatedOpacity { get; set; }

        [Parameter("Bull OB Color", DefaultValue = "#CC1848CC", Group = "Visuals")]
        public Color BullObColor { get; set; }

        [Parameter("Bear OB Color", DefaultValue = "#CCB22833", Group = "Visuals")]
        public Color BearObColor { get; set; }

        [Parameter("Show BOS/CHoCH Labels", DefaultValue = true, Group = "Visuals")]
        public bool ShowStructureLabels { get; set; }

        [Parameter("Bull Structure Color", DefaultValue = "#089981", Group = "Visuals")]
        public Color BullStructureColor { get; set; }

        [Parameter("Bear Structure Color", DefaultValue = "#F23645", Group = "Visuals")]
        public Color BearStructureColor { get; set; }

        [Parameter("Show Signal Arrows", DefaultValue = true, Group = "Visuals")]
        public bool ShowSignalArrows { get; set; }

        [Parameter("Signal Offset (pips)", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1, Group = "Visuals")]
        public double SignalOffsetPips { get; set; }

        [Parameter("Line Width Liquidated", DefaultValue = 1, MinValue = 1, MaxValue = 4, Group = "Visuals")]
        public int LineWidthLiquidated { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Output series
        //
        //  LongSignal / ShortSignal — EXACT ICT_01 PATTERN
        //  ────────────────────────────────────────────────
        //  Written UNCONDITIONALLY on every tick:
        //    LongSignal[index]  = _isLongSignal  ? 1.0 : 0.0
        //    ShortSignal[index] = _isShortSignal ? 1.0 : 0.0
        //  PlotType.DiscontinuousLine matches ICT_01's declaration.
        //  cBot detects with: IsSignal(v) = !double.IsNaN(v) && v != 0.0
        //
        //  LongSwingObBottom / ShortSwingObTop — OB PRICE LEVEL
        //  ────────────────────────────────────────────────────
        //  Written only when condSwing fires (NaN otherwise).
        //  Carry the OB reference price for SL/logging — near current price,
        //  so never compress the Y-axis.
        // ════════════════════════════════════════════════════════════════════════

        [Output("Long Signal",  LineColor = "#089981", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries LongSignal { get; set; }

        [Output("Short Signal", LineColor = "#F23645", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries ShortSignal { get; set; }

        [Output("Long Swing OB Bottom", LineColor = "Transparent", PlotType = PlotType.Points, Thickness = 1)]
        public IndicatorDataSeries LongSwingObBottom { get; set; }

        [Output("Short Swing OB Top", LineColor = "Transparent", PlotType = PlotType.Points, Thickness = 1)]
        public IndicatorDataSeries ShortSwingObTop { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Inner types
        // ════════════════════════════════════════════════════════════════════════

        private sealed class OrderBlock
        {
            public int            Index;
            public double         Top;
            public double         Bottom;
            public bool           Bullish;
            public bool           SignalFired;
            public bool           Mitigated;
            public int            StructureBreakIndex;
            public ChartRectangle Box;
            public DateTime       Time;
        }

        // ConfirmedBar replaces the old Entry bool.
        // Allows same-bar re-evaluation on every tick (including closing tick).
        // Blocks re-firing on subsequent bars.
        private sealed class SignalState
        {
            public double Point        = double.NaN;
            public double ObSlLevel    = double.NaN;
            public bool   IsBull;
            public int    ConfirmedBar = -1;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Private fields
        // ════════════════════════════════════════════════════════════════════════

        private readonly List<OrderBlock> _bullObs     = new List<OrderBlock>();
        private readonly List<OrderBlock> _bearObs     = new List<OrderBlock>();
        private readonly List<double>     _parsedHighs = new List<double>();
        private readonly List<double>     _parsedLows  = new List<double>();
        private SignalState               _swingObSignal;

        private int    _swingLeg;
        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;
        private bool   _swingHighCrossed;
        private bool   _swingLowCrossed;
        private int    _swingTrend;

        private int       _obIdCounter;
        private const int MaxOBs = 500;
        private int       _minBarsWarmup;

        // ── ICT_01-style boolean signal state ─────────────────────────────────
        // Set each tick from condSwing — exactly how ICT_01 sets _isLongSignal.
        // _isLongSignal  = true  on the bar where the UP   arrow is drawn.
        // _isShortSignal = true  on the bar where the DOWN arrow is drawn.
        private bool _isLongSignal;
        private bool _isShortSignal;

        // ════════════════════════════════════════════════════════════════════════
        //  Initialize
        // ════════════════════════════════════════════════════════════════════════

        protected override void Initialize()
        {
            _minBarsWarmup = SwingsLengthInput + 5;
            _swingObSignal = NewEmptySignal();
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Calculate — called on every tick
        //
        //  Signal output mirrors ICT_01 exactly:
        //    1. Run detection engine → condSwing
        //    2. _isLongSignal  = (condSwing ==  1)
        //       _isShortSignal = (condSwing == -1)
        //    3. LongSignal[index]  = _isLongSignal  ? 1.0 : 0.0   ← ALWAYS
        //       ShortSignal[index] = _isShortSignal ? 1.0 : 0.0   ← ALWAYS
        //    4. LongSwingObBottom / ShortSwingObTop written only when condSwing fires
        //
        //  On the closing tick of bar N, condSwing reflects the definitive
        //  closed-bar prices (ConfirmedBar allows re-evaluation on same bar).
        //  OnBar() at bar N+1 reads LongSignal[N] = 1.0 → IsSignal() = true ✓
        // ════════════════════════════════════════════════════════════════════════

        public override void Calculate(int index)
        {
            // Fill parsed arrays every bar (needed for OB creation by absolute index).
            UpdateParsedArrays(index);

            // Reset NaN-based OB level series each tick.
            // These are safe to NaN-reset because ConfirmedBar re-evaluates on same bar.
            LongSwingObBottom[index] = double.NaN;
            ShortSwingObTop[index]   = double.NaN;

            // _isLongSignal/_isShortSignal reset to false each tick, exactly like ICT_01.
            _isLongSignal  = false;
            _isShortSignal = false;

            if (index < _minBarsWarmup)
            {
                // Warmup: write 0.0 so these bars are readable (no NaN in flag series).
                LongSignal[index]  = 0.0;
                ShortSignal[index] = 0.0;
                return;
            }

            // ── Swing leg ─────────────────────────────────────────────────────
            int sLen        = Math.Max(5, SwingsLengthInput);
            int swingLegNow = ComputeLeg(index, sLen, _swingLeg);
            int dc          = swingLegNow - _swingLeg;
            if (dc != 0)
            {
                if (dc == 1)
                {
                    _lastSwingLow      = Bars.LowPrices[index - sLen];
                    _lastSwingLowIndex = index - sLen;
                    _swingLowCrossed   = false;
                }
                else
                {
                    _lastSwingHigh      = Bars.HighPrices[index - sLen];
                    _lastSwingHighIndex = index - sLen;
                    _swingHighCrossed   = false;
                }
            }
            _swingLeg = swingLegNow;

            // ── Structure crosses + OB creation ──────────────────────────────
            ProcessSwingStructureCrosses(index);

            // ── OB lifecycle + touch detection ────────────────────────────────
            ManageObList(_bullObs, index, bullish: true,  BullObColor);
            ManageObList(_bearObs, index, bullish: false, BearObColor);

            // ── Signal evaluation ─────────────────────────────────────────────
            double close     = Bars.ClosePrices[index];
            double openPrice = Bars.OpenPrices[index];
            int    candleDir = close > openPrice ? 1 : -1;
            int    condSwing = EvaluateSignal(_swingObSignal, close, candleDir, index);

            // ── Set boolean flags — exactly like ICT_01 ───────────────────────
            // _isLongSignal  = true on the bar where the UP   arrow fires.
            // _isShortSignal = true on the bar where the DOWN arrow fires.
            _isLongSignal  = (condSwing ==  1);
            _isShortSignal = (condSwing == -1);

            // ── Write LongSignal / ShortSignal unconditionally — ICT_01 pattern ─
            // 1.0 = signal confirmed this tick.  0.0 = no signal this tick.
            // The closing tick of bar N writes the definitive value.
            // cBot reads with: IsSignal(v) = !double.IsNaN(v) && v != 0.0
            LongSignal[index]  = _isLongSignal  ? 1.0 : 0.0;
            ShortSignal[index] = _isShortSignal ? 1.0 : 0.0;

            // ── Write OB level series (NaN-based, Y-axis safe) ────────────────
            // These carry the actual OB price for SL reference — near current price.
            if (_isLongSignal)  LongSwingObBottom[index]  = _swingObSignal.ObSlLevel;
            if (_isShortSignal) ShortSwingObTop[index]    = _swingObSignal.ObSlLevel;

            // ── Draw signal arrows on chart ────────────────────────────────────
            // Arrows drawn only when signal fires AND ShowSignalArrows is on.
            // Index-only icon IDs so re-draws on the same bar replace rather than stack.
            if (ShowSignalArrows)
            {
                double offset = SignalOffsetPips * Symbol.PipSize;
                if (_isLongSignal)
                    Chart.DrawIcon($"sig_buy_{index}",
                        ChartIconType.UpArrow,
                        Bars.OpenTimes[index],
                        Bars.LowPrices[index] - offset,
                        BullStructureColor);
                if (_isShortSignal)
                    Chart.DrawIcon($"sig_sell_{index}",
                        ChartIconType.DownArrow,
                        Bars.OpenTimes[index],
                        Bars.HighPrices[index] + offset,
                        BearStructureColor);
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Detection engine — verbatim from Swing_OB_Detector_cBot
        // ════════════════════════════════════════════════════════════════════════

        private void UpdateParsedArrays(int index)
        {
            double atr     = AverageTrueRange(index, ObFilterAtrPeriod);
            bool   highVol = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * atr;
            _parsedHighs.Add(highVol ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( highVol ? Bars.HighPrices[index] : Bars.LowPrices[index]);
        }

        private double AverageTrueRange(int index, int period)
        {
            int    start = Math.Max(1, index - period + 1);
            double sum   = 0.0;
            int    n     = 0;
            for (int i = start; i <= index; i++)
            {
                double pc = Bars.ClosePrices[i - 1];
                sum += Math.Max(Bars.HighPrices[i] - Bars.LowPrices[i],
                       Math.Max(Math.Abs(Bars.HighPrices[i] - pc),
                                Math.Abs(Bars.LowPrices[i]  - pc)));
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;
            double highest = double.MinValue, lowest = double.MaxValue;
            for (int i = Math.Max(0, index - size + 1); i <= index; i++)
            {
                if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lowest)  lowest  = Bars.LowPrices[i];
            }
            if (Bars.HighPrices[index - size] > highest) return 0;
            if (Bars.LowPrices[index  - size] < lowest)  return 1;
            return previousLeg;
        }

        private void ProcessSwingStructureCrosses(int index)
        {
            if (!double.IsNaN(_lastSwingHigh) && !_swingHighCrossed
                && CrossedUp(index, _lastSwingHigh))
            {
                _swingHighCrossed = true;
                bool choch  = _swingTrend < 0;
                _swingTrend = 1;
                if (ShowStructureLabels)
                    DrawStructureLine(index, _lastSwingHigh,
                        choch ? "CHoCH" : "BOS", BullStructureColor);
                StoreObFromPivot(_lastSwingHighIndex, bias: 1, breakIndex: index);
            }

            if (!double.IsNaN(_lastSwingLow) && !_swingLowCrossed
                && CrossedDown(index, _lastSwingLow))
            {
                _swingLowCrossed = true;
                bool choch  = _swingTrend > 0;
                _swingTrend = -1;
                if (ShowStructureLabels)
                    DrawStructureLine(index, _lastSwingLow,
                        choch ? "CHoCH" : "BOS", BearStructureColor);
                StoreObFromPivot(_lastSwingLowIndex, bias: -1, breakIndex: index);
            }
        }

        private bool CrossedUp(int index, double level)
            => StructureSourceInput == 1
               ? Bars.HighPrices[index]  > level
               : Bars.ClosePrices[index] > level;

        private bool CrossedDown(int index, double level)
            => StructureSourceInput == 1
               ? Bars.LowPrices[index]   < level
               : Bars.ClosePrices[index] < level;

        private void StoreObFromPivot(int pivotIndex, int bias, int breakIndex)
        {
            if (pivotIndex < 0 || pivotIndex >= breakIndex
                || breakIndex >= _parsedHighs.Count) return;

            int parsedIndex = pivotIndex;
            if (bias == -1)
            {
                double maxV = double.MinValue;
                for (int i = pivotIndex; i <= breakIndex; i++)
                    if (_parsedHighs[i] > maxV) { maxV = _parsedHighs[i]; parsedIndex = i; }
            }
            else
            {
                double minV = double.MaxValue;
                for (int i = pivotIndex; i <= breakIndex; i++)
                    if (_parsedLows[i] < minV) { minV = _parsedLows[i]; parsedIndex = i; }
            }

            bool bullish = (bias == 1);
            var ob = new OrderBlock
            {
                Index               = parsedIndex,
                Top                 = _parsedHighs[parsedIndex],
                Bottom              = _parsedLows[parsedIndex],
                Bullish             = bullish,
                SignalFired         = false,
                Mitigated           = false,
                StructureBreakIndex = breakIndex,
                Box                 = null,
                Time                = Bars.OpenTimes[parsedIndex]
            };
            _obIdCounter++;

            var list = bullish ? _bullObs : _bearObs;
            if (list.Count >= MaxOBs) list.RemoveAt(list.Count - 1);
            list.Insert(0, ob);
        }

        private void ManageObList(List<OrderBlock> list, int index, bool bullish, Color color)
        {
            for (int i = list.Count - 1; i >= 0; i--)
            {
                OrderBlock ob = list[i];

                if (!ob.SignalFired && ob.Index < index)
                {
                    if (bullish && Bars.LowPrices[index] <= ob.Top)
                    {
                        ob.SignalFired = true;
                        bool sameBar    = (index == ob.StructureBreakIndex)
                                       && (Bars.ClosePrices[index] > Bars.OpenPrices[index]);
                        bool cooldownOk = index > ob.StructureBreakIndex + MinBarsAfterStructureBreak;
                        if (ob.Index + MinDist < index && !sameBar && cooldownOk)
                        {
                            DrawLiquidationLine($"liq_{ob.Index}_{index}",
                                ob.Time, Bars.OpenTimes[index], ob.Top, color);
                            _swingObSignal = new SignalState
                            {
                                Point        = ob.Top,
                                ObSlLevel    = ob.Bottom,
                                IsBull       = true,
                                ConfirmedBar = -1
                            };
                        }
                    }
                    else if (!bullish && Bars.HighPrices[index] >= ob.Bottom)
                    {
                        ob.SignalFired = true;
                        bool sameBar    = (index == ob.StructureBreakIndex)
                                       && (Bars.ClosePrices[index] < Bars.OpenPrices[index]);
                        bool cooldownOk = index > ob.StructureBreakIndex + MinBarsAfterStructureBreak;
                        if (ob.Index + MinDist < index && !sameBar && cooldownOk)
                        {
                            DrawLiquidationLine($"liq_{ob.Index}_{index}",
                                ob.Time, Bars.OpenTimes[index], ob.Bottom, color);
                            _swingObSignal = new SignalState
                            {
                                Point        = ob.Bottom,
                                ObSlLevel    = ob.Top,
                                IsBull       = false,
                                ConfirmedBar = -1
                            };
                        }
                    }
                }

                double bearSrc = MitigationModeInput == 0
                    ? Bars.ClosePrices[index] : Bars.HighPrices[index];
                double bullSrc = MitigationModeInput == 0
                    ? Bars.ClosePrices[index] : Bars.LowPrices[index];
                bool mitigated = (!bullish && bearSrc > ob.Top)
                              || ( bullish && bullSrc < ob.Bottom);

                if (mitigated)
                {
                    ob.Mitigated = true;
                    if (ShowMitigatedObs && ob.Box != null)
                    {
                        int   alpha    = (int)Math.Round(255.0 * MitigatedOpacity / 100.0);
                        Color dimColor = Color.FromArgb(alpha, color.R, color.G, color.B);
                        ob.Box.Time2    = Bars.OpenTimes[index];
                        ob.Box.Color    = dimColor;
                        ob.Box.IsFilled = true;
                        ob.Box          = null;
                    }
                    else if (ob.Box != null)
                    {
                        Chart.RemoveObject(ob.Box.Name);
                        ob.Box = null;
                    }
                    list.RemoveAt(i);
                    continue;
                }

                if (!ShowAllHistoricalObs && i >= SwingOBSize)
                {
                    if (ob.Box != null) { Chart.RemoveObject(ob.Box.Name); ob.Box = null; }
                    list.RemoveAt(i);
                    continue;
                }

                if (ShowObBoxes)
                {
                    int right = Math.Min(index + 1, Bars.Count - 1);
                    if (ob.Box == null)
                    {
                        string id = $"ob_{(bullish ? "b" : "r")}_{ob.Index}_{_obIdCounter}";
                        var rect   = Chart.DrawRectangle(id,
                            ob.Time, ob.Top, Bars.OpenTimes[right], ob.Bottom, color);
                        rect.IsFilled  = true;
                        rect.Color     = color;
                        rect.LineStyle = LineStyle.Solid;
                        ob.Box         = rect;
                    }
                    else
                    {
                        ob.Box.Time1    = ob.Time;
                        ob.Box.Time2    = Bars.OpenTimes[right];
                        ob.Box.Y1       = ob.Top;
                        ob.Box.Y2       = ob.Bottom;
                        ob.Box.Color    = color;
                        ob.Box.IsFilled = true;
                    }
                }
            }
        }

        // ── EvaluateSignal ────────────────────────────────────────────────────
        // ConfirmedBar allows re-evaluation on the SAME bar so the closing tick
        // gets the definitive result.  Blocks re-firing on subsequent bars.
        private int EvaluateSignal(SignalState signal, double close, int candleDir, int index)
        {
            if (double.IsNaN(signal.Point)) return 0;

            // Block only if confirmed on a prior bar.
            if (signal.ConfirmedBar >= 0 && signal.ConfirmedBar < index) return 0;

            if (close > signal.Point && signal.IsBull && candleDir == 1)
            {
                signal.ConfirmedBar = index;
                return 1;
            }
            if (close < signal.Point && !signal.IsBull && candleDir == -1)
            {
                signal.ConfirmedBar = index;
                return -1;
            }
            return 0;
        }

        private static SignalState NewEmptySignal()
            => new SignalState { Point = double.NaN, ObSlLevel = double.NaN, ConfirmedBar = -1 };

        // ════════════════════════════════════════════════════════════════════════
        //  Chart helpers
        // ════════════════════════════════════════════════════════════════════════

        private void DrawLiquidationLine(string id, DateTime from, DateTime to,
                                          double price, Color color)
        {
            var line = Chart.DrawTrendLine(id, from, price, to, price,
                           color, LineWidthLiquidated, LineStyle.LinesDots);
            line.ExtendToInfinity = false;
        }

        private void DrawStructureLine(int index, double level, string label, Color color)
        {
            int x1 = Math.Max(index - 10, 0);
            Chart.DrawTrendLine($"str_{label}_{index}", x1, level, index, level,
                                color, 1, LineStyle.Solid);
            Chart.DrawText($"str_t_{label}_{index}", label, index, level, color);
        }
    }
}
