// =============================================================================
// Auto TrendLines [TradingFinder] Support Resistance Signal Alerts
// C# cTrader Indicator — port of Pine Script v6 by TFlab
// =============================================================================
//
// ARCHITECTURE MAP (Pine → C#)
// ─────────────────────────────────────────────────────────────────────────────
//  ZZ()                     → UpdateZigZag(index)
//    ta.pivothigh/low        → DetectPivotHigh / DetectPivotLow (strict rightmost)
//    Major init              → inside UpdateZigZag, fires once at ZZ count == 2
//    Lock0/Lock1 seed        → inside UpdateZigZag
//  SyncAdvArray()            → Pine lines 354-364
//  UpdateMajorMinor(index)   → Pine lines 366-492
//  x_0/y_0/t_0 update        → inline in Calculate(), Pine lines 552-556
//  Pointer()                 → UpdatePointers(), Pine lines 495-512
//  Correction_Checker()      → ProcessTrendLine(), Pine lines 514-548
//  AlertSender               → EmitAlert()
//  plotshape icons           → Chart.DrawIcon()
//
// TL INDEX MAP
//  0 = MjExUp   (MLL)  Major External Up     isUp=true
//  1 = MjExDown (MHH)  Major External Down   isUp=false
//  2 = MjInUp   (MHL)  Major Internal Up     isUp=true
//  3 = MjInDown (MLH)  Major Internal Down   isUp=false
//  4 = MnExUp   (mLL)  Minor External Up     isUp=true
//  5 = MnExDown (mHH)  Minor External Down   isUp=false
//  6 = MnInUp   (mHL)  Minor Internal Up     isUp=true
//  7 = MnInDown (mLH)  Minor Internal Down   isUp=false
//
// ICON PLACEMENT (mirrors Pine plotshape lines 603-625):
//  Break Up   → DownTriangle ABOVE bar HIGH  (price broke below rising support)
//  React Up   → UpTriangle   BELOW bar LOW   (price bounced off rising support)
//  Break Down → UpTriangle   BELOW bar LOW   (price broke above falling resistance)
//  React Down → DownTriangle ABOVE bar HIGH  (price rejected off falling resistance)
//
// KNOWN LIMITATIONS vs Pine:
//  • extend.left : cTrader has no left-extension → no extension applied.
//  • extend.both : cTrader right-extension only → ExtendToInfinity = true.
//  • Line tracks right while valid (Option A): ExtendToInfinity=true while valid,
//    frozen at break bar on PermitSet→false transition.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    // =========================================================================
    // ENUMS for dropdown parameters
    // =========================================================================

    public enum TlStyle      { Solid, Dashed, Dotted }
    public enum TlExtend     { None, Right, Both }
    public enum AlertFreq    { All, OncePerBar, PerBarClose }
    public enum SignalIcon   { Triangle, Arrow }

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class AutoTrendLinesTFlab : Indicator
    {
        // =====================================================================
        // PARAMETERS — Logic
        // =====================================================================

        [Parameter("Pivot Period", DefaultValue = 5, MinValue = 1, Group = "Zig Zag Logic")]
        public int PP { get; set; }

        // =====================================================================
        // PARAMETERS — Alert (bool checkbox → dropdown true/false)
        // =====================================================================

        [Parameter("Alert Name", DefaultValue = "Auto TrendLines Alerts [TradingFinder]", Group = "Alert")]
        public string AlertName { get; set; }

        [Parameter("Message Frequency", DefaultValue = AlertFreq.OncePerBar, Group = "Alert")]
        public AlertFreq Frequency { get; set; }

        [Parameter("Show Alert time by Time Zone", DefaultValue = "UTC", Group = "Alert")]
        public string AlertTimeZone { get; set; }

        [Parameter("Break Major External Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjExUp_B { get; set; }
        [Parameter("React Major External Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjExUp_R { get; set; }

        [Parameter("Break Major External Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjExDown_B { get; set; }
        [Parameter("React Major External Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjExDown_R { get; set; }

        [Parameter("Break Major Internal Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjInUp_B { get; set; }
        [Parameter("React Major Internal Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjInUp_R { get; set; }

        [Parameter("Break Major Internal Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjInDown_B { get; set; }
        [Parameter("React Major Internal Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MjInDown_R { get; set; }

        [Parameter("Break Minor External Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnExUp_B { get; set; }
        [Parameter("React Minor External Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnExUp_R { get; set; }

        [Parameter("Break Minor External Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnExDown_B { get; set; }
        [Parameter("React Minor External Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnExDown_R { get; set; }

        [Parameter("Break Minor Internal Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnInUp_B { get; set; }
        [Parameter("React Minor Internal Up TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnInUp_R { get; set; }

        [Parameter("Break Minor Internal Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnInDown_B { get; set; }
        [Parameter("React Minor Internal Down TrendLine", DefaultValue = true, Group = "Alert")]
        public bool Alert_MnInDown_R { get; set; }

        // =====================================================================
        // PARAMETERS — Signal Icon Style
        // Long  signal = upward  icon placed BELOW bar low
        // Short signal = downward icon placed ABOVE bar high
        // =====================================================================

        [Parameter("Icon Shape", DefaultValue = SignalIcon.Triangle, Group = "Signal Icons")]
        public SignalIcon IconShape { get; set; }

        [Parameter("Long Signal Color", DefaultValue = "Green", Group = "Signal Icons")]
        public Color LongSignalColor { get; set; }

        [Parameter("Short Signal Color", DefaultValue = "Red", Group = "Signal Icons")]
        public Color ShortSignalColor { get; set; }

        [Parameter("Signal Offset (pips)", DefaultValue = 2.0, MinValue = 0.0, Group = "Signal Icons")]
        public double SignalOffsetPips { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major External Up
        // =====================================================================

        [Parameter("Show Major External Up TrendLine", DefaultValue = true, Group = "Major External Up TrendLine")]
        public bool Show_MjExUp { get; set; }
        [Parameter("Delete Previous Major External Up TrendLine", DefaultValue = true, Group = "Major External Up TrendLine")]
        public bool Delete_Pre_MjExUp { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#016b05", Group = "Major External Up TrendLine")]
        public string Color_MjExUp { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Solid, Group = "Major External Up TrendLine")]
        public TlStyle Style_MjExUp { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Major External Up TrendLine")]
        public TlExtend Extend_MjExUp { get; set; }
        [Parameter("Width", DefaultValue = 2, MinValue = 1, Group = "Major External Up TrendLine")]
        public int Width_MjExUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major External Down
        // =====================================================================

        [Parameter("Show Major External Down TrendLine", DefaultValue = true, Group = "Major External Down TrendLine")]
        public bool Show_MjExDown { get; set; }
        [Parameter("Delete Previous Major External Down TrendLine", DefaultValue = true, Group = "Major External Down TrendLine")]
        public bool Delete_Pre_MjExDown { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#aa0202", Group = "Major External Down TrendLine")]
        public string Color_MjExDown { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Solid, Group = "Major External Down TrendLine")]
        public TlStyle Style_MjExDown { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Major External Down TrendLine")]
        public TlExtend Extend_MjExDown { get; set; }
        [Parameter("Width", DefaultValue = 2, MinValue = 1, Group = "Major External Down TrendLine")]
        public int Width_MjExDown { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major Internal Up
        // =====================================================================

        [Parameter("Show Major Internal Up TrendLine", DefaultValue = true, Group = "Major Internal Up TrendLine")]
        public bool Show_MjInUp { get; set; }
        [Parameter("Delete Previous Major Internal Up TrendLine", DefaultValue = true, Group = "Major Internal Up TrendLine")]
        public bool Delete_Pre_MjInUp { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#016b05", Group = "Major Internal Up TrendLine")]
        public string Color_MjInUp { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Solid, Group = "Major Internal Up TrendLine")]
        public TlStyle Style_MjInUp { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Major Internal Up TrendLine")]
        public TlExtend Extend_MjInUp { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Major Internal Up TrendLine")]
        public int Width_MjInUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major Internal Down
        // =====================================================================

        [Parameter("Show Major Internal Down TrendLine", DefaultValue = true, Group = "Major Internal Down TrendLine")]
        public bool Show_MjInDown { get; set; }
        [Parameter("Delete Previous Major Internal Down TrendLine", DefaultValue = true, Group = "Major Internal Down TrendLine")]
        public bool Delete_Pre_MjInDown { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#aa0202", Group = "Major Internal Down TrendLine")]
        public string Color_MjInDown { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Solid, Group = "Major Internal Down TrendLine")]
        public TlStyle Style_MjInDown { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Major Internal Down TrendLine")]
        public TlExtend Extend_MjInDown { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Major Internal Down TrendLine")]
        public int Width_MjInDown { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor External Up
        // =====================================================================

        [Parameter("Show Minor External Up TrendLine", DefaultValue = true, Group = "Minor External Up TrendLine")]
        public bool Show_MnExUp { get; set; }
        [Parameter("Delete Previous Minor External Up TrendLine", DefaultValue = true, Group = "Minor External Up TrendLine")]
        public bool Delete_Pre_MnExUp { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#016b05a6", Group = "Minor External Up TrendLine")]
        public string Color_MnExUp { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Dashed, Group = "Minor External Up TrendLine")]
        public TlStyle Style_MnExUp { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Minor External Up TrendLine")]
        public TlExtend Extend_MnExUp { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor External Up TrendLine")]
        public int Width_MnExUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor External Down
        // =====================================================================

        [Parameter("Show Minor External Down TrendLine", DefaultValue = true, Group = "Minor External Down TrendLine")]
        public bool Show_MnExDown { get; set; }
        [Parameter("Delete Previous Minor External Down TrendLine", DefaultValue = true, Group = "Minor External Down TrendLine")]
        public bool Delete_Pre_MnExDown { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#aa0202a6", Group = "Minor External Down TrendLine")]
        public string Color_MnExDown { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Dashed, Group = "Minor External Down TrendLine")]
        public TlStyle Style_MnExDown { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Minor External Down TrendLine")]
        public TlExtend Extend_MnExDown { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor External Down TrendLine")]
        public int Width_MnExDown { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor Internal Up
        // =====================================================================

        [Parameter("Show Minor Internal Up TrendLine", DefaultValue = true, Group = "Minor Internal Up TrendLine")]
        public bool Show_MnInUp { get; set; }
        [Parameter("Delete Previous Minor Internal Up TrendLine", DefaultValue = true, Group = "Minor Internal Up TrendLine")]
        public bool Delete_Pre_MnInUp { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#016b05a6", Group = "Minor Internal Up TrendLine")]
        public string Color_MnInUp { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Dotted, Group = "Minor Internal Up TrendLine")]
        public TlStyle Style_MnInUp { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Minor Internal Up TrendLine")]
        public TlExtend Extend_MnInUp { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor Internal Up TrendLine")]
        public int Width_MnInUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor Internal Down
        // =====================================================================

        [Parameter("Show Minor Internal Down TrendLine", DefaultValue = true, Group = "Minor Internal Down TrendLine")]
        public bool Show_MnInDown { get; set; }
        [Parameter("Delete Previous Minor Internal Down TrendLine", DefaultValue = true, Group = "Minor Internal Down TrendLine")]
        public bool Delete_Pre_MnInDown { get; set; }
        [Parameter("Color (#RRGGBB or #AARRGGBB)", DefaultValue = "#aa0202a6", Group = "Minor Internal Down TrendLine")]
        public string Color_MnInDown { get; set; }
        [Parameter("Style", DefaultValue = TlStyle.Dotted, Group = "Minor Internal Down TrendLine")]
        public TlStyle Style_MnInDown { get; set; }
        [Parameter("Extend", DefaultValue = TlExtend.None, Group = "Minor Internal Down TrendLine")]
        public TlExtend Extend_MnInDown { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor Internal Down TrendLine")]
        public int Width_MnInDown { get; set; }

        // =====================================================================
        // INTERNAL STATE
        // =====================================================================

        // ZZ arrays — oldest at index 0, newest at Count-1
        private readonly List<string> _zzType  = new List<string>();
        private readonly List<double> _zzValue = new List<double>();
        private readonly List<int>    _zzIndex = new List<int>();

        // Advanced arrays — oldest at index 0, newest at Count-1
        private readonly List<string> _advType  = new List<string>();
        private readonly List<double> _advValue = new List<double>();
        private readonly List<int>    _advIndex = new List<int>();

        // Major level tracking
        private double _majorHighLevel          = double.NaN;
        private double _majorLowLevel           = double.NaN;
        private bool   _majorLevelsInitialized  = false;

        // ADV seeding locks
        private bool _lock0          = true;
        private bool _lock1          = true;
        private bool _skipSyncThisBar = false;

        // Running last confirmed pivot values (ta.valuewhen equivalents)
        private double _lastHighPivotValue = double.NaN;
        private int    _lastHighPivotIndex = -1;
        private double _lastLowPivotValue  = double.NaN;
        private int    _lastLowPivotIndex  = -1;

        // x_0 / y_0 / t_0 — last ADV entry, Pine lines 552-556
        private int    _x0     = 0;
        private double _y0     = 0.0;
        private string _t0     = string.Empty;
        private string _t0Prev = string.Empty;   // t_0[1]

        // Snapshot of ZZ last element for SyncAdvArray change detection
        private double _prevZzLastValue      = double.NaN;
        private char   _prevZzLastTypeSuffix = '\0';

        // TL type names indexed 0-7
        private static readonly string[] TlTypeNames =
            { "MLL", "MHH", "MHL", "MLH", "mLL", "mHH", "mHL", "mLH" };

        // Pointer state — 8 independent rolling windows
        private readonly int[]    _ptrX0 = new int[8];
        private readonly double[] _ptrY0 = new double[8];
        private readonly int[]    _ptrX1 = new int[8];
        private readonly double[] _ptrY1 = new double[8];

        // Per-trendline state
        private sealed class TlState
        {
            public ChartTrendLine ActiveLine     = null;
            public ChartTrendLine PrevLine       = null;
            public bool  PermitSet               = false;  // BUG3 FIX: init false
            public bool  PermitSetPrev           = false;
            public int   LastAnchorX0            = 0;      // X_0[1] equivalent
            public int   LastAlertBreakBar        = -1;
            public int   LastAlertReactBar        = -1;
        }
        private readonly TlState[] _tlStates = new TlState[8];

        // Alert frequency helpers
        private DateTime _lastBarOpenTime             = DateTime.MinValue;
        private readonly bool[] _prevBarBreakSignal   = new bool[8];
        private readonly bool[] _prevBarReactSignal   = new bool[8];

        // =====================================================================
        // INITIALIZE
        // =====================================================================

        protected override void Initialize()
        {
            for (int i = 0; i < 8; i++)
                _tlStates[i] = new TlState();
        }

        // =====================================================================
        // CALCULATE
        // =====================================================================

        public override void Calculate(int index)
        {
            // 1. ZZ state machine + ADV seeding
            UpdateZigZag(index);

            // 2. Sync ADV minor entries (Pine lines 354-364)
            SyncAdvArray();

            // 3. Major/minor promotion (Pine lines 366-492)
            UpdateMajorMinor(index);

            // 4. Update x_0/y_0/t_0 from last ADV entry (Pine lines 552-556)
            if (_advType.Count > 2)
            {
                int last = _advType.Count - 1;
                _x0 = _advIndex[last];
                _y0 = _advValue[last];
                _t0 = _advType[last];
            }

            // 5. Update Pointer rolling windows (Pine lines 495-512)
            UpdatePointers();

            // 6. Correction_Checker + alerts for all 8 TLs
            bool isLive   = (index == Bars.Count - 1);
            bool isNewBar = (_lastBarOpenTime != DateTime.MinValue)
                         && (Bars.OpenTimes[index] != _lastBarOpenTime);

            ProcessAllTrendLines(index, isLive, isNewBar);

            // 7. Save end-of-bar state for next bar's [1] comparisons
            _t0Prev = _t0;
            if (_zzType.Count > 0)
            {
                int n = _zzType.Count - 1;
                _prevZzLastValue      = _zzValue[n];
                _prevZzLastTypeSuffix = _zzType[n][_zzType[n].Length - 1];
            }
            _lastBarOpenTime = Bars.OpenTimes[index];
        }

        // =====================================================================
        // PIVOT DETECTION
        // Mirrors ta_pivot.txt: strict rightmost-max/min rule.
        // Window = [index - 2*PP .. index]. Pivot at index-PP.
        // The LAST bar in the window with the max/min MUST be exactly index-PP.
        // =====================================================================

        private bool DetectPivotHigh(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;

            int pivotBar    = index - PP;
            int windowStart = index - 2 * PP;
            double candidate = Bars.HighPrices[pivotBar];

            // Find global max in window
            double max = double.MinValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];

            if (candidate != max) return false;

            // Last occurrence of max in window must be at pivotBar
            int lastMaxBar = windowStart;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] == max) lastMaxBar = i;

            if (lastMaxBar != pivotBar) return false;

            pivotValue = candidate;
            return true;
        }

        private bool DetectPivotLow(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;

            int pivotBar    = index - PP;
            int windowStart = index - 2 * PP;
            double candidate = Bars.LowPrices[pivotBar];

            double min = double.MaxValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];

            if (candidate != min) return false;

            int lastMinBar = windowStart;
            for (int i = windowStart; i <= index; i++)
                if (Bars.LowPrices[i] == min) lastMinBar = i;

            if (lastMinBar != pivotBar) return false;

            pivotValue = candidate;
            return true;
        }

        // =====================================================================
        // ZZ STATE MACHINE (Pine ZZ() function, lines 114-493)
        // Labels: H/L/HH/LL/HL/LH (no Major prefix here)
        // =====================================================================

        private void UpdateZigZag(int index)
        {
            bool hasHigh = DetectPivotHigh(index, out double highValue);
            bool hasLow  = DetectPivotLow(index,  out double lowValue);
            if (!hasHigh && !hasLow) return;

            int pivotBar   = index - PP;
            double barClose = Bars.ClosePrices[index];

            if (hasHigh) { _lastHighPivotValue = highValue; _lastHighPivotIndex = pivotBar; }
            if (hasLow)  { _lastLowPivotValue  = lowValue;  _lastLowPivotIndex  = pivotBar; }

            // ---- Label helpers (evaluated AFTER any removes, count may have changed) ----
            string LabelHigh(double v)
            {
                int n = _zzType.Count;
                return n > 2 ? (_zzValue[n - 2] < v ? "HH" : "LH") : "H";
            }
            string LabelLow(double v)
            {
                int n = _zzType.Count;
                return n > 2 ? (_zzValue[n - 2] < v ? "HL" : "LL") : "L";
            }

            // ---- Mutators ----
            void RemoveLast()
            {
                int n = _zzType.Count - 1;
                _zzType.RemoveAt(n); _zzValue.RemoveAt(n); _zzIndex.RemoveAt(n);
            }
            void PushHigh(double v, int bar) { _zzType.Add(LabelHigh(v)); _zzValue.Add(v); _zzIndex.Add(bar); }
            void PushLow (double v, int bar) { _zzType.Add(LabelLow(v));  _zzValue.Add(v); _zzIndex.Add(bar); }

            int cnt = _zzType.Count;

            // ==================================================================
            // CASE A: Both high AND low confirm simultaneously (Pine 140-222)
            // ==================================================================
            if (hasHigh && hasLow)
            {
                if (cnt == 0)
                {
                    // Q2: insert High first on empty array
                    _zzType.Add("H"); _zzValue.Add(highValue); _zzIndex.Add(pivotBar);
                }
                else
                {
                    string last    = _zzType[cnt - 1];
                    double lastVal = _zzValue[cnt - 1];

                    if (last == "L" || last == "LL")
                    {
                        if (lowValue < lastVal) { RemoveLast(); PushLow(lowValue, pivotBar); }
                        else                    { PushHigh(highValue, pivotBar); }
                    }
                    else if (last == "H" || last == "HH")
                    {
                        if (highValue > lastVal) { RemoveLast(); PushHigh(highValue, pivotBar); }
                        else                     { PushLow(lowValue, pivotBar); }
                    }
                    else if (last == "LH")
                    {
                        if (highValue < lastVal)
                        {
                            PushLow(lowValue, pivotBar);
                        }
                        else if (highValue > lastVal)
                        {
                            if      (barClose < lastVal) { RemoveLast(); PushHigh(highValue, pivotBar); }
                            else if (barClose > lastVal) { PushLow(lowValue, pivotBar); }
                        }
                    }
                    else if (last == "HL")
                    {
                        if (lowValue > lastVal)
                        {
                            PushHigh(highValue, pivotBar);
                        }
                        else if (lowValue < lastVal)
                        {
                            if      (barClose > lastVal) { RemoveLast(); PushLow(lowValue, pivotBar); }
                            else if (barClose < lastVal) { PushHigh(highValue, pivotBar); }
                        }
                    }
                }
            }
            // ==================================================================
            // CASE B: Only High pivot (Pine 223-256)
            // ==================================================================
            else if (hasHigh)
            {
                cnt = _zzType.Count;
                if (cnt == 0)
                {
                    _zzType.Insert(0, "H"); _zzValue.Insert(0, highValue); _zzIndex.Insert(0, pivotBar);
                }
                else
                {
                    string last    = _zzType[cnt - 1];
                    double lastVal = _zzValue[cnt - 1];

                    if (last == "L" || last == "HL" || last == "LL")
                    {
                        if (highValue > lastVal)
                            PushHigh(highValue, pivotBar);
                        else if (highValue < lastVal)
                        {
                            RemoveLast();
                            if (!double.IsNaN(_lastLowPivotValue) && _lastLowPivotIndex >= 0)
                                PushLow(_lastLowPivotValue, _lastLowPivotIndex);
                        }
                    }
                    else if (last == "H" || last == "HH" || last == "LH")
                    {
                        if (lastVal < highValue) { RemoveLast(); PushHigh(highValue, pivotBar); }
                    }
                }
            }
            // ==================================================================
            // CASE C: Only Low pivot (Pine 257-290)
            // ==================================================================
            else
            {
                cnt = _zzType.Count;
                if (cnt == 0)
                {
                    _zzType.Insert(0, "L"); _zzValue.Insert(0, lowValue); _zzIndex.Insert(0, pivotBar);
                }
                else
                {
                    string last    = _zzType[cnt - 1];
                    double lastVal = _zzValue[cnt - 1];

                    if (last == "H" || last == "HH" || last == "LH")
                    {
                        if (lowValue < lastVal)
                            PushLow(lowValue, pivotBar);
                        else if (lowValue > lastVal)
                        {
                            RemoveLast();
                            if (!double.IsNaN(_lastHighPivotValue) && _lastHighPivotIndex >= 0)
                                PushHigh(_lastHighPivotValue, _lastHighPivotIndex);
                        }
                    }
                    else if (last == "L" || last == "HL" || last == "LL")
                    {
                        if (lastVal > lowValue) { RemoveLast(); PushLow(lowValue, pivotBar); }
                    }
                }
            }

            // ---- Major levels init: fires ONCE when ZZ count first == 2 (Pine 316-332) ----
            if (!_majorLevelsInitialized && _zzType.Count == 2)
            {
                if (_zzType[0] == "H")
                { _majorHighLevel = _zzValue[0]; _majorLowLevel  = _zzValue[1]; }
                else
                { _majorHighLevel = _zzValue[1]; _majorLowLevel  = _zzValue[0]; }
                _majorLevelsInitialized = true;
            }

            // ---- ADV seeding: Lock0 fires when ZZ count first >= 1 (Pine 338-344) ----
            if (_lock0 && _zzType.Count >= 1)
            {
                _advType.Insert(0, "M" + _zzType[0]);
                _advValue.Insert(0, _zzValue[0]);
                _advIndex.Insert(0, _zzIndex[0]);
                _lock0 = false;
            }

            // ---- ADV seeding: Lock1 fires when ZZ count first >= 2 (Pine 346-352) ----
            if (_lock1 && _zzType.Count >= 2)
            {
                _advType.Insert(1, "M" + _zzType[1]);
                _advValue.Insert(1, _zzValue[1]);
                _advIndex.Insert(1, _zzIndex[1]);
                _lock1 = false;
                _skipSyncThisBar = true;  // prevent SyncAdvArray from double-adding
            }
        }

        // =====================================================================
        // ADV SYNC (Pine lines 354-364)
        // When ZZ last-value changed bar-over-bar:
        //   suffix changed → push new 'm' entry
        //   suffix same    → update value/index of last ADV entry (no type change)
        // =====================================================================

        private void SyncAdvArray()
        {
            if (_skipSyncThisBar) { _skipSyncThisBar = false; return; }
            if (_zzType.Count <= 1 || _advType.Count == 0) return;

            int    zzLast        = _zzType.Count - 1;
            double currentZzVal  = _zzValue[zzLast];
            string currentZzType = _zzType[zzLast];
            char   currentSuffix = currentZzType[currentZzType.Length - 1];

            if (double.IsNaN(_prevZzLastValue) || currentZzVal == _prevZzLastValue) return;

            if (currentSuffix != _prevZzLastTypeSuffix)
            {
                // New type suffix → push fresh minor entry
                _advType.Add("m" + currentZzType);
                _advValue.Add(currentZzVal);
                _advIndex.Add(_zzIndex[zzLast]);
            }
            else
            {
                // Same type suffix, value updated (e.g. new lower low) → update last ADV value/index
                int advLast = _advType.Count - 1;
                _advValue[advLast] = currentZzVal;
                _advIndex[advLast] = _zzIndex[zzLast];
            }
        }

        // =====================================================================
        // MAJOR/MINOR PROMOTION (Pine lines 366-492)
        //
        // Four triggers, exactly as in Pine:
        //   A) close > MajorHighLevel  → promote last minor Low  → Major Low
        //   B) lastAdvVal > MajorHighLevel → promote last minor High → Major High
        //   C) close < MajorLowLevel   → promote last minor High → Major High
        //   D) lastAdvVal < MajorLowLevel  → promote last minor Low  → Major Low
        // =====================================================================

        private void UpdateMajorMinor(int index)
        {
            if (!_majorLevelsInitialized || _advType.Count <= 1) return;

            double cls = Bars.ClosePrices[index];

            // Helper: get ZZ type at (Count-1-offset), empty string if out of range
            string ZzType(int offset = 0)
            {
                int n = _zzType.Count - 1 - offset;
                return n >= 0 ? _zzType[n] : string.Empty;
            }

            // ---- A) close > MajorHighLevel (Pine 370-406) ----
            if (cls > _majorHighLevel)
            {
                int    last = _advType.Count - 1;
                string t    = _advType[last];

                if (t == "mL")
                {
                    _advType[last] = "ML";
                    _majorLowLevel = _advValue[last];
                }
                else if (t == "mHL" || t == "mLL")
                {
                    string p = "M" + ZzType();
                    if (p.Length > 1) _advType[last] = p;
                    _majorLowLevel = _advValue[last];
                }
                else if (t == "mLH" || t == "mHH" || t == "MLH" || t == "MHH")
                {
                    if (last >= 1)
                    {
                        string t2 = _advType[last - 1];
                        if (t2 == "mHL" || t2 == "mLL")
                        {
                            string p = "M" + ZzType(1);
                            if (p.Length > 1) _advType[last - 1] = p;
                            _majorLowLevel = _advValue[last - 1];
                        }
                    }
                }
            }

            // ---- B) lastAdvVal > MajorHighLevel (Pine 408-429) ----
            {
                int    last = _advType.Count - 1;
                string t    = _advType[last];

                if (_advValue[last] > _majorHighLevel)
                {
                    if (t == "mH")
                    {
                        _advType[last]  = "MH";
                        _majorHighLevel = _advValue[last];
                    }
                    else if (t == "mLH" || t == "mHH")
                    {
                        string p = "M" + ZzType();
                        if (p.Length > 1) _advType[last] = p;
                        _majorHighLevel = _advValue[last];
                    }
                    else if (t == "mHL" || t == "mLL" || t == "MHL" || t == "MLL")
                    {
                        if (last >= 1)
                        {
                            string t2 = _advType[last - 1];
                            if (t2 == "mLH" || t2 == "mHH")
                            {
                                string p = "M" + ZzType(1);
                                if (p.Length > 1) _advType[last - 1] = p;
                                _majorHighLevel = _advValue[last - 1];
                            }
                        }
                    }
                }
            }

            // ---- C) close < MajorLowLevel (Pine 432-468) ----
            // BUG7 FIX: second-to-last check uses only mLH and mHH (not mHL)
            if (cls < _majorLowLevel)
            {
                int    last = _advType.Count - 1;
                string t    = _advType[last];

                if (t == "mH")
                {
                    _advType[last]  = "MH";
                    _majorHighLevel = _advValue[last];
                }
                else if (t == "mLH" || t == "mHH")
                {
                    string p = "M" + ZzType();
                    if (p.Length > 1) _advType[last] = p;
                    _majorHighLevel = _advValue[last];
                }
                else if (t == "mHL" || t == "mLL" || t == "MHL" || t == "MLL")
                {
                    if (last >= 1)
                    {
                        string t2 = _advType[last - 1];
                        // Pine checks only 'mLH' and 'mHH' for the second-to-last
                        if (t2 == "mLH" || t2 == "mHH")
                        {
                            string p = "M" + ZzType(1);
                            if (p.Length > 1) _advType[last - 1] = p;
                            _majorHighLevel = _advValue[last - 1];
                        }
                    }
                }
            }

            // ---- D) lastAdvVal < MajorLowLevel (Pine 470-492) ----
            {
                int    last = _advType.Count - 1;
                string t    = _advType[last];

                if (_advValue[last] < _majorLowLevel)
                {
                    if (t == "mL")
                    {
                        _advType[last]  = "ML";
                        _majorLowLevel  = _advValue[last];
                    }
                    else if (t == "mHL" || t == "mLL" || t == "MLL")
                    {
                        string p = "M" + ZzType();
                        if (p.Length > 1) _advType[last] = p;
                        _majorLowLevel = _advValue[last];
                    }
                }
            }
        }

        // =====================================================================
        // POINTER UPDATE (Pine lines 495-512)
        // Tracks last 2 occurrences of each ADV type in t_0 stream.
        // Fires only when t_0 changes this bar (t_0 != t_0[1]).
        // =====================================================================

        private void UpdatePointers()
        {
            if (_t0 == _t0Prev) return;

            for (int i = 0; i < 8; i++)
            {
                if (_t0 != TlTypeNames[i]) continue;

                if (_ptrX0[i] == 0)
                {
                    _ptrX0[i] = _x0; _ptrY0[i] = _y0;
                }
                else if (_ptrX1[i] == 0)
                {
                    _ptrX1[i] = _x0; _ptrY1[i] = _y0;
                }
                else
                {
                    _ptrX0[i] = _ptrX1[i]; _ptrY0[i] = _ptrY1[i];
                    _ptrX1[i] = _x0;       _ptrY1[i] = _y0;
                }
            }
        }

        // =====================================================================
        // TRENDLINE DISPATCH — all 8 TLs
        // =====================================================================

        private void ProcessAllTrendLines(int index, bool isLive, bool isNewBar)
        {
            ProcessTrendLine(index, 0, true,
                Show_MjExUp, Delete_Pre_MjExUp,
                ParseColor(Color_MjExUp, "#016b05"), MapStyle(Style_MjExUp), Width_MjExUp,
                isLive, isNewBar,
                Alert_MjExUp_B,   Alert_MjExUp_R,
                "Break Major External Up TrendLine",   "React Major External Up TrendLine");

            ProcessTrendLine(index, 1, false,
                Show_MjExDown, Delete_Pre_MjExDown,
                ParseColor(Color_MjExDown, "#aa0202"), MapStyle(Style_MjExDown), Width_MjExDown,
                isLive, isNewBar,
                Alert_MjExDown_B, Alert_MjExDown_R,
                "Break Major External Down TrendLine", "React Major External Down TrendLine");

            ProcessTrendLine(index, 2, true,
                Show_MjInUp, Delete_Pre_MjInUp,
                ParseColor(Color_MjInUp, "#016b05"), MapStyle(Style_MjInUp), Width_MjInUp,
                isLive, isNewBar,
                Alert_MjInUp_B,   Alert_MjInUp_R,
                "Break Major Internal Up TrendLine",   "React Major Internal Up TrendLine");

            ProcessTrendLine(index, 3, false,
                Show_MjInDown, Delete_Pre_MjInDown,
                ParseColor(Color_MjInDown, "#aa0202"), MapStyle(Style_MjInDown), Width_MjInDown,
                isLive, isNewBar,
                Alert_MjInDown_B, Alert_MjInDown_R,
                "Break Major Internal Down TrendLine", "React Major Internal Down TrendLine");

            ProcessTrendLine(index, 4, true,
                Show_MnExUp, Delete_Pre_MnExUp,
                ParseColor(Color_MnExUp, "#016b05a6"), MapStyle(Style_MnExUp), Width_MnExUp,
                isLive, isNewBar,
                Alert_MnExUp_B,   Alert_MnExUp_R,
                "Break Minor External Up TrendLine",   "React Minor External Up TrendLine");

            ProcessTrendLine(index, 5, false,
                Show_MnExDown, Delete_Pre_MnExDown,
                ParseColor(Color_MnExDown, "#aa0202a6"), MapStyle(Style_MnExDown), Width_MnExDown,
                isLive, isNewBar,
                Alert_MnExDown_B, Alert_MnExDown_R,
                "Break Minor External Down TrendLine", "React Minor External Down TrendLine");

            ProcessTrendLine(index, 6, true,
                Show_MnInUp, Delete_Pre_MnInUp,
                ParseColor(Color_MnInUp, "#016b05a6"), MapStyle(Style_MnInUp), Width_MnInUp,
                isLive, isNewBar,
                Alert_MnInUp_B,   Alert_MnInUp_R,
                "Break Minor Internal Up TrendLine",   "React Minor Internal Up TrendLine");

            ProcessTrendLine(index, 7, false,
                Show_MnInDown, Delete_Pre_MnInDown,
                ParseColor(Color_MnInDown, "#aa0202a6"), MapStyle(Style_MnInDown), Width_MnInDown,
                isLive, isNewBar,
                Alert_MnInDown_B, Alert_MnInDown_R,
                "Break Minor Internal Down TrendLine", "React Minor Internal Down TrendLine");
        }

        // =====================================================================
        // CORRECTION_CHECKER (Pine lines 514-548)
        //
        // BLOCK 1 — fires on anchor change (X_0 != X_0[1]):
        //   • Check slope direction
        //   • Scan ALL closes from x0+1 to current bar (strict > or <)
        //   • If valid:   draw/replace line, PermitSet = true
        //   • If invalid: do NOT touch PermitSet (BUG4 FIX)
        //
        // BLOCK 2 — every bar:
        //   • If PermitSet=true and ActiveLine!=null: check close on correct side
        //     → wrong side: PermitSet=false, freeze line at this bar (BUG6 FIX)
        //   • If PermitSet=true but ActiveLine==null: PermitSet=false (BUG5 FIX)
        //
        // Alerts (live bars only):
        //   Break = PermitSetPrev && !PermitSet
        //   React = close on valid side, but wick crossed line (isUp ? low<lp : high>lp)
        // =====================================================================

        private void ProcessTrendLine(int index, int tlIdx, bool isUp,
            bool show, bool deletePrev, Color color, LineStyle lineStyle, int width,
            bool isLive, bool isNewBar,
            bool alertBreakEnabled, bool alertReactEnabled,
            string breakMsg, string reactMsg)
        {
            TlState state = _tlStates[tlIdx];

            int    x0 = _ptrX0[tlIdx];
            double y0 = _ptrY0[tlIdx];
            int    x1 = _ptrX1[tlIdx];
            double y1 = _ptrY1[tlIdx];

            // Snapshot PermitSet before this bar's logic (= Permit_set[1] in Pine)
            state.PermitSetPrev = state.PermitSet;

            // ---- BLOCK 1: Anchor-change event ----
            if (x0 != 0 && x1 != 0 && x0 != state.LastAnchorX0)
            {
                state.LastAnchorX0 = x0;

                bool correctSlope = isUp ? (y1 > y0) : (y1 < y0);
                bool permit       = false;

                if (correctSlope)
                {
                    // Scan all closes from x0+1 to current bar (Pine: for i=1 to bar_index-X_0)
                    permit = true;
                    for (int barI = x0 + 1; barI <= index; barI++)
                    {
                        double lp  = LinePrice(x0, y0, x1, y1, barI);
                        double cls = Bars.ClosePrices[barI];
                        if (isUp ? cls <= lp : cls >= lp) { permit = false; break; }
                    }
                }

                if (permit)
                {
                    if (show)
                    {
                        if (deletePrev && state.PrevLine != null)
                        {
                            Chart.RemoveObject(state.PrevLine.Name);
                            state.PrevLine = null;
                        }

                        string lineId  = $"ATL_{tlIdx}_{x0}_{x1}";
                        var    newLine = Chart.DrawTrendLine(lineId, x0, y0, x1, y1, color, width, lineStyle);
                        // Option A: extend infinitely while valid
                        newLine.ExtendToInfinity = true;

                        state.PrevLine   = state.ActiveLine;
                        state.ActiveLine = newLine;
                    }
                    state.PermitSet = true;
                }
                // BUG4 FIX: when scan fails, do NOT touch PermitSet.
                // The per-bar check (Block 2) will handle the old line.
            }

            // ---- BLOCK 2: Per-bar validity ----
            if (state.PermitSet)
            {
                // BUG5 FIX: no active line → PermitSet must be false (Pine: close>na=false)
                if (state.ActiveLine == null)
                {
                    state.PermitSet = false;
                }
                else
                {
                    double lp    = LinePrice(x0, y0, x1, y1, index);
                    double cls   = Bars.ClosePrices[index];
                    bool   onSide = isUp ? cls > lp : cls < lp;

                    if (!onSide)
                    {
                        state.PermitSet = false;
                        // BUG6 FIX: freeze line at the break bar (not at the original x1)
                        state.ActiveLine.ExtendToInfinity = false;
                        state.ActiveLine.Time2 = Bars.OpenTimes[index];
                        state.ActiveLine.Y2    = lp;
                    }
                    // If onSide: ExtendToInfinity stays true — line tracks right naturally
                }
            }

            // ---- ALERTS (live bars only) ----
            if (isLive)
            {
                bool alertBreak = state.PermitSetPrev && !state.PermitSet;
                bool alertReact = false;

                if (state.PermitSet && state.ActiveLine != null)
                {
                    double lp  = LinePrice(x0, y0, x1, y1, index);
                    double cls  = Bars.ClosePrices[index];
                    double high = Bars.HighPrices[index];
                    double low  = Bars.LowPrices[index];
                    // Up:   close above line AND low dipped below it
                    // Down: close below line AND high spiked above it
                    alertReact = isUp
                        ? (cls > lp && low  < lp)
                        : (cls < lp && high > lp);
                }

                EmitAlert(index, tlIdx, isUp, alertBreak, alertReact,
                    alertBreakEnabled, alertReactEnabled, breakMsg, reactMsg, isNewBar);
            }
        }

        // =====================================================================
        // ALERT EMISSION (Pine lines 578-625)
        //
        // Icon placement (BUG2 FIX) mirrors Pine plotshape calls:
        //   Break Up   → DownTriangle ABOVE bar HIGH  (red)
        //   React Up   → UpTriangle   BELOW bar LOW   (green)
        //   Break Down → UpTriangle   BELOW bar LOW   (green)
        //   React Down → DownTriangle ABOVE bar HIGH  (red)
        //
        // Frequency modes:
        //   All         → every tick on live bar
        //   OncePerBar  → first tick per bar index
        //   PerBarClose → first tick of next bar, using previous bar's signal
        // =====================================================================

        private void EmitAlert(int index, int tlIdx, bool isUp,
            bool alertBreak, bool alertReact,
            bool breakEnabled, bool reactEnabled,
            string breakMsg, string reactMsg,
            bool isNewBar)
        {
            // ---- Break ----
            if (breakEnabled)
            {
                bool fire = false;
                switch (Frequency)
                {
                    case AlertFreq.All:
                        fire = alertBreak;
                        break;
                    case AlertFreq.OncePerBar:
                        if (alertBreak && _tlStates[tlIdx].LastAlertBreakBar != index)
                        { fire = true; _tlStates[tlIdx].LastAlertBreakBar = index; }
                        break;
                    case AlertFreq.PerBarClose:
                        if (isNewBar && _prevBarBreakSignal[tlIdx]) fire = true;
                        break;
                }
                if (fire)
                {
                    // Break Up = short (downward icon above high); Break Down = long (upward icon below low)
                    if (isUp)
                        DrawSignalIcon($"BRK_{tlIdx}_{index}", index, isLong: false,
                            Bars.HighPrices[index], Bars.LowPrices[index]);
                    else
                        DrawSignalIcon($"BRK_{tlIdx}_{index}", index, isLong: true,
                            Bars.HighPrices[index], Bars.LowPrices[index]);

                    Print("{0} | {1} | Bar={2} | TZ={3}", AlertName, breakMsg, index, AlertTimeZone);
                }
            }

            // ---- React ----
            if (reactEnabled)
            {
                bool fire = false;
                switch (Frequency)
                {
                    case AlertFreq.All:
                        fire = alertReact;
                        break;
                    case AlertFreq.OncePerBar:
                        if (alertReact && _tlStates[tlIdx].LastAlertReactBar != index)
                        { fire = true; _tlStates[tlIdx].LastAlertReactBar = index; }
                        break;
                    case AlertFreq.PerBarClose:
                        if (isNewBar && _prevBarReactSignal[tlIdx]) fire = true;
                        break;
                }
                if (fire)
                {
                    // React Up = long (upward icon below low); React Down = short (downward icon above high)
                    if (isUp)
                        DrawSignalIcon($"RCT_{tlIdx}_{index}", index, isLong: true,
                            Bars.HighPrices[index], Bars.LowPrices[index]);
                    else
                        DrawSignalIcon($"RCT_{tlIdx}_{index}", index, isLong: false,
                            Bars.HighPrices[index], Bars.LowPrices[index]);

                    Print("{0} | {1} | Bar={2} | TZ={3}", AlertName, reactMsg, index, AlertTimeZone);
                }
            }

            // Save for PerBarClose mode
            _prevBarBreakSignal[tlIdx] = alertBreak;
            _prevBarReactSignal[tlIdx] = alertReact;
        }

        // =====================================================================
        // SIGNAL ICON HELPER
        // isLong=true  → upward  icon below bar low  (long  signal)
        // isLong=false → downward icon above bar high (short signal)
        // Shape and colour driven by IconShape / LongSignalColor / ShortSignalColor params.
        // =====================================================================

        private void DrawSignalIcon(string id, int barIndex, bool isLong,
            double barHigh, double barLow)
        {
            Color  c      = isLong ? LongSignalColor : ShortSignalColor;
            double offset = SignalOffsetPips * Symbol.PipSize;
            double price  = isLong ? barLow - offset : barHigh + offset;

            ChartIconType icon;
            if (IconShape == SignalIcon.Arrow)
                icon = isLong ? ChartIconType.UpArrow : ChartIconType.DownArrow;
            else
                icon = isLong ? ChartIconType.UpTriangle : ChartIconType.DownTriangle;

            Chart.DrawIcon(id, icon, barIndex, price, c);
        }

        // =====================================================================
        // LINE PRICE — linear extrapolation through (x0,y0),(x1,y1) at atBar
        // Mirrors Pine line.get_price()
        // =====================================================================

        private static double LinePrice(int x0, double y0, int x1, double y1, int atBar)
        {
            if (x1 == x0) return y0;
            return y0 + (y1 - y0) * (double)(atBar - x0) / (x1 - x0);
        }

        // =====================================================================
        // COLOR PARSING
        // Accepts #RRGGBB (6 digits) or #AARRGGBB (8 digits, Pine format).
        // =====================================================================

        private static Color ParseColor(string hex, string fallback)
        {
            if (string.IsNullOrWhiteSpace(hex)) hex = fallback;
            hex = hex.TrimStart('#').Trim();
            try
            {
                if (hex.Length == 8)
                {
                    int a = Convert.ToInt32(hex.Substring(0, 2), 16);
                    int r = Convert.ToInt32(hex.Substring(2, 2), 16);
                    int g = Convert.ToInt32(hex.Substring(4, 2), 16);
                    int b = Convert.ToInt32(hex.Substring(6, 2), 16);
                    return Color.FromArgb(a, r, g, b);
                }
                if (hex.Length == 6)
                {
                    int r = Convert.ToInt32(hex.Substring(0, 2), 16);
                    int g = Convert.ToInt32(hex.Substring(2, 2), 16);
                    int b = Convert.ToInt32(hex.Substring(4, 2), 16);
                    return Color.FromArgb(255, r, g, b);
                }
            }
            catch { /* fall through */ }

            string fb = fallback.TrimStart('#');
            try
            {
                int r = Convert.ToInt32(fb.Substring(0, 2), 16);
                int g = Convert.ToInt32(fb.Substring(2, 2), 16);
                int b = Convert.ToInt32(fb.Substring(4, 2), 16);
                return Color.FromArgb(255, r, g, b);
            }
            catch { return Color.Gray; }
        }

        // =====================================================================
        // LINE STYLE MAPPING (enum → cTrader LineStyle)
        // =====================================================================

        private static LineStyle MapStyle(TlStyle s)
        {
            switch (s)
            {
                case TlStyle.Dashed: return LineStyle.Lines;
                case TlStyle.Dotted: return LineStyle.Dots;
                default:             return LineStyle.Solid;
            }
        }
    }
}
