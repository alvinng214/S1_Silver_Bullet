// =============================================================================
// Auto TrendLines [TradingFinder] Support Resistance Signal Alerts
// C# cTrader Indicator — faithful port of Pine Script v6 by TFlab
// =============================================================================
//
// ARCHITECTURE MAP (Pine → C#)
// ─────────────────────────────────────────────────────────────────────────────
//  ZZ()                → UpdateZigZag(index)
//    ta.pivothigh/low  → DetectPivotHigh / DetectPivotLow
//    Major init        → inside UpdateZigZag, fires once at ZZ count == 2
//    Lock0/Lock1 seed  → inside UpdateZigZag, fires once per lock
//  ADV sync            → SyncAdvArray(index)   [Pine lines 354-364]
//  Major/Minor promote → UpdateMajorMinor(index) [Pine lines 366-492]
//  x_0/y_0/t_0 update → inline in Calculate()  [Pine lines 552-556]
//  Pointer()           → UpdatePointers()        [Pine lines 495-512]
//  Correction_Checker()→ ProcessTrendLine()      [Pine lines 514-548]
//  AlertSender         → EmitAlert()             [Pine lines 578-600]
//  plotshape icons     → Chart.DrawIcon()        [Pine lines 603-625]
//
// TL INDEX MAP
//  0 = MjExUp   (MLL)   — Major External Up
//  1 = MjExDown (MHH)   — Major External Down
//  2 = MjInUp   (MHL)   — Major Internal Up
//  3 = MjInDown (MLH)   — Major Internal Down
//  4 = MnExUp   (mLL)   — Minor External Up
//  5 = MnExDown (mHH)   — Minor External Down
//  6 = MnInUp   (mHL)   — Minor Internal Up
//  7 = MnInDown (mLH)   — Minor Internal Down
//
// KNOWN LIMITATIONS vs Pine
//  • extend.left  : cTrader has no left-extension; maps to no extension.
//  • extend.both  : cTrader only supports right-extension; maps to right only.
//  • Line dynamic tracking (Pine line.set_xy2 every bar): handled via
//    ExtendToInfinity=true while valid, ExtendToInfinity=false on break.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class AutoTrendLinesTFlab : Indicator
    {
        public enum ToggleOption
        {
            On,
            Off
        }

        public enum AlertFrequencyOption
        {
            All,
            OncePerBar,
            PerBarClose
        }

        public enum TrendLineStyleOption
        {
            Solid,
            Dashed,
            Dotted
        }

        public enum TrendLineExtendOption
        {
            None,
            Right,
            Both,
            Left
        }

        // =====================================================================
        // PARAMETERS — Logic
        // =====================================================================

        [Parameter("Pivot Period", DefaultValue = 5, MinValue = 1, Group = "Zig Zag Logic")]
        public int PP { get; set; }

        // =====================================================================
        // PARAMETERS — Alert
        // =====================================================================

        [Parameter("Alert Name", DefaultValue = "Auto TrendLines Alerts [TradingFinder]", Group = "Alert")]
        public string AlertName { get; set; }

        [Parameter("Message Frequency", DefaultValue = AlertFrequencyOption.OncePerBar, Group = "Alert")]
        public AlertFrequencyOption Frequency { get; set; }

        [Parameter("Show Alert time by Time Zone", DefaultValue = "UTC", Group = "Alert")]
        public string AlertTimeZone { get; set; }

        [Parameter("Break Major External Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjExUp_B { get; set; }
        [Parameter("React Major External Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjExUp_R { get; set; }

        [Parameter("Break Major External Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjExDown_B { get; set; }
        [Parameter("React Major External Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjExDown_R { get; set; }

        [Parameter("Break Major Internal Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjInUp_B { get; set; }
        [Parameter("React Major Internal Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjInUp_R { get; set; }

        [Parameter("Break Major Internal Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjInDown_B { get; set; }
        [Parameter("React Major Internal Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MjInDown_R { get; set; }

        [Parameter("Break Minor External Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnExUp_B { get; set; }
        [Parameter("React Minor External Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnExUp_R { get; set; }

        [Parameter("Break Minor External Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnExDown_B { get; set; }
        [Parameter("React Minor External Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnExDown_R { get; set; }

        [Parameter("Break Minor Internal Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnInUp_B { get; set; }
        [Parameter("React Minor Internal Up TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnInUp_R { get; set; }

        [Parameter("Break Minor Internal Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnInDown_B { get; set; }
        [Parameter("React Minor Internal Down TrendLine Alert", DefaultValue = ToggleOption.On, Group = "Alert")]
        public ToggleOption Alert_MnInDown_R { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major External Up
        // =====================================================================

        [Parameter("Show Major External Up TrendLine", DefaultValue = true, Group = "Major External Up TrendLine")]
        public bool Show_MjExUp { get; set; }
        [Parameter("Delete Previous Major External Up TrendLine", DefaultValue = true, Group = "Major External Up TrendLine")]
        public bool Delete_Pre_MjExUp { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#016b05", Group = "Major External Up TrendLine")]
        public string Color_MjExUp { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Solid, Group = "Major External Up TrendLine")]
        public TrendLineStyleOption Style_MjExUp { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Major External Up TrendLine")]
        public TrendLineExtendOption Extend_MjExUp { get; set; }
        [Parameter("Width", DefaultValue = 2, MinValue = 1, Group = "Major External Up TrendLine")]
        public int Width_MjExUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major External Down
        // =====================================================================

        [Parameter("Show Major External Down TrendLine", DefaultValue = true, Group = "Major External Down TrendLine")]
        public bool Show_MjExDown { get; set; }
        [Parameter("Delete Previous Major External Down TrendLine", DefaultValue = true, Group = "Major External Down TrendLine")]
        public bool Delete_Pre_MjExDown { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#aa0202", Group = "Major External Down TrendLine")]
        public string Color_MjExDown { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Solid, Group = "Major External Down TrendLine")]
        public TrendLineStyleOption Style_MjExDown { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Major External Down TrendLine")]
        public TrendLineExtendOption Extend_MjExDown { get; set; }
        [Parameter("Width", DefaultValue = 2, MinValue = 1, Group = "Major External Down TrendLine")]
        public int Width_MjExDown { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major Internal Up
        // =====================================================================

        [Parameter("Show Major Internal Up TrendLine", DefaultValue = true, Group = "Major Internal Up TrendLine")]
        public bool Show_MjInUp { get; set; }
        [Parameter("Delete Previous Major Internal Up TrendLine", DefaultValue = true, Group = "Major Internal Up TrendLine")]
        public bool Delete_Pre_MjInUp { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#016b05", Group = "Major Internal Up TrendLine")]
        public string Color_MjInUp { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Solid, Group = "Major Internal Up TrendLine")]
        public TrendLineStyleOption Style_MjInUp { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Major Internal Up TrendLine")]
        public TrendLineExtendOption Extend_MjInUp { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Major Internal Up TrendLine")]
        public int Width_MjInUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Major Internal Down
        // =====================================================================

        [Parameter("Show Major Internal Down TrendLine", DefaultValue = true, Group = "Major Internal Down TrendLine")]
        public bool Show_MjInDown { get; set; }
        [Parameter("Delete Previous Major Internal Down TrendLine", DefaultValue = true, Group = "Major Internal Down TrendLine")]
        public bool Delete_Pre_MjInDown { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#aa0202", Group = "Major Internal Down TrendLine")]
        public string Color_MjInDown { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Solid, Group = "Major Internal Down TrendLine")]
        public TrendLineStyleOption Style_MjInDown { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Major Internal Down TrendLine")]
        public TrendLineExtendOption Extend_MjInDown { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Major Internal Down TrendLine")]
        public int Width_MjInDown { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor External Up
        // =====================================================================

        [Parameter("Show Minor External Up TrendLine", DefaultValue = true, Group = "Minor External Up TrendLine")]
        public bool Show_MnExUp { get; set; }
        [Parameter("Delete Previous Minor External Up TrendLine", DefaultValue = true, Group = "Minor External Up TrendLine")]
        public bool Delete_Pre_MnExUp { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#016b05a6", Group = "Minor External Up TrendLine")]
        public string Color_MnExUp { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Dashed, Group = "Minor External Up TrendLine")]
        public TrendLineStyleOption Style_MnExUp { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Minor External Up TrendLine")]
        public TrendLineExtendOption Extend_MnExUp { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor External Up TrendLine")]
        public int Width_MnExUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor External Down
        // =====================================================================

        [Parameter("Show Minor External Down TrendLine", DefaultValue = true, Group = "Minor External Down TrendLine")]
        public bool Show_MnExDown { get; set; }
        [Parameter("Delete Previous Minor External Down TrendLine", DefaultValue = true, Group = "Minor External Down TrendLine")]
        public bool Delete_Pre_MnExDown { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#aa0202a6", Group = "Minor External Down TrendLine")]
        public string Color_MnExDown { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Dashed, Group = "Minor External Down TrendLine")]
        public TrendLineStyleOption Style_MnExDown { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Minor External Down TrendLine")]
        public TrendLineExtendOption Extend_MnExDown { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor External Down TrendLine")]
        public int Width_MnExDown { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor Internal Up
        // =====================================================================

        [Parameter("Show Minor Internal Up TrendLine", DefaultValue = true, Group = "Minor Internal Up TrendLine")]
        public bool Show_MnInUp { get; set; }
        [Parameter("Delete Previous Minor Internal Up TrendLine", DefaultValue = true, Group = "Minor Internal Up TrendLine")]
        public bool Delete_Pre_MnInUp { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#016b05a6", Group = "Minor Internal Up TrendLine")]
        public string Color_MnInUp { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Dotted, Group = "Minor Internal Up TrendLine")]
        public TrendLineStyleOption Style_MnInUp { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Minor Internal Up TrendLine")]
        public TrendLineExtendOption Extend_MnInUp { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor Internal Up TrendLine")]
        public int Width_MnInUp { get; set; }

        // =====================================================================
        // PARAMETERS — Display: Minor Internal Down
        // =====================================================================

        [Parameter("Show Minor Internal Down TrendLine", DefaultValue = true, Group = "Minor Internal Down TrendLine")]
        public bool Show_MnInDown { get; set; }
        [Parameter("Delete Previous Minor Internal Down TrendLine", DefaultValue = true, Group = "Minor Internal Down TrendLine")]
        public bool Delete_Pre_MnInDown { get; set; }
        [Parameter("Color (RRGGBB or AARRGGBB)", DefaultValue = "#aa0202a6", Group = "Minor Internal Down TrendLine")]
        public string Color_MnInDown { get; set; }
        [Parameter("Style", DefaultValue = TrendLineStyleOption.Dotted, Group = "Minor Internal Down TrendLine")]
        public TrendLineStyleOption Style_MnInDown { get; set; }
        [Parameter("Extend", DefaultValue = TrendLineExtendOption.None, Group = "Minor Internal Down TrendLine")]
        public TrendLineExtendOption Extend_MnInDown { get; set; }
        [Parameter("Width", DefaultValue = 1, MinValue = 1, Group = "Minor Internal Down TrendLine")]
        public int Width_MnInDown { get; set; }

        // =====================================================================
        // INTERNAL STATE
        // =====================================================================

        // ZZ arrays — Pine: ArrayType, ArrayValue, ArrayIndex (oldest at [0])
        private readonly List<string> _zzType  = new List<string>();
        private readonly List<double> _zzValue = new List<double>();
        private readonly List<int>    _zzIndex = new List<int>();

        // Advanced arrays — Pine: ArrayTypeAdv, ArrayValueAdv, ArrayIndexAdv
        private readonly List<string> _advType  = new List<string>();
        private readonly List<double> _advValue = new List<double>();
        private readonly List<int>    _advIndex = new List<int>();

        // Major level tracking — Pine: var float Major_HighLevel, Major_LowLevel
        private double _majorHighLevel        = double.NaN;
        private double _majorLowLevel         = double.NaN;
        private bool   _majorLevelsInitialized = false;

        // ADV seeding locks — Pine: var bool Lock0 = true, Lock1 = true
        private bool _lock0 = true;
        private bool _lock1 = true;
        // Guard: skip SyncAdvArray on the same bar Lock1 fires to avoid duplicate
        private bool _skipSyncThisBar = false;

        // Last confirmed pivot values (needed for cross-use branches in ZZ)
        // These mirror ta.valuewhen(bool(HighPivot), High[PP], 0) etc.
        private double _lastHighPivotValue = double.NaN;
        private int    _lastHighPivotIndex = -1;
        private double _lastLowPivotValue  = double.NaN;
        private int    _lastLowPivotIndex  = -1;

        // x_0, y_0, t_0 — Pine: var int x_0 = 0; var float y_0 = 0.0; var string t_0 = ''
        private int    _x0     = 0;
        private double _y0     = 0.0;
        private string _t0     = string.Empty;
        private string _t0Prev = string.Empty;  // t_0[1] equivalent

        // Previous ZZ last-element snapshot (for SyncAdvArray change detection)
        private double _prevZzLastValue      = double.NaN;
        private char   _prevZzLastTypeSuffix = '\0';

        // TL type names indexed 0-7, matching Pointer and Correction_Checker order
        private static readonly string[] TlTypeNames =
            { "MLL", "MHH", "MHL", "MLH", "mLL", "mHH", "mHL", "mLH" };

        // Pointer state — Pine: var X_0=0, Y_0=0.0, X_1=0, Y_1=0.0 per call instance
        private readonly int[]    _ptrX0 = new int[8];
        private readonly double[] _ptrY0 = new double[8];
        private readonly int[]    _ptrX1 = new int[8];
        private readonly double[] _ptrY1 = new double[8];

        // Correction_Checker state per TL — Pine: var line/bool per function instance
        private sealed class TlState
        {
            public ChartTrendLine ActiveLine = null;   // Line_Origin
            public ChartTrendLine PrevLine   = null;   // Line_Origin[1] for Delete_Pre
            public bool PermitSet     = true;          // Pine: var bool Permit_set = true
            public bool PermitSetPrev = true;          // Permit_set[1]
            public int  LastX0        = 0;             // X_0[1] — detects anchor change
            public int  LastAlertBreakBar = -1;
            public int  LastAlertReactBar = -1;
        }
        private readonly TlState[] _tlStates = new TlState[8];

        // Alert frequency helpers
        private DateTime _lastBarOpenTime           = DateTime.MinValue;
        private readonly bool[] _prevBarBreakSignal = new bool[8];
        private readonly bool[] _prevBarReactSignal = new bool[8];

        // =====================================================================
        // INITIALIZE
        // =====================================================================

        protected override void Initialize()
        {
            for (int i = 0; i < 8; i++)
                _tlStates[i] = new TlState();
        }

        // =====================================================================
        // CALCULATE — main per-bar entry point
        // =====================================================================

        public override void Calculate(int index)
        {
            // Step 1: Run ZZ state machine (pivot detection + array updates)
            // Also handles MajorLevel init and Lock seeding of ADV
            UpdateZigZag(index);

            // Step 2: Sync ADV array with ZZ last-element changes (minor 'm' entries)
            // Pine lines 354-364
            SyncAdvArray();

            // Step 3: Promote minor 'm' → Major 'M' when price confirms structure break
            // Pine lines 366-492
            UpdateMajorMinor(index);

            // Step 4: Update x_0/y_0/t_0 from last ADV entry (Pine lines 552-556)
            if (_advType.Count > 2)
            {
                int last = _advType.Count - 1;
                _x0 = _advIndex[last];
                _y0 = _advValue[last];
                _t0 = _advType[last];
            }

            // Step 5: Update Pointer rolling windows (Pine: Pointer() calls, lines 558-565)
            UpdatePointers();

            // Step 6: Correction_Checker + alert for all 8 trendlines
            bool isNewBar = _lastBarOpenTime != DateTime.MinValue
                         && Bars.OpenTimes[index] != _lastBarOpenTime;

            ProcessAllTrendLines(index, isNewBar);

            // Step 7: Save end-of-bar state for next bar's [1] comparisons
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
        // Pivot confirmed at bar (index - PP); full window = [index-2*PP .. index].
        // Ties to the RIGHT of the candidate bar disqualify the pivot.
        // =====================================================================

        private bool DetectPivotHigh(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;

            int    pivotBar   = index - PP;
            double candidate  = Bars.HighPrices[pivotBar];
            int    windowStart = index - 2 * PP;

            // Find max in window
            double max = double.MinValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];

            if (candidate != max) return false;

            // Rightmost occurrence of max must be at pivotBar (strict: no tie to the right)
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

            int    pivotBar    = index - PP;
            double candidate   = Bars.LowPrices[pivotBar];
            int    windowStart = index - 2 * PP;

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
        // ZZ STATE MACHINE (Pine: ZZ() function, lines 114-493)
        //
        // Maintains _zzType / _zzValue / _zzIndex with swing labels:
        //   H / L      — first high/low (no predecessor to compare)
        //   HH / LL    — higher high / lower low
        //   HL / LH    — higher low / lower high
        //
        // Also seeds ADV array (Lock0/Lock1) and initialises Major levels.
        // =====================================================================

        private void UpdateZigZag(int index)
        {
            bool hasHigh = DetectPivotHigh(index, out double highValue);
            bool hasLow  = DetectPivotLow(index,  out double lowValue);

            if (!hasHigh && !hasLow) return;

            int pivotBar = index - PP;

            // Update running last-pivot trackers (mirrors ta.valuewhen)
            if (hasHigh) { _lastHighPivotValue = highValue; _lastHighPivotIndex = pivotBar; }
            if (hasLow)  { _lastLowPivotValue  = lowValue;  _lastLowPivotIndex  = pivotBar; }

            double barClose = Bars.ClosePrices[index];

            // ------------------------------------------------------------------
            // Inline label helpers — evaluated at call time AFTER any removes
            // so _zzType.Count reflects the post-remove size.
            // Pine: "ArrayType.size() > 2 ? ArrayValue.get(size-2) < Value ? 'HL'/'HH' : 'LL'/'LH' : 'L'/'H'"
            // ------------------------------------------------------------------
            string LabelLow(double newLow)
            {
                int n = _zzType.Count;
                return n > 2 ? (_zzValue[n - 2] < newLow ? "HL" : "LL") : "L";
            }
            string LabelHigh(double newHigh)
            {
                int n = _zzType.Count;
                return n > 2 ? (_zzValue[n - 2] < newHigh ? "HH" : "LH") : "H";
            }

            // Mutators
            void RemoveLast()
            {
                int n = _zzType.Count - 1;
                _zzType.RemoveAt(n); _zzValue.RemoveAt(n); _zzIndex.RemoveAt(n);
            }
            void PushLow(double v, int barIdx)
            {
                _zzType.Add(LabelLow(v)); _zzValue.Add(v); _zzIndex.Add(barIdx);
            }
            void PushHigh(double v, int barIdx)
            {
                _zzType.Add(LabelHigh(v)); _zzValue.Add(v); _zzIndex.Add(barIdx);
            }

            int cnt = _zzType.Count;

            // ==================================================================
            // CASE A: Both high AND low pivot confirm on same bar
            // Pine lines 140-222
            // ==================================================================
            if (hasHigh && hasLow)
            {
                if (cnt == 0)
                {
                    // Q2 decision: insert High first when both fire on empty array
                    _zzType.Add("H"); _zzValue.Add(highValue); _zzIndex.Add(pivotBar);
                }
                else
                {
                    string last    = _zzType[cnt - 1];
                    double lastVal = _zzValue[cnt - 1];

                    if (last == "L" || last == "LL")
                    {
                        // Pine lines 145-160
                        if (lowValue < lastVal)
                        {
                            // Replace last Low with deeper Low
                            RemoveLast();
                            PushLow(lowValue, pivotBar);
                        }
                        else
                        {
                            // Last Low is lower → continue with a High
                            PushHigh(highValue, pivotBar);
                        }
                    }
                    else if (last == "H" || last == "HH")
                    {
                        // Pine lines 161-176
                        if (highValue > lastVal)
                        {
                            // Replace last High with higher High
                            RemoveLast();
                            PushHigh(highValue, pivotBar);
                        }
                        else
                        {
                            PushLow(lowValue, pivotBar);
                        }
                    }
                    else if (last == "LH")
                    {
                        // Pine lines 177-199
                        if (highValue < lastVal)
                        {
                            PushLow(lowValue, pivotBar);
                        }
                        else if (highValue > lastVal)
                        {
                            if (barClose < lastVal)
                            {
                                RemoveLast();
                                PushHigh(highValue, pivotBar);
                            }
                            else if (barClose > lastVal)
                            {
                                PushLow(lowValue, pivotBar);
                            }
                            // barClose == lastVal: no action (Pine has no else branch here)
                        }
                    }
                    else if (last == "HL")
                    {
                        // Pine lines 200-222
                        if (lowValue > lastVal)
                        {
                            PushHigh(highValue, pivotBar);
                        }
                        else if (lowValue < lastVal)
                        {
                            if (barClose > lastVal)
                            {
                                RemoveLast();
                                PushLow(lowValue, pivotBar);
                            }
                            else if (barClose < lastVal)
                            {
                                PushHigh(highValue, pivotBar);
                            }
                        }
                    }
                    // No branch for last == "HL" with lowValue == lastVal (Pine has no branch)
                }
            }
            // ==================================================================
            // CASE B: Only High pivot
            // Pine lines 223-256
            // ==================================================================
            else if (hasHigh)
            {
                cnt = _zzType.Count;
                if (cnt == 0)
                {
                    // Pine line 225: array.insert(0, 'H', HighValue, HighIndex)
                    _zzType.Insert(0, "H"); _zzValue.Insert(0, highValue); _zzIndex.Insert(0, pivotBar);
                }
                else
                {
                    string last    = _zzType[cnt - 1];
                    double lastVal = _zzValue[cnt - 1];

                    if (last == "L" || last == "HL" || last == "LL")
                    {
                        // Pine lines 231-246
                        if (highValue > lastVal)
                        {
                            PushHigh(highValue, pivotBar);
                        }
                        else if (highValue < lastVal)
                        {
                            // Cross-use: remove stale Low, push a new Low using
                            // last confirmed LowPivot value (ta.valuewhen equivalent)
                            RemoveLast();
                            if (!double.IsNaN(_lastLowPivotValue) && _lastLowPivotIndex >= 0)
                                PushLow(_lastLowPivotValue, _lastLowPivotIndex);
                        }
                        // highValue == lastVal: no action
                    }
                    else if (last == "H" || last == "HH" || last == "LH")
                    {
                        // Pine lines 247-256
                        if (lastVal < highValue)
                        {
                            RemoveLast();
                            PushHigh(highValue, pivotBar);
                        }
                        // else: existing High is higher, no action
                    }
                }
            }
            // ==================================================================
            // CASE C: Only Low pivot
            // Pine lines 257-290
            // ==================================================================
            else // hasLow only
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
                        // Pine lines 265-280
                        if (lowValue < lastVal)
                        {
                            PushLow(lowValue, pivotBar);
                        }
                        else if (lowValue > lastVal)
                        {
                            // Cross-use: remove stale High, push new High using
                            // last confirmed HighPivot value
                            RemoveLast();
                            if (!double.IsNaN(_lastHighPivotValue) && _lastHighPivotIndex >= 0)
                                PushHigh(_lastHighPivotValue, _lastHighPivotIndex);
                        }
                    }
                    else if (last == "L" || last == "HL" || last == "LL")
                    {
                        // Pine lines 281-290
                        if (lastVal > lowValue)
                        {
                            RemoveLast();
                            PushLow(lowValue, pivotBar);
                        }
                        // lastVal <= lowValue: no action
                    }
                }
            }

            // ------------------------------------------------------------------
            // Major levels initialisation — fires ONCE when ZZ count becomes 2
            // Pine lines 316-332
            // ------------------------------------------------------------------
            if (!_majorLevelsInitialized && _zzType.Count == 2)
            {
                if (_zzType[0] == "H")
                {
                    _majorHighLevel = _zzValue[0];
                    _majorLowLevel  = _zzValue[1];
                }
                else
                {
                    _majorHighLevel = _zzValue[1];
                    _majorLowLevel  = _zzValue[0];
                }
                _majorLevelsInitialized = true;
            }

            // ------------------------------------------------------------------
            // ADV seeding — Lock0: fires once when ZZ first reaches count == 1
            // Pine lines 338-344
            // ------------------------------------------------------------------
            if (_lock0 && _zzType.Count >= 1)
            {
                _advType.Insert(0, "M" + _zzType[0]);
                _advValue.Insert(0, _zzValue[0]);
                _advIndex.Insert(0, _zzIndex[0]);
                _lock0 = false;
            }

            // ------------------------------------------------------------------
            // ADV seeding — Lock1: fires once when ZZ first reaches count == 2
            // Pine lines 346-352
            // ------------------------------------------------------------------
            if (_lock1 && _zzType.Count >= 2)
            {
                // Insert at position 1 (after the Lock0 element)
                _advType.Insert(1, "M" + _zzType[1]);
                _advValue.Insert(1, _zzValue[1]);
                _advIndex.Insert(1, _zzIndex[1]);
                _lock1 = false;
                // Guard: prevent SyncAdvArray from double-adding on this same bar
                _skipSyncThisBar = true;
            }
        }

        // =====================================================================
        // ADV SYNC (Pine lines 354-364)
        //
        // When the ZZ last-element VALUE changes bar-over-bar AND ZZ count > 1:
        //   • If the type SUFFIX changed (e.g. last was 'H', now 'L') → push new 'm' entry
        //   • If the type SUFFIX is the same (e.g. 'HH' updated to higher 'HH') → update value/index
        //
        // Pine uses array history ([1]) to detect value changes. We snapshot at
        // end of each Calculate into _prevZzLastValue / _prevZzLastTypeSuffix.
        // =====================================================================

        private void SyncAdvArray()
        {
            // Guard against Lock1 seeding collision
            if (_skipSyncThisBar)
            {
                _skipSyncThisBar = false;
                return;
            }

            if (_zzType.Count <= 1) return;
            if (_advType.Count == 0) return;

            int    zzLast         = _zzType.Count - 1;
            double currentZzVal   = _zzValue[zzLast];
            string currentZzType  = _zzType[zzLast];
            char   currentSuffix  = currentZzType[currentZzType.Length - 1];

            // Fire only when ZZ last value changed since previous bar
            if (double.IsNaN(_prevZzLastValue) || currentZzVal == _prevZzLastValue)
                return;

            if (currentSuffix != _prevZzLastTypeSuffix)
            {
                // New pivot type suffix → push a fresh minor entry
                _advType.Add("m" + currentZzType);
                _advValue.Add(currentZzVal);
                _advIndex.Add(_zzIndex[zzLast]);
            }
            else
            {
                // Same type suffix, value improved → update the LAST ADV entry's value/index.
                // Type string is NOT changed (stays as whatever it was promoted to).
                int advLast = _advType.Count - 1;
                _advValue[advLast] = currentZzVal;
                _advIndex[advLast] = _zzIndex[zzLast];
            }
        }

        // =====================================================================
        // MAJOR / MINOR PROMOTION (Pine lines 366-492)
        //
        // Runs every bar to promote minor ('m') ADV entries to Major ('M') when
        // price breaks the opposing structure level, confirming significance.
        //
        // The Pine code has four main triggers:
        //   A) close > MajorHighLevel → last minor Low becomes Major Low
        //   B) lastAdvValue > MajorHighLevel → last minor High becomes Major High
        //   C) close < MajorLowLevel → last minor High becomes Major High
        //   D) lastAdvValue < MajorLowLevel → last minor Low becomes Major Low
        // =====================================================================

        private void UpdateMajorMinor(int index)
        {
            if (!_majorLevelsInitialized) return;
            if (_advType.Count <= 1) return;

            double cls = Bars.ClosePrices[index];

            // Helper: get last ZZ type, guarded
            string ZzLastType(int offset = 0)
            {
                int n = _zzType.Count - 1 - offset;
                return n >= 0 ? _zzType[n] : string.Empty;
            }

            // ---- A) close > MajorHighLevel (Pine lines 370-406) ----
            if (cls > _majorHighLevel)
            {
                int last = _advType.Count - 1;
                string t = _advType[last];

                if (t == "mL")
                {
                    _advType[last]  = "ML";
                    _majorLowLevel  = _advValue[last];
                }
                else if (t == "mHL" || t == "mLL")
                {
                    string promoted = "M" + ZzLastType();
                    if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                        _advType[last] = promoted;
                    _majorLowLevel = _advValue[last];
                }
                else if (t == "mLH" || t == "mHH" || t == "MLH" || t == "MHH")
                {
                    // Check second-to-last (Pine: ArrayTypeAdv.get(size-2))
                    if (last >= 1)
                    {
                        string t2 = _advType[last - 1];
                        if (t2 == "mHL" || t2 == "mLL")
                        {
                            string promoted = "M" + ZzLastType(1);
                            if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                                _advType[last - 1] = promoted;
                            _majorLowLevel = _advValue[last - 1];
                        }
                    }
                }
            }

            // ---- B) lastAdvValue > MajorHighLevel (Pine lines 408-429) ----
            {
                int last = _advType.Count - 1;
                if (_advValue[last] > _majorHighLevel)
                {
                    string t = _advType[last];
                    if (t == "mH")
                    {
                        _advType[last]   = "MH";
                        _majorHighLevel  = _advValue[last];
                    }
                    else if (t == "mLH" || t == "mHH")
                    {
                        string promoted = "M" + ZzLastType();
                        if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                            _advType[last] = promoted;
                        _majorHighLevel = _advValue[last];
                    }
                    else if (t == "mHL" || t == "mLL" || t == "MHL" || t == "MLL")
                    {
                        if (last >= 1)
                        {
                            string t2 = _advType[last - 1];
                            if (t2 == "mLH" || t2 == "mHH")
                            {
                                string promoted = "M" + ZzLastType(1);
                                if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                                    _advType[last - 1] = promoted;
                                _majorHighLevel = _advValue[last - 1];
                            }
                        }
                    }
                }
            }

            // ---- C) close < MajorLowLevel (Pine lines 432-468) ----
            if (cls < _majorLowLevel)
            {
                int last = _advType.Count - 1;
                string t = _advType[last];

                if (t == "mH")
                {
                    _advType[last]   = "MH";
                    _majorHighLevel  = _advValue[last];
                }
                else if (t == "mLH" || t == "mHH")
                {
                    string promoted = "M" + ZzLastType();
                    if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                        _advType[last] = promoted;
                    _majorHighLevel = _advValue[last];
                }
                else if (t == "mHL" || t == "mLL" || t == "MHL" || t == "MLL")
                {
                    if (last >= 1)
                    {
                        string t2 = _advType[last - 1];
                        if (t2 == "mLH" || t2 == "mHH")
                        {
                            string promoted = "M" + ZzLastType(1);
                            if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                                _advType[last - 1] = promoted;
                            _majorHighLevel = _advValue[last - 1];
                        }
                        else if (t2 == "mHL")
                        {
                            string promoted = "M" + ZzLastType(1);
                            if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                                _advType[last - 1] = promoted;
                            _majorHighLevel = _advValue[last - 1];
                        }
                    }
                }
            }

            // ---- D) lastAdvValue < MajorLowLevel (Pine lines 470-492) ----
            {
                int last = _advType.Count - 1;
                if (_advValue[last] < _majorLowLevel)
                {
                    string t = _advType[last];
                    if (t == "mL")
                    {
                        _advType[last]  = "ML";
                        _majorLowLevel  = _advValue[last];
                    }
                    else if (t == "mHL" || t == "mLL" || t == "MLL")
                    {
                        string promoted = "M" + ZzLastType();
                        if (!string.IsNullOrEmpty(promoted) && promoted.Length > 1)
                            _advType[last] = promoted;
                        _majorLowLevel = _advValue[last];
                    }
                }
            }
        }

        // =====================================================================
        // POINTER UPDATE (Pine: Pointer() function, lines 495-512)
        //
        // Tracks the last 2 occurrences of each labeled ADV type.
        // Fires only when _t0 changes (t_0 != t_0[1]).
        // Rolling window: (X0,Y0) = older, (X1,Y1) = more recent.
        // =====================================================================

        private void UpdatePointers()
        {
            if (_t0 == _t0Prev) return;  // no change this bar

            for (int i = 0; i < 8; i++)
            {
                if (_t0 != TlTypeNames[i]) continue;

                if (_ptrX0[i] == 0)
                {
                    // First occurrence
                    _ptrX0[i] = _x0;
                    _ptrY0[i] = _y0;
                }
                else if (_ptrX1[i] == 0)
                {
                    // Second occurrence
                    _ptrX1[i] = _x0;
                    _ptrY1[i] = _y0;
                }
                else
                {
                    // Third+ occurrence: roll the window
                    _ptrX0[i] = _ptrX1[i];
                    _ptrY0[i] = _ptrY1[i];
                    _ptrX1[i] = _x0;
                    _ptrY1[i] = _y0;
                }
            }
        }

        // =====================================================================
        // TRENDLINE DISPATCH — calls ProcessTrendLine for all 8 TLs
        // Pine lines 567-575 (Correction_Checker calls)
        // =====================================================================

        private void ProcessAllTrendLines(int index, bool isNewBar)
        {
            // MjExUp  → MLL anchors
            ProcessTrendLine(index, 0, true,
                Show_MjExUp,   Delete_Pre_MjExUp,
                ParseColor(Color_MjExUp,   "#016b05"),   ParseStyle(Style_MjExUp), ParseExtend(Extend_MjExUp),   Width_MjExUp,
                isNewBar,
                Alert_MjExUp_B,   Alert_MjExUp_R,
                "Break Major External Up TrendLine",   "React Major External Up TrendLine");

            // MjExDown → MHH anchors
            ProcessTrendLine(index, 1, false,
                Show_MjExDown, Delete_Pre_MjExDown,
                ParseColor(Color_MjExDown, "#aa0202"),   ParseStyle(Style_MjExDown), ParseExtend(Extend_MjExDown), Width_MjExDown,
                isNewBar,
                Alert_MjExDown_B, Alert_MjExDown_R,
                "Break Major External Down TrendLine", "React Major External Down TrendLine");

            // MjInUp  → MHL anchors
            ProcessTrendLine(index, 2, true,
                Show_MjInUp,   Delete_Pre_MjInUp,
                ParseColor(Color_MjInUp,   "#016b05"),   ParseStyle(Style_MjInUp), ParseExtend(Extend_MjInUp),   Width_MjInUp,
                isNewBar,
                Alert_MjInUp_B,   Alert_MjInUp_R,
                "Break Major Internal Up TrendLine",   "React Major Internal Up TrendLine");

            // MjInDown → MLH anchors
            ProcessTrendLine(index, 3, false,
                Show_MjInDown, Delete_Pre_MjInDown,
                ParseColor(Color_MjInDown, "#aa0202"),   ParseStyle(Style_MjInDown), ParseExtend(Extend_MjInDown), Width_MjInDown,
                isNewBar,
                Alert_MjInDown_B, Alert_MjInDown_R,
                "Break Major Internal Down TrendLine", "React Major Internal Down TrendLine");

            // MnExUp  → mLL anchors
            ProcessTrendLine(index, 4, true,
                Show_MnExUp,   Delete_Pre_MnExUp,
                ParseColor(Color_MnExUp,   "#016b05a6"), ParseStyle(Style_MnExUp), ParseExtend(Extend_MnExUp),   Width_MnExUp,
                isNewBar,
                Alert_MnExUp_B,   Alert_MnExUp_R,
                "Break Minor External Up TrendLine",   "React Minor External Up TrendLine");

            // MnExDown → mHH anchors
            ProcessTrendLine(index, 5, false,
                Show_MnExDown, Delete_Pre_MnExDown,
                ParseColor(Color_MnExDown, "#aa0202a6"), ParseStyle(Style_MnExDown), ParseExtend(Extend_MnExDown), Width_MnExDown,
                isNewBar,
                Alert_MnExDown_B, Alert_MnExDown_R,
                "Break Minor External Down TrendLine", "React Minor External Down TrendLine");

            // MnInUp  → mHL anchors
            ProcessTrendLine(index, 6, true,
                Show_MnInUp,   Delete_Pre_MnInUp,
                ParseColor(Color_MnInUp,   "#016b05a6"), ParseStyle(Style_MnInUp), ParseExtend(Extend_MnInUp),   Width_MnInUp,
                isNewBar,
                Alert_MnInUp_B,   Alert_MnInUp_R,
                "Break Minor Internal Up TrendLine",   "React Minor Internal Up TrendLine");

            // MnInDown → mLH anchors
            ProcessTrendLine(index, 7, false,
                Show_MnInDown, Delete_Pre_MnInDown,
                ParseColor(Color_MnInDown, "#aa0202a6"), ParseStyle(Style_MnInDown), ParseExtend(Extend_MnInDown), Width_MnInDown,
                isNewBar,
                Alert_MnInDown_B, Alert_MnInDown_R,
                "Break Minor Internal Down TrendLine", "React Minor Internal Down TrendLine");
        }

        // =====================================================================
        // CORRECTION_CHECKER (Pine lines 514-548)
        //
        // Block 1 — fires only when X0 changes (new anchor pair):
        //   • Check slope direction
        //   • Scan all closes from X0+1 to current bar (strict: close must stay
        //     on the correct side of the extrapolated line)
        //   • If valid: draw/replace line, set PermitSet = true
        //   • If invalid: PermitSet = false, no line drawn
        //
        // Block 2 — runs every bar:
        //   • Check if current close is still on the valid side
        //   • If yes: line extends to infinity (Option A)
        //   • If no:  PermitSet = false, freeze line by disabling ExtendToInfinity
        //
        // Alerts:
        //   • Break = PermitSet was true last bar, false this bar (barstate.isconfirmed)
        //   • React = close on valid side, but wick crossed the line (confirmed bar)
        // =====================================================================

        private void ProcessTrendLine(int index, int tlIdx, bool isUp,
            bool show, bool deletePrev, Color color, LineStyle lineStyle, TrendLineExtendOption extendMode, int width,
            bool isNewBar,
            ToggleOption alertBreakSetting, ToggleOption alertReactSetting,
            string breakMsg, string reactMsg)
        {
            TlState state = _tlStates[tlIdx];

            int    x0 = _ptrX0[tlIdx];
            double y0 = _ptrY0[tlIdx];
            int    x1 = _ptrX1[tlIdx];
            double y1 = _ptrY1[tlIdx];

            // Snapshot PermitSet before this bar's logic (for break detection = Permit_set[1])
            state.PermitSetPrev = state.PermitSet;

            // ---- BLOCK 1: Anchor-change event (X_0 != X_0[1]) ----
            if (x0 != 0 && x1 != 0 && x0 != state.LastX0)
            {
                state.LastX0 = x0;

                // Slope direction check (Pine line 522)
                // Up TL: Y_1 > Y_0 (newer anchor is higher)  → rising from lows
                // Down TL: Y_1 < Y_0 (newer anchor is lower) → falling from highs
                bool correctSlope = isUp ? (y1 > y0) : (y1 < y0);

                bool permit = false;

                if (correctSlope)
                {
                    // Validity scan: all closes from x0+1 to current bar (Pine lines 524-526)
                    // close[bar_index - x0 - i] maps to absolute bar (x0 + i)
                    // For Up: close > linePrice (strictly above)
                    // For Down: close < linePrice (strictly below)
                    permit = true;
                    for (int barI = x0 + 1; barI <= index; barI++)
                    {
                        double lineP = LinePrice(x0, y0, x1, y1, barI);
                        double barC  = Bars.ClosePrices[barI];
                        if (isUp ? barC <= lineP : barC >= lineP)
                        {
                            permit = false;
                            break;
                        }
                    }
                }

                if (permit)
                {
                    if (show)
                    {
                        // Delete previous line if requested (Pine lines 531-532)
                        if (deletePrev && state.PrevLine != null)
                        {
                            Chart.RemoveObject(state.PrevLine.Name);
                            state.PrevLine = null;
                        }

                        // Draw new line (Pine line 529)
                        string lineId   = $"ATL_{tlIdx}_{x0}_{x1}";
                        var    newLine  = Chart.DrawTrendLine(lineId, x0, y0, x1, y1, color, width, lineStyle);
                        // Option A: extend to infinity while valid (mirrors Pine set_xy2 per bar)
                        ApplyLineExtension(newLine, x0, y0, x1, y1, index, extendMode, true);

                        state.PrevLine  = state.ActiveLine;
                        state.ActiveLine = newLine;
                    }

                    state.PermitSet = true;  // Pine line 533: Permit_set := true
                }
                else
                {
                    state.PermitSet = false;
                    // Freeze any existing active line
                    if (state.ActiveLine != null)
                        ApplyLineExtension(state.ActiveLine, x0, y0, x1, y1, index, extendMode, false);
                }
            }

            // ---- BLOCK 2: Per-bar validity check (Pine lines 538-541) ----
            // Pine semantics:
            // if close is still on the valid side and Permit_set => extend line
            // else Permit_set := false
            bool validThisBar = false;
            if (state.PermitSet && state.ActiveLine != null)
            {
                double lp  = LinePrice(x0, y0, x1, y1, index);
                double cls = Bars.ClosePrices[index];
                validThisBar = isUp ? cls > lp : cls < lp;

                if (validThisBar)
                    ApplyLineExtension(state.ActiveLine, x0, y0, x1, y1, index, extendMode, true);
            }

            if (!validThisBar)
            {
                state.PermitSet = false;
                if (state.ActiveLine != null)
                    ApplyLineExtension(state.ActiveLine, x0, y0, x1, y1, index, extendMode, false);  // freeze line at break bar
            }

            // ---- ALERT LOGIC (Pine lines 543-548) ----
            // Pine barstate.isconfirmed is true on historical bars and on live bar close.
            // In cTrader indicator backfill, each historical bar is processed once, so
            // compute and plot alerts on every Calculate call.
            bool alertBreak = state.PermitSetPrev && !state.PermitSet;

            bool alertReact = false;
            if (state.PermitSet && state.ActiveLine != null)
            {
                double lp   = LinePrice(x0, y0, x1, y1, index);
                double cls  = Bars.ClosePrices[index];
                double high = Bars.HighPrices[index];
                double low  = Bars.LowPrices[index];
                // Up:   close above line AND low dipped below it
                // Down: close below line AND high spiked above it
                alertReact = isUp
                    ? (cls > lp && low < lp)
                    : (cls < lp && high > lp);
            }

            EmitAlert(index, tlIdx, alertBreak, alertReact,
                alertBreakSetting, alertReactSetting, breakMsg, reactMsg, isNewBar);
        }

        // =====================================================================
        // ALERT EMISSION (Pine: AlertSender calls, lines 578-600;
        //                 Pine: plotshape calls, lines 603-625)
        //
        // Frequency modes:
        //   "All"          → fire every tick on live bar
        //   "Once Per Bar" → fire once per bar (first tick)
        //   "Per Bar Close"→ fire on first tick of the NEW bar using previous bar's signal
        // =====================================================================

        private void EmitAlert(int index, int tlIdx,
            bool alertBreak, bool alertReact,
            ToggleOption breakSetting, ToggleOption reactSetting,
            string breakMsg, string reactMsg,
            bool isNewBar)
        {
            AlertFrequencyOption freq = Frequency;

            // --- Break ---
            if (breakSetting == ToggleOption.On)
            {
                bool fire = false;
                if (freq == AlertFrequencyOption.All)
                {
                    fire = alertBreak;
                }
                else if (freq == AlertFrequencyOption.OncePerBar)
                {
                    if (alertBreak && _tlStates[tlIdx].LastAlertBreakBar != index)
                    {
                        fire = true;
                        _tlStates[tlIdx].LastAlertBreakBar = index;
                    }
                }
                else if (freq == AlertFrequencyOption.PerBarClose)
                {
                    if (isNewBar && _prevBarBreakSignal[tlIdx])
                        fire = true;
                }

                if (fire)
                {
                    // Pine plotshape: DownTriangle above bar (break = bearish confirmation)
                    Chart.DrawIcon($"BRK_{tlIdx}_{index}", ChartIconType.DownTriangle,
                        index, Bars.HighPrices[index], Color.Red);
                    Print("{0} | {1} | Bar={2} | TZ={3}", AlertName, breakMsg, index, AlertTimeZone);
                }
            }

            // --- React ---
            if (reactSetting == ToggleOption.On)
            {
                bool fire = false;
                if (freq == AlertFrequencyOption.All)
                {
                    fire = alertReact;
                }
                else if (freq == AlertFrequencyOption.OncePerBar)
                {
                    if (alertReact && _tlStates[tlIdx].LastAlertReactBar != index)
                    {
                        fire = true;
                        _tlStates[tlIdx].LastAlertReactBar = index;
                    }
                }
                else if (freq == AlertFrequencyOption.PerBarClose)
                {
                    if (isNewBar && _prevBarReactSignal[tlIdx])
                        fire = true;
                }

                if (fire)
                {
                    // Pine plotshape: UpTriangle below bar (react = bullish bounce)
                    Chart.DrawIcon($"RCT_{tlIdx}_{index}", ChartIconType.UpTriangle,
                        index, Bars.LowPrices[index], Color.Green);
                    Print("{0} | {1} | Bar={2} | TZ={3}", AlertName, reactMsg, index, AlertTimeZone);
                }
            }

            // Save for "Per Bar Close" mode (checked on next bar's isNewBar tick)
            _prevBarBreakSignal[tlIdx] = alertBreak;
            _prevBarReactSignal[tlIdx] = alertReact;
        }

        // =====================================================================
        // MATH — Line price at arbitrary bar (mirrors Pine line.get_price)
        // Linear extrapolation/interpolation through two points (x0,y0),(x1,y1).
        // =====================================================================

        private static double LinePrice(int x0, double y0, int x1, double y1, int atBar)
        {
            if (x1 == x0) return y0;  // degenerate (vertical): return anchor price
            return y0 + (y1 - y0) * (double)(atBar - x0) / (x1 - x0);
        }

        // =====================================================================
        // COLOR PARSING
        // Accepts "#RRGGBB" (6 digits, fully opaque) or "#AARRGGBB" (8 digits
        // where first 2 hex digits are alpha, matching Pine's color format).
        // Falls back to fallbackHex if parsing fails.
        // =====================================================================

        private static Color ParseColor(string hex, string fallbackHex)
        {
            if (string.IsNullOrWhiteSpace(hex)) hex = fallbackHex;
            hex = hex.TrimStart('#').Trim();
            try
            {
                if (hex.Length == 8)
                {
                    // Pine AARRGGBB — first pair is alpha (0=transparent, ff=opaque)
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

            // Fallback
            string fb = fallbackHex.TrimStart('#');
            try
            {
                int r = Convert.ToInt32(fb.Substring(0, 2), 16);
                int g = Convert.ToInt32(fb.Substring(2, 2), 16);
                int b = Convert.ToInt32(fb.Substring(4, 2), 16);
                return Color.FromArgb(255, r, g, b);
            }
            catch
            {
                return Color.Gray;
            }
        }

        // =====================================================================
        // LINE STYLE PARSING (Pine: line.style_solid / dashed / dotted)
        // =====================================================================

        private static LineStyle ParseStyle(TrendLineStyleOption style)
        {
            switch (style)
            {
                case TrendLineStyleOption.Dashed:
                    return LineStyle.Lines;
                case TrendLineStyleOption.Dotted:
                    return LineStyle.Dots;
                default:
                    return LineStyle.Solid;
            }
        }

        private static TrendLineExtendOption ParseExtend(TrendLineExtendOption extend)
        {
            return extend;
        }

        private DateTime TimeOf(int index)
        {
            if (index < 0)
                index = 0;
            if (index >= Bars.Count)
                index = Bars.Count - 1;
            return Bars.OpenTimes[index];
        }

        private void ApplyLineExtension(ChartTrendLine line, int x0, double y0, int x1, double y1, int index, TrendLineExtendOption extendMode, bool active)
        {
            if (line == null || index < 0)
                return;

            bool infinityMode = extendMode == TrendLineExtendOption.Right ||
                                extendMode == TrendLineExtendOption.Both ||
                                extendMode == TrendLineExtendOption.Left;

            if (active && infinityMode)
            {
                line.ExtendToInfinity = true;
                return;
            }

            line.ExtendToInfinity = false;

            int endIndex = active
                ? (extendMode == TrendLineExtendOption.None ? index + 1 : index)
                : index;

            line.Time2 = TimeOf(endIndex);
            line.Y2 = LinePrice(x0, y0, x1, y1, endIndex);
        }
    }
}
