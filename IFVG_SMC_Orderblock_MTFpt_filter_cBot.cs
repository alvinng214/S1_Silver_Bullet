using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IFVG_SMC_Orderblock_MTFpt_filter_cBot : Robot
    {
        public enum ObFilter       { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum MtfLogicMode   { OR, AND }

        private sealed class BslPivot
        {
            public double Price;
            public int    BarIndex;
            public int    Type;
        }

        private sealed class BslPool
        {
            public double Price;
            public int    PivotIndex;
        }

        private sealed class SmcObRecord
        {
            public int      Index;
            public double   Top;
            public double   Bottom;
            public bool     Bullish;
            public bool     Internal;
            public int      StructureBreakIndex;
            public DateTime Time;
        }

        private sealed class MtfTfState
        {
            public Bars TfBars;
            public int PivotLen;
            public bool IsLowerTf;
            public int TfMinutes;
            public int LastProcessedTfBar = -1;
            public bool CurrentTrend;
            public bool HasTrend;
            public double LastPivotHigh = double.NaN;
            public double LastPivotLow = double.NaN;
            public double LastBrokenHigh = double.NaN;
            public double LastBrokenLow = double.NaN;
            public DateTime PivotHighTime = DateTime.MinValue;
            public DateTime PivotLowTime = DateTime.MinValue;
        }

        [Parameter("FVG Search Lookback (bars)", DefaultValue = 15, MinValue = 1, Group = "IFVG")]
        public int FvgGapBars { get; set; }

        [Parameter("Min FVG Size (pips)", DefaultValue = 0.0, MinValue = 0.0, Group = "IFVG")]
        public double MinFvgPips { get; set; }

        [Parameter("FVG Epsilon (price units)", DefaultValue = 0.0, MinValue = 0.0, Group = "IFVG")]
        public double FvgEpsilonPoints { get; set; }

        [Parameter("MA Period", DefaultValue = 21, MinValue = 1, Group = "IFVG")]
        public int MaPeriod { get; set; }

        [Parameter("MA Type (SMA / EMA)", DefaultValue = "EMA", Group = "IFVG")]
        public string MaType { get; set; }

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        [Parameter("Risk % per trade", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio", DefaultValue = 2.0, MinValue = 0.1, Step = 0.1, Group = "Risk Management")]
        public double RiskRewardRatio { get; set; }

        [Parameter("Max Open Positions", DefaultValue = 3, MinValue = 1, MaxValue = 100, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 1.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("Instance Name", DefaultValue = "IFVG_SMC_MTFpt_cBot", Group = "Risk Management")]
        public string InstanceName { get; set; }

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "SMC Filter — Swing Structure")]
        public int SmcSwingsLengthInput { get; set; }

        [Parameter("Order Block Filter", DefaultValue = ObFilter.Atr, Group = "SMC Filter — Order Blocks")]
        public ObFilter SmcOrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "SMC Filter — Order Blocks")]
        public int SmcObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "SMC Filter — Order Blocks")]
        public MitigationMode SmcOrderBlockMitigationInput { get; set; }

        [Parameter("Enable Filter 1 (Internal OB)", DefaultValue = true, Group = "Filter 1 — Internal OB")]
        public bool EnableFilter1 { get; set; }

        [Parameter("OB Touch Window — Internal (bars)", DefaultValue = 10, MinValue = 0, Group = "Filter 1 — Internal OB")]
        public int Filter1Lookback { get; set; }

        [Parameter("Enable Filter 2 (Swing OB)", DefaultValue = false, Group = "Filter 2 — Swing OB")]
        public bool EnableFilter2 { get; set; }

        [Parameter("OB Touch Window — Swing (bars)", DefaultValue = 10, MinValue = 0, Group = "Filter 2 — Swing OB")]
        public int Filter2Lookback { get; set; }

        [Parameter("Enable Min Bars From OB Origin", DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableMinBarsFromOrigin { get; set; }

        [Parameter("Min Bars — Internal OB", DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginInternal { get; set; }

        [Parameter("Min Bars — Swing OB", DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginSwing { get; set; }

        [Parameter("Enable ATR Distance Filter", DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableAtrDistanceFilter { get; set; }

        [Parameter("ATR Distance Multiplier", DefaultValue = 1.0, MinValue = 0.1, Step = 0.1, Group = "OB Quality Filters")]
        public double AtrDistanceMultiplier { get; set; }

        [Parameter("Enable MTF Trend Filter", DefaultValue = false, Group = "MTF Trend Filter — General")]
        public bool EnableMtfFilter { get; set; }

        [Parameter("Multi-TF Logic (OR / AND)", DefaultValue = MtfLogicMode.OR, Group = "MTF Trend Filter — General")]
        public MtfLogicMode MtfFilterLogic { get; set; }

        [Parameter("Enable TF1 Filter", DefaultValue = true, Group = "MTF Trend Filter — TF1")]
        public bool EnableMtfTf1 { get; set; }
        [Parameter("TF1 Timeframe", DefaultValue = "15", Group = "MTF Trend Filter — TF1")]
        public string MtfTimeframe1 { get; set; }
        [Parameter("TF1 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF1")]
        public int MtfPivotStrength1 { get; set; }
        [Parameter("TF1 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF1")]
        public bool MtfIsLowerTf1 { get; set; }

        [Parameter("Enable TF2 Filter", DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool EnableMtfTf2 { get; set; }
        [Parameter("TF2 Timeframe", DefaultValue = "30", Group = "MTF Trend Filter — TF2")]
        public string MtfTimeframe2 { get; set; }
        [Parameter("TF2 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF2")]
        public int MtfPivotStrength2 { get; set; }
        [Parameter("TF2 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool MtfIsLowerTf2 { get; set; }

        [Parameter("Enable TF3 Filter", DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool EnableMtfTf3 { get; set; }
        [Parameter("TF3 Timeframe", DefaultValue = "60", Group = "MTF Trend Filter — TF3")]
        public string MtfTimeframe3 { get; set; }
        [Parameter("TF3 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF3")]
        public int MtfPivotStrength3 { get; set; }
        [Parameter("TF3 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool MtfIsLowerTf3 { get; set; }

        [Parameter("Enable TF4 Filter", DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool EnableMtfTf4 { get; set; }
        [Parameter("TF4 Timeframe", DefaultValue = "240", Group = "MTF Trend Filter — TF4")]
        public string MtfTimeframe4 { get; set; }
        [Parameter("TF4 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF4")]
        public int MtfPivotStrength4 { get; set; }
        [Parameter("TF4 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool MtfIsLowerTf4 { get; set; }

        private const int MaxBslPivots = 10;

        private IndicatorDataSeries _maSeries;
        private SimpleMovingAverage _sma;
        private ExponentialMovingAverage _ema;

        private readonly LinkedList<BslPivot> _bslPivots = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool> _bslBuysidePools = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool> _bslSellsidePools = new LinkedList<BslPool>();
        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

        private readonly List<SmcObRecord> _smcInternalBullObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcInternalBearObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBullObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBearObs = new List<SmcObRecord>();
        private readonly List<double> _parsedHighs = new List<double>();
        private readonly List<double> _parsedLows = new List<double>();
        private readonly List<DateTime> _times = new List<DateTime>();
        private const int SmcAtrPeriod = 200;
        private double _smcAtrWilder = double.NaN;
        private double _smcAtrWilderSum = 0.0;
        private double _smcCumTr = 0.0;
        private int _swingLeg;
        private int _swingTrend;
        private double _lastSwingHigh = double.NaN;
        private double _lastSwingLow = double.NaN;
        private int _lastSwingHighIndex = -1;
        private int _lastSwingLowIndex = -1;
        private bool _swingHighCrossed;
        private bool _swingLowCrossed;
        private int _internalLeg;
        private int _internalTrend;
        private double _internalHighLevel = double.NaN;
        private double _internalLowLevel = double.NaN;
        private int _internalHighIndex = -1;
        private int _internalLowIndex = -1;
        private bool _internalHighCrossed;
        private bool _internalLowCrossed;
        private int _smcObIdCounter;
        private const int MaxSmcObs = 500;
        private int _lastParsedIndex = -1;
        private int _smcWarmup;

        private MtfTfState _mtfState1;
        private MtfTfState _mtfState2;
        private MtfTfState _mtfState3;
        private MtfTfState _mtfState4;

        private int _lastProcessed = -1;

        protected override void OnStart()
        {
            _smcWarmup = Math.Max(SmcSwingsLengthInput, 5) + 5;

            _maSeries = CreateDataSeries();
            _sma = Indicators.SimpleMovingAverage(Bars.ClosePrices, MaPeriod);
            _ema = Indicators.ExponentialMovingAverage(Bars.ClosePrices, MaPeriod);

            if (EnableMtfFilter)
            {
                _mtfState1 = EnableMtfTf1 ? MtfCreateState(MtfTimeframe1, MtfPivotStrength1, MtfIsLowerTf1) : null;
                _mtfState2 = EnableMtfTf2 ? MtfCreateState(MtfTimeframe2, MtfPivotStrength2, MtfIsLowerTf2) : null;
                _mtfState3 = EnableMtfTf3 ? MtfCreateState(MtfTimeframe3, MtfPivotStrength3, MtfIsLowerTf3) : null;
                _mtfState4 = EnableMtfTf4 ? MtfCreateState(MtfTimeframe4, MtfPivotStrength4, MtfIsLowerTf4) : null;

                Print("MTF filter ON. Logic={0}. TF1={1}({2}) TF2={3}({4}) TF3={5}({6}) TF4={7}({8})",
                    MtfFilterLogic,
                    EnableMtfTf1, MtfTimeframe1, EnableMtfTf2, MtfTimeframe2,
                    EnableMtfTf3, MtfTimeframe3, EnableMtfTf4, MtfTimeframe4);
            }

            Print("IFVG_SMC_Orderblock_MTFpt_filter_cBot started. MaxPositions={0}, Risk={1}%, RR={2}",
                  MaxOpenPositions, RiskPercent, RiskRewardRatio);
        }

        protected override void OnStop()
        {
            Print("IFVG_SMC_Orderblock_MTFpt_filter_cBot stopped.");
        }

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;

            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);
                RunSmcFilter(i);
            }
            _lastProcessed = signalBar;

            if (EnableMtfFilter)
            {
                var chartTime = Bars.OpenTimes[signalBar];
                MtfAdvanceState(_mtfState1, chartTime);
                MtfAdvanceState(_mtfState2, chartTime);
                MtfAdvanceState(_mtfState3, chartTime);
                MtfAdvanceState(_mtfState4, chartTime);
            }

            if (signalBar < Math.Max(PivotLeft + PivotRight + 1, 3)) return;

            var maValue = CalculateMa(signalBar);
            var signalDir = DetectIfvgSignal(signalBar, maValue);
            if (signalDir == 0) return;

            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached — IFVG signal skipped.", signalBar, MaxOpenPositions);
                return;
            }

            if (!CheckFilters(signalBar, signalDir)) return;
            if (!CheckMtfFilter(signalDir > 0, signalBar)) return;

            if (signalDir == 1)
                TryEnterLong(signalBar);
            else
                TryEnterShort(signalBar);
        }

        private double CalculateMa(int index)
        {
            _maSeries[index] = string.Equals(MaType, "SMA", StringComparison.OrdinalIgnoreCase)
                ? _sma.Result[index]
                : _ema.Result[index];
            return _maSeries[index];
        }

        private int DetectIfvgSignal(int index, double maValue)
        {
            var minSizeValue = MinFvgPips * Symbol.PipSize;

            for (var i = 1; i <= FvgGapBars; i++)
            {
                var fvgType = DetectFvg(index, i, FvgEpsilonPoints);
                if (fvgType == 0) continue;

                int signalDir;
                if (TryProcessFvgCandidate(index, i, fvgType, minSizeValue, maValue, out signalDir))
                    return signalDir;
            }

            return 0;
        }

        private int DetectFvg(int currentIndex, int idx, double epsVal)
        {
            if (idx + 2 > currentIndex) return 0;

            var h2 = Bars.HighPrices[currentIndex - (idx + 2)];
            var l2 = Bars.LowPrices[currentIndex - (idx + 2)];
            var lt = Bars.LowPrices[currentIndex - idx];
            var ht = Bars.HighPrices[currentIndex - idx];

            if (lt > h2 - epsVal) return 1;
            if (ht < l2 + epsVal) return -1;
            return 0;
        }

        private bool TryProcessFvgCandidate(int index, int i, int fvgType, double minSizeValue, double maValue, out int signalDir)
        {
            signalDir = 0;

            var isBearishGap = fvgType == 1;
            var gapLow = isBearishGap ? Bars.HighPrices[index - (i + 2)] : Bars.HighPrices[index - i];
            var gapHigh = isBearishGap ? Bars.LowPrices[index - i] : Bars.LowPrices[index - (i + 2)];

            if ((gapHigh - gapLow) < minSizeValue) return false;

            if (i > 1)
            {
                for (var k = i - 1; k >= 1; k--)
                {
                    var close = Bars.ClosePrices[index - k];
                    if ((isBearishGap && close < gapLow) || (!isBearishGap && close > gapHigh))
                        return false;
                }
            }

            var breakout = isBearishGap ? Bars.ClosePrices[index] < gapLow : Bars.ClosePrices[index] > gapHigh;
            if (!breakout) return false;

            var maReady = !double.IsNaN(maValue) && !double.IsNaN(_maSeries[index - 1]);
            var maCondition = isBearishGap
                ? maReady && maValue < _maSeries[index - 1] && Bars.ClosePrices[index] < maValue
                : maReady && maValue > _maSeries[index - 1] && Bars.ClosePrices[index] > maValue;
            if (!maCondition) return false;

            signalDir = isBearishGap ? -1 : 1;
            return true;
        }

        private bool CheckFilters(int index, int cond)
        {
            var isBull = cond > 0;
            bool? f1 = null;
            bool? f2 = null;

            if (EnableFilter1)
            {
                var pool = isBull ? _smcInternalBullObs : _smcInternalBearObs;
                f1 = HasActiveTouchInLookback(pool, index, Filter1Lookback, isBull);
            }
            if (EnableFilter2)
            {
                var pool = isBull ? _smcSwingBullObs : _smcSwingBearObs;
                f2 = HasActiveTouchInLookback(pool, index, Filter2Lookback, isBull);
            }

            if (f1.HasValue && f2.HasValue)
            {
                if (!f1.Value && !f2.Value)
                {
                    Print("[Filter BLOCKED] {0} at bar {1}: no unmitigated internal or swing OB touched within {2}/{3} bars of IFVG signal.",
                          isBull ? "Long" : "Short", index, Filter1Lookback, Filter2Lookback);
                    return false;
                }
                return true;
            }
            if (f1.HasValue && !f1.Value)
            {
                Print("[Filter1 BLOCKED] {0} at bar {1}: no unmitigated internal OB touched within {2} bars of IFVG signal.",
                      isBull ? "Long" : "Short", index, Filter1Lookback);
                return false;
            }
            if (f2.HasValue && !f2.Value)
            {
                Print("[Filter2 BLOCKED] {0} at bar {1}: no unmitigated swing OB touched within {2} bars of IFVG signal.",
                      isBull ? "Long" : "Short", index, Filter2Lookback);
                return false;
            }
            return true;
        }

        private bool HasActiveTouchInLookback(List<SmcObRecord> pool, int signalBar, int lookback, bool bullish)
        {
            if (pool.Count == 0) return false;

            var currentAtr = double.IsNaN(_smcAtrWilder) ? 0.0 : _smcAtrWilder;
            var closeNow = Bars.ClosePrices[signalBar];

            foreach (var ob in pool)
            {
                if (EnableMinBarsFromOrigin)
                {
                    var minBars = ob.Internal ? MinBarsFromOriginInternal : MinBarsFromOriginSwing;
                    if (signalBar - ob.Index < minBars) continue;
                }

                if (EnableAtrDistanceFilter && AtrDistanceMultiplier > 0 && currentAtr > 0)
                {
                    var advance = bullish ? closeNow - ob.Top : ob.Bottom - closeNow;
                    if (advance < AtrDistanceMultiplier * currentAtr) continue;
                }

                var lastTouchBar = -1;
                for (var b = ob.StructureBreakIndex + 1; b <= signalBar; b++)
                {
                    if (bullish && Bars.LowPrices[b] <= ob.Top) lastTouchBar = b;
                    if (!bullish && Bars.HighPrices[b] >= ob.Bottom) lastTouchBar = b;
                }

                if (lastTouchBar < 0) continue;
                if (signalBar - lastTouchBar <= lookback) return true;
            }
            return false;
        }

        private void RunSmcFilter(int index)
        {
            for (var i = _lastParsedIndex + 1; i <= index; i++)
                UpdateSmcParsedArrays(i);
            _lastParsedIndex = index;

            if (index < _smcWarmup) return;

            const int iLen = 5;
            var sLen = Math.Max(5, SmcSwingsLengthInput);

            var internalLegNow = ComputeLeg(index, iLen, _internalLeg);
            var internalDc = internalLegNow - _internalLeg;
            if (internalDc != 0)
            {
                if (internalDc == 1)
                { _internalLowLevel = Bars.LowPrices[index - iLen]; _internalLowIndex = index - iLen; _internalLowCrossed = false; }
                else
                { _internalHighLevel = Bars.HighPrices[index - iLen]; _internalHighIndex = index - iLen; _internalHighCrossed = false; }
            }
            _internalLeg = internalLegNow;

            var swingLegNow = ComputeLeg(index, sLen, _swingLeg);
            var swingDc = swingLegNow - _swingLeg;
            if (swingDc != 0)
            {
                if (swingDc == 1)
                { _lastSwingLow = Bars.LowPrices[index - sLen]; _lastSwingLowIndex = index - sLen; _swingLowCrossed = false; }
                else
                { _lastSwingHigh = Bars.HighPrices[index - sLen]; _lastSwingHighIndex = index - sLen; _swingHighCrossed = false; }
            }
            _swingLeg = swingLegNow;

            var close = Bars.ClosePrices[index];

            if (!double.IsNaN(_internalHighLevel) && !_internalHighCrossed && close > _internalHighLevel)
            { _internalHighCrossed = true; _internalTrend = 1; StoreSmcOrderBlock(_internalHighIndex, true, 1, index); }
            if (!double.IsNaN(_internalLowLevel) && !_internalLowCrossed && close < _internalLowLevel)
            { _internalLowCrossed = true; _internalTrend = -1; StoreSmcOrderBlock(_internalLowIndex, true, -1, index); }
            if (!double.IsNaN(_lastSwingHigh) && !_swingHighCrossed && close > _lastSwingHigh)
            { _swingHighCrossed = true; _swingTrend = 1; StoreSmcOrderBlock(_lastSwingHighIndex, false, 1, index); }
            if (!double.IsNaN(_lastSwingLow) && !_swingLowCrossed && close < _lastSwingLow)
            { _swingLowCrossed = true; _swingTrend = -1; StoreSmcOrderBlock(_lastSwingLowIndex, false, -1, index); }

            ManageSmcObList(_smcInternalBullObs, index, true);
            ManageSmcObList(_smcInternalBearObs, index, false);
            ManageSmcObList(_smcSwingBullObs, index, true);
            ManageSmcObList(_smcSwingBearObs, index, false);
        }

        private void UpdateSmcParsedArrays(int index)
        {
            double tr;
            if (index == 0)
            {
                _smcCumTr = 0; _smcAtrWilderSum = 0; _smcAtrWilder = double.NaN;
                tr = Bars.HighPrices[0] - Bars.LowPrices[0];
            }
            else
            {
                var pc = Bars.ClosePrices[index - 1];
                tr = Math.Max(Bars.HighPrices[index] - Bars.LowPrices[index],
                     Math.Max(Math.Abs(Bars.HighPrices[index] - pc),
                              Math.Abs(Bars.LowPrices[index] - pc)));
                _smcCumTr += tr;

                if (index < SmcAtrPeriod)
                { _smcAtrWilderSum += tr; _smcAtrWilder = double.NaN; }
                else if (index == SmcAtrPeriod)
                { _smcAtrWilderSum += tr; _smcAtrWilder = _smcAtrWilderSum / SmcAtrPeriod; }
                else
                { _smcAtrWilder = (_smcAtrWilder * (SmcAtrPeriod - 1) + tr) / SmcAtrPeriod; }
            }

            var vm = SmcOrderBlockFilterInput == ObFilter.Atr
                ? (double.IsNaN(_smcAtrWilder) ? double.MaxValue : _smcAtrWilder)
                : (_smcCumTr / Math.Max(1, index));

            var hv = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * vm;
            _parsedHighs.Add(hv ? Bars.LowPrices[index] : Bars.HighPrices[index]);
            _parsedLows.Add(hv ? Bars.HighPrices[index] : Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        private void StoreSmcOrderBlock(int pivotIndex, bool isInternal, int bias, int breakIndex)
        {
            if (pivotIndex < 0 || pivotIndex >= breakIndex || breakIndex >= _parsedHighs.Count)
                return;

            var parsedIndex = pivotIndex;
            if (bias == -1)
            {
                var maxV = double.MinValue;
                for (var i = pivotIndex; i < breakIndex; i++)
                    if (_parsedHighs[i] > maxV) { maxV = _parsedHighs[i]; parsedIndex = i; }
            }
            else
            {
                var minV = double.MaxValue;
                for (var i = pivotIndex; i < breakIndex; i++)
                    if (_parsedLows[i] < minV) { minV = _parsedLows[i]; parsedIndex = i; }
            }

            var bullish = bias == 1;
            var ob = new SmcObRecord
            {
                Index = parsedIndex,
                Top = _parsedHighs[parsedIndex],
                Bottom = _parsedLows[parsedIndex],
                Bullish = bullish,
                Internal = isInternal,
                Time = _times[parsedIndex],
                StructureBreakIndex = breakIndex
            };
            _smcObIdCounter++;

            var list = isInternal
                ? (bullish ? _smcInternalBullObs : _smcInternalBearObs)
                : (bullish ? _smcSwingBullObs : _smcSwingBearObs);

            if (list.Count >= MaxSmcObs) list.RemoveAt(list.Count - 1);
            list.Insert(0, ob);
        }

        private void ManageSmcObList(List<SmcObRecord> list, int index, bool bullish)
        {
            var bearSrc = SmcOrderBlockMitigationInput == MitigationMode.Close
                ? Bars.ClosePrices[index] : Bars.HighPrices[index];
            var bullSrc = SmcOrderBlockMitigationInput == MitigationMode.Close
                ? Bars.ClosePrices[index] : Bars.LowPrices[index];

            for (var i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];
                if ((bullish && bullSrc < ob.Bottom) || (!bullish && bearSrc > ob.Top))
                    list.RemoveAt(i);
            }
        }

        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;
            var refHigh = Bars.HighPrices[index - size];
            var refLow = Bars.LowPrices[index - size];
            var highest = double.MinValue;
            var lowest = double.MaxValue;
            var start = Math.Max(0, index - size + 1);
            for (var i = start; i <= index; i++)
            {
                if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i];
                if (Bars.LowPrices[i] < lowest) lowest = Bars.LowPrices[i];
            }
            if (refHigh > highest) return 0;
            if (refLow < lowest) return 1;
            return previousLeg;
        }

        private void RunBslSsl(int index)
        {
            BslDetectAndStoreConfirmedPivots(index);
            BslAddPoolFromNewPivot(index);
            BslClearMitigated(index);
            _bslCurrentBsl = _bslBuysidePools.First != null ? _bslBuysidePools.First.Value.Price : double.NaN;
            _bslCurrentSsl = _bslSellsidePools.First != null ? _bslSellsidePools.First.Value.Price : double.NaN;
        }

        private void BslDetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;
            if (pivotIndex <= 0) return;
            int leftStart = pivotIndex - PivotLeft;
            int rightEnd = pivotIndex + PivotRight;
            if (leftStart < 0 || rightEnd >= Bars.Count) return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow = Bars.LowPrices[pivotIndex];
            if (BslIsPivotHigh(candidateHigh, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type = 1 });
            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow, BarIndex = pivotIndex, Type = -1 });
        }

        private bool BslIsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++) if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool BslIsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++) if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type && Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1) return;
            }
            _bslPivots.AddFirst(p);
            while (_bslPivots.Count > MaxBslPivots) _bslPivots.RemoveLast();
        }

        private void BslAddPoolFromNewPivot(int currentIndex)
        {
            int confirmedIdx = currentIndex - PivotRight;
            foreach (var pivot in _bslPivots)
            {
                if (pivot.BarIndex != confirmedIdx) continue;
                var pool = new BslPool { Price = pivot.Price, PivotIndex = pivot.BarIndex };
                if (pivot.Type == 1) _bslBuysidePools.AddFirst(pool);
                if (pivot.Type == -1) _bslSellsidePools.AddFirst(pool);
            }
        }

        private void BslClearMitigated(int index)
        {
            var node = _bslSellsidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.LowPrices[index] <= node.Value.Price) _bslSellsidePools.Remove(node);
                node = next;
            }
            node = _bslBuysidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.HighPrices[index] >= node.Value.Price) _bslBuysidePools.Remove(node);
                node = next;
            }
        }

        private void TryEnterLong(int signalBar)
        {
            double entry = Symbol.Ask;
            double sslLevel = _bslCurrentSsl;
            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            { Print("Bar {0}: LONG skipped – SSL unavailable.", signalBar); return; }
            if (sslLevel >= entry)
            { Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.", signalBar, sslLevel, entry); return; }
            double slPips = (entry - sslLevel) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: LONG  | Entry={1:F5} | SSL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, sslLevel, slPips, slPips * RiskRewardRatio, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, InstanceName, slPips, slPips * RiskRewardRatio);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry = Symbol.Bid;
            double bslLevel = _bslCurrentBsl;
            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            { Print("Bar {0}: SHORT skipped – BSL unavailable.", signalBar); return; }
            if (bslLevel <= entry)
            { Print("Bar {0}: SHORT skipped – BSL {1:F5} not above entry {2:F5}.", signalBar, bslLevel, entry); return; }
            double slPips = (bslLevel - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: SHORT | Entry={1:F5} | BSL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, bslLevel, slPips, slPips * RiskRewardRatio, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, InstanceName, slPips, slPips * RiskRewardRatio);
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1} < min {3:F1} pips.", signalBar, direction, slPips, MinSlPips); return false; }
            if (slPips > MaxSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1} > max {3:F1} pips.", signalBar, direction, slPips, MaxSlPips); return false; }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0) return 0;
            double volume = Symbol.NormalizeVolumeInUnits(Symbol.VolumeForFixedRisk(riskAmount, slPips), RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return 0;
            if (volume > Symbol.VolumeInUnitsMax) volume = Symbol.VolumeInUnitsMax;
            return volume;
        }

        private MtfTfState MtfCreateState(string tfInput, int pivotStrength, bool isLowerTf)
        {
            var tf = MtfParseTimeFrame(tfInput);
            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            return new MtfTfState
            {
                TfBars = bars,
                PivotLen = Math.Max(1, pivotStrength),
                IsLowerTf = isLowerTf,
                TfMinutes = MtfTfMinutes(tf)
            };
        }

        private void MtfAdvanceState(MtfTfState s, DateTime chartTime)
        {
            if (s == null) return;

            var tfBarIndex = MtfResolveTfBar(s, chartTime);
            if (tfBarIndex < 0) return;

            for (var i = s.LastProcessedTfBar + 1; i <= tfBarIndex; i++)
                MtfProcessCalcBar(s, i);

            if (tfBarIndex > s.LastProcessedTfBar)
                s.LastProcessedTfBar = tfBarIndex;
        }

        private void MtfProcessCalcBar(MtfTfState s, int tfBarIndex)
        {
            var bars = s.TfBars;
            var prevLastPivotHigh = s.LastPivotHigh;
            var prevLastPivotLow = s.LastPivotLow;

            if (tfBarIndex >= s.PivotLen * 2)
            {
                var pivotIdx = tfBarIndex - s.PivotLen;

                if (MtfIsPivotHigh(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.HighPrices[pivotIdx];
                    s.LastPivotHigh = s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotHigh) ? pp : Math.Max(pp, s.LastPivotHigh))
                        : pp;
                    if (s.LastPivotHigh != prevLastPivotHigh)
                        s.PivotHighTime = bars.OpenTimes[pivotIdx];
                }

                if (MtfIsPivotLow(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.LowPrices[pivotIdx];
                    s.LastPivotLow = !s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotLow) ? pp : Math.Min(pp, s.LastPivotLow))
                        : pp;
                    if (s.LastPivotLow != prevLastPivotLow)
                        s.PivotLowTime = bars.OpenTimes[pivotIdx];
                }
            }

            var close = bars.ClosePrices[tfBarIndex];
            var prevClose = tfBarIndex > 0 ? bars.ClosePrices[tfBarIndex - 1] : close;

            if (!double.IsNaN(s.LastPivotHigh) && !double.IsNaN(prevLastPivotHigh))
            {
                if (prevClose <= prevLastPivotHigh && close > s.LastPivotHigh)
                {
                    s.CurrentTrend = true;
                    s.HasTrend = true;
                    s.LastBrokenHigh = s.LastPivotHigh;
                    s.LastBrokenLow = double.NaN;
                }
            }

            if (!double.IsNaN(s.LastPivotLow) && !double.IsNaN(prevLastPivotLow))
            {
                if (prevClose >= prevLastPivotLow && close < s.LastPivotLow)
                {
                    s.CurrentTrend = false;
                    s.HasTrend = true;
                    s.LastBrokenLow = s.LastPivotLow;
                    s.LastBrokenHigh = double.NaN;
                }
            }
        }

        private int MtfResolveTfBar(MtfTfState s, DateTime chartTime)
        {
            if (!s.IsLowerTf)
            {
                var adjusted = chartTime.AddMinutes(-(s.TfMinutes > 0 ? s.TfMinutes : 1));
                return MtfFindAtOrBefore(s.TfBars, adjusted);
            }

            var chartBarIndex = MtfFindAtOrBefore(Bars, chartTime);
            if (chartBarIndex < 0) return -1;

            var chartOpen = Bars.OpenTimes[chartBarIndex];
            DateTime chartNextOpen;
            if (chartBarIndex + 1 < Bars.Count)
                chartNextOpen = Bars.OpenTimes[chartBarIndex + 1];
            else
            {
                var m = MtfTfMinutes(Bars.TimeFrame);
                chartNextOpen = chartOpen.AddMinutes(m > 0 ? m : 1);
            }

            var first = MtfFindAtOrAfter(s.TfBars, chartOpen);
            if (first < 0) return MtfFindAtOrBefore(s.TfBars, chartTime);
            if (s.TfBars.OpenTimes[first] >= chartNextOpen)
                return MtfFindAtOrBefore(s.TfBars, chartTime);
            return first;
        }

        private bool CheckMtfFilter(bool isLong, int signalBar)
        {
            if (!EnableMtfFilter) return true;

            var isAnd = MtfFilterLogic == MtfLogicMode.AND;
            var enabled = 0;
            var passing = 0;

            if (_mtfState1 != null)
            {
                enabled++;
                if (!_mtfState1.HasTrend || _mtfState1.CurrentTrend == isLong) passing++;
            }
            if (_mtfState2 != null)
            {
                enabled++;
                if (!_mtfState2.HasTrend || _mtfState2.CurrentTrend == isLong) passing++;
            }
            if (_mtfState3 != null)
            {
                enabled++;
                if (!_mtfState3.HasTrend || _mtfState3.CurrentTrend == isLong) passing++;
            }
            if (_mtfState4 != null)
            {
                enabled++;
                if (!_mtfState4.HasTrend || _mtfState4.CurrentTrend == isLong) passing++;
            }

            if (enabled == 0) return true;

            var result = isAnd ? (passing == enabled) : (passing > 0);

            if (!result)
            {
                Print("[MTF BLOCKED] {0} bar={1} passing={2}/{3} logic={4} | TF1:{5}({6}) TF2:{7}({8}) TF3:{9}({10}) TF4:{11}({12})",
                    isLong ? "Long" : "Short", signalBar, passing, enabled,
                    isAnd ? "AND" : "OR",
                    _mtfState1 != null ? (_mtfState1.HasTrend ? (_mtfState1.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState1 != null ? _mtfState1.LastProcessedTfBar : -1,
                    _mtfState2 != null ? (_mtfState2.HasTrend ? (_mtfState2.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState2 != null ? _mtfState2.LastProcessedTfBar : -1,
                    _mtfState3 != null ? (_mtfState3.HasTrend ? (_mtfState3.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState3 != null ? _mtfState3.LastProcessedTfBar : -1,
                    _mtfState4 != null ? (_mtfState4.HasTrend ? (_mtfState4.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState4 != null ? _mtfState4.LastProcessedTfBar : -1);
            }

            return result;
        }

        private static bool MtfIsPivotHigh(Bars bars, int idx, int len)
        {
            var left = idx - len;
            var right = idx + len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.HighPrices[idx];
            for (var i = left; i <= right; i++)
                if (i != idx && bars.HighPrices[i] >= pivot) return false;
            return true;
        }

        private static bool MtfIsPivotLow(Bars bars, int idx, int len)
        {
            var left = idx - len;
            var right = idx + len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.LowPrices[idx];
            for (var i = left; i <= right; i++)
                if (i != idx && bars.LowPrices[i] <= pivot) return false;
            return true;
        }

        private static int MtfFindAtOrBefore(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                var midTime = bars.OpenTimes[mid];
                if (midTime == time) return mid;
                else if (midTime < time) lo = mid + 1;
                else hi = mid - 1;
            }
            return hi;
        }

        private static int MtfFindAtOrAfter(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1; var ans = -1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                if (bars.OpenTimes[mid] >= time) { ans = mid; hi = mid - 1; }
                else lo = mid + 1;
            }
            return ans;
        }

        private static TimeFrame MtfParseTimeFrame(string text)
        {
            switch ((text ?? string.Empty).Trim().ToUpperInvariant())
            {
                case "1": return TimeFrame.Minute;
                case "2": return TimeFrame.Minute2;
                case "3": return TimeFrame.Minute3;
                case "4": return TimeFrame.Minute4;
                case "5": return TimeFrame.Minute5;
                case "10": return TimeFrame.Minute10;
                case "15": return TimeFrame.Minute15;
                case "30": return TimeFrame.Minute30;
                case "45": return TimeFrame.Minute45;
                case "60":
                case "1H": return TimeFrame.Hour;
                case "120":
                case "2H": return TimeFrame.Hour2;
                case "240":
                case "4H": return TimeFrame.Hour4;
                case "480":
                case "8H": return TimeFrame.Hour8;
                case "720":
                case "12H": return TimeFrame.Hour12;
                case "D":
                case "1D": return TimeFrame.Daily;
                case "W":
                case "1W": return TimeFrame.Weekly;
                case "M":
                case "1M": return TimeFrame.Monthly;
                default: return TimeFrame.Minute15;
            }
        }

        private static int MtfTfMinutes(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute) return 1;
            if (tf == TimeFrame.Minute2) return 2;
            if (tf == TimeFrame.Minute3) return 3;
            if (tf == TimeFrame.Minute4) return 4;
            if (tf == TimeFrame.Minute5) return 5;
            if (tf == TimeFrame.Minute10) return 10;
            if (tf == TimeFrame.Minute15) return 15;
            if (tf == TimeFrame.Minute30) return 30;
            if (tf == TimeFrame.Minute45) return 45;
            if (tf == TimeFrame.Hour) return 60;
            if (tf == TimeFrame.Hour2) return 120;
            if (tf == TimeFrame.Hour4) return 240;
            if (tf == TimeFrame.Hour8) return 480;
            if (tf == TimeFrame.Hour12) return 720;
            if (tf == TimeFrame.Daily) return 1440;
            if (tf == TimeFrame.Weekly) return 10080;
            if (tf == TimeFrame.Monthly) return 43200;
            return 0;
        }
    }
}
